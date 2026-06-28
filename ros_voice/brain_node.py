#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command          (std_msgs/String) — 用户指令文本。
订阅 /vision/scene_objects    (std_msgs/String) — 机械臂视觉返回的当前可见物体(JSON)。
订阅 /vision/emotion_context  (std_msgs/String) — 当前情绪(JSON)。
发布 /command                 (std_msgs/String) — JSON 指令数组。

设计要点：
- scene_objects / emotion_context 以约 0.5s/帧 的频率刷新（无物体或无人脸时不发新数据）。
  二者本身不触发任何动作，只把"当前状态"缓存下来，待用户下一次语音指令到来时，
  作为视觉/情绪上下文一并喂给 LLM。
- 为去除偶发异常帧（例如夹在一连串"开心"中的一帧"伤心"），缓存最近 3 秒的数据做
  滑动窗口众数投票去抖。窗口足够短，对真实变化（物体被拿走、情绪切换）的反应延迟
  最多约 1.5 秒，无需额外加权：
    · 情绪 → 取窗口内众数（需占多数且 >=2 帧）作为稳定情绪，单帧异常被丢弃。
      （情绪默认只有一个人，不计数量。）
    · 物体 → 仅保留在窗口内出现帧数达标(>=50%)的物体，闪烁误检被滤除。
      支持多个不同物体、且每类可有多个（如 2 个苹果 + 3 个香蕉），按类分别计数，
      数量取该类各帧出现数量的众数。
- 唯一例外：稳定情绪为负面(intervention_required=true)时，主动触发一次安抚对话。
- 数据靠 3 秒窗口自然老化，无需保留历史。
"""
import json
import math
import time
import queue
import threading
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module.pipeline import process_command
from voice_brain_module.llm import generate_response
from voice_brain_module.tts import stream_play


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")
        self._instr_pub = self.create_publisher(String, "/command", 10)
        self.create_subscription(String, "/voice/command", self._on_command, 10)

        # ── 视觉/情绪话题订阅（仅缓存，不直接触发动作）─────────
        self.create_subscription(String, "/vision/scene_objects",
                                 self._on_scene_objects, 10)
        self.create_subscription(String, "/vision/emotion_context",
                                 self._on_emotion_context, 10)

        # ── 滑动窗口缓存（最近 3 秒，约 6 帧，用于去除异常帧）──
        # 窗口足够短，对变化（物体被拿走、情绪切换）的反应延迟最多约 1.5 秒，
        # 直接用帧数众数投票即可，无需额外加权。
        self._window_sec = 3.0      # 缓存时长（窗口越短，对变化反应越快）
        self._stable_ratio = 0.5    # 物体/情绪在窗口内出现比例阈值
        self._scene_window = deque()    # [(stamp, data), ...]
        self._emotion_window = deque()  # [(stamp, data), ...]
        self._lock = threading.Lock()   # 保护窗口跨线程访问

        # ── 情绪干预冷却 ─────────────────────────────────────
        self._last_intervention_time = 0.0
        self._intervention_cooldown = 30.0  # 两次干预最小间隔(秒)

        self._work_q = queue.Queue()
        threading.Thread(target=self._work_loop, daemon=True).start()

        self.get_logger().info(
            "brain_node 就绪，等待 /voice/command 及 /vision/* 话题")

    # ── 窗口工具（线程安全）──────────────────────────────────

    def _push(self, window: deque, data: dict):
        """追加一帧并淘汰超出窗口时长的旧帧。"""
        now = time.time()
        with self._lock:
            window.append((now, data))
            while window and (now - window[0][0]) > self._window_sec:
                window.popleft()

    def _snapshot(self, window: deque) -> list:
        """淘汰过期帧后返回窗口快照（避免迭代时被其它线程修改）。"""
        now = time.time()
        with self._lock:
            while window and (now - window[0][0]) > self._window_sec:
                window.popleft()
            return list(window)

    def _stable_threshold(self, total: int) -> int:
        """窗口内被认定为"稳定"所需的最小出现帧数：至少 2 帧且达到比例阈值。"""
        return max(2, math.ceil(total * self._stable_ratio))

    # ── 视觉话题回调（仅缓存）─────────────────────────────────

    def _on_scene_objects(self, msg: String):
        """缓存机械臂视觉返回的当前可见物体，不触发任何动作。"""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("scene_objects JSON 解析失败")
            return
        self._push(self._scene_window, data)

    def _on_emotion_context(self, msg: String):
        """缓存情绪帧。基于最近窗口内的稳定情绪决定是否主动安抚。"""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("emotion_context JSON 解析失败")
            return
        self._push(self._emotion_window, data)
        stable = self._stable_emotion()
        if stable:
            self._check_emotion_intervention(stable)

    # ── 滑动窗口去抖 ─────────────────────────────────────────

    def _stable_emotion(self):
        """
        对最近窗口内的情绪做众数投票，去除偶发异常帧。
        胜出情绪需占多数且出现 >=2 帧（防止单帧异常在帧数很少时被误判为稳定）。
        返回代表性情绪 data（含 emotion_zh / intervention_required），不稳定则返回 None。
        """
        frames = self._snapshot(self._emotion_window)
        emotions = [d.get("emotion") for (_, d) in frames if d.get("emotion")]
        if not emotions:
            return None
        dominant, count = Counter(emotions).most_common(1)[0]
        if count < self._stable_threshold(len(emotions)):
            return None  # 没有占多数的情绪，视为不稳定（如正处于情绪切换/抖动）
        # 取窗口中最近一条该情绪的 data 作为代表
        for (_, d) in reversed(frames):
            if d.get("emotion") == dominant:
                return d
        return None

    def _stable_objects(self):
        """
        对最近窗口内的物体检测做帧频统计，仅保留出现比例达标的物体，去除闪烁误检。
        某类被认定为稳定需占窗口 >=50% 的帧且出现 >=2 帧。
        支持同类多个（如 2 个苹果）：每帧按 class_name 计数，代表数量取各帧数量的众数。
        每个稳定物体附带 "count" 字段表示代表数量；返回值含 "count" 表示物体总数。
        返回 {"objects":[...], "graspable":[...], "dialogue_only":[...], "count": N} 或 None。
        """
        frames = self._snapshot(self._scene_window)
        if not frames:
            return None
        total = len(frames)
        appear     = {}   # class_name -> 出现帧数（用于稳定判定）
        per_counts = {}   # class_name -> [每帧该类数量, ...]（用于代表数量）
        latest     = {}   # class_name -> 最近一次出现的完整 object dict
        for (_, data) in frames:
            frame_counter = {}
            for o in data.get("objects", []):
                cls = o.get("class_name")
                if not cls:
                    continue
                latest[cls] = o
                frame_counter[cls] = frame_counter.get(cls, 0) + 1
            for cls, c in frame_counter.items():
                appear[cls] = appear.get(cls, 0) + 1
                per_counts.setdefault(cls, []).append(c)
        threshold = self._stable_threshold(total)
        stable_classes = [c for c, n in appear.items() if n >= threshold]
        if not stable_classes:
            return None
        stable_objs = []
        for c in stable_classes:
            rep = dict(latest[c])
            rep["count"] = Counter(per_counts[c]).most_common(1)[0][0]  # 代表数量=众数
            stable_objs.append(rep)
        graspable = [o for o in stable_objs
                     if o.get("graspable") or o.get("action") == "grasp_allowed"]
        dialogue_only = [o for o in stable_objs
                         if not (o.get("graspable") or o.get("action") == "grasp_allowed")]
        total_count = sum(o["count"] for o in stable_objs)
        return {"objects": stable_objs,
                "graspable": graspable,
                "dialogue_only": dialogue_only,
                "count": total_count}

    # ── 情绪干预 ─────────────────────────────────────────────

    def _check_emotion_intervention(self, stable_emotion: dict):
        """
        负面情绪自动干预（唯一由视觉话题主动触发的动作）。
        仅当稳定情绪 intervention_required=true（low_mood / negative_distress）时触发，
        带 30 秒冷却防止重复干预。
        """
        if not stable_emotion.get("intervention_required"):
            return
        now = time.time()
        if now - self._last_intervention_time < self._intervention_cooldown:
            return
        self._last_intervention_time = now
        emotion_zh = stable_emotion.get("emotion_zh", "情绪不佳")
        self.get_logger().warn(f"触发情绪干预: {emotion_zh}")
        self._work_q.put(f"__EMOTION_INTERVENTION__:{emotion_zh}")

    # ── 构建视觉上下文 ───────────────────────────────────────

    @staticmethod
    def _fmt_obj(o: dict) -> str:
        """格式化单个物体，带数量：如 "苹果×2" / "香蕉"。"""
        name = o.get("name_zh", o.get("class_name", ""))
        cnt = o.get("count", 1)
        return f"{name}×{cnt}" if cnt > 1 else name

    def _build_vision_context(self) -> str:
        """
        从去抖后的稳定视觉/情绪状态构建注入 LLM 的上下文字符串。
        物体附带数量（同类多个时以 ×N 标注），并给出物体总数，便于 LLM 理解多物体场景。
        "桌上有什么"的中文表达由 LLM 自行组织，这里只提供结构化的物体清单与情绪。
        """
        parts = []

        objs = self._stable_objects()
        if objs:
            names = [self._fmt_obj(o) for o in objs["objects"]]
            if names:
                parts.append(
                    f"【当前桌面物体】共{objs['count']}个：{', '.join(names)}")
            if objs["graspable"]:
                gnames = [self._fmt_obj(o) for o in objs["graspable"]]
                parts.append(f"【可抓取物体】{', '.join(gnames)}")
            if objs["dialogue_only"]:
                dnames = [self._fmt_obj(o) for o in objs["dialogue_only"]]
                parts.append(f"【仅可对话物体】{', '.join(dnames)}")

        emo = self._stable_emotion()
        if emo:
            emo_zh = emo.get("emotion_zh", "")
            if emo_zh:
                parts.append(f"【用户情绪】{emo_zh}")

        return "\n".join(parts) if parts else ""

    # ── 指令处理 ─────────────────────────────────────────────

    def _on_command(self, msg: String):
        self._work_q.put(msg.data)

    def _work_loop(self):
        while True:
            cmd = self._work_q.get()

            # ── 情绪干预：主动触发 LLM 安抚对话，不产生机械指令 ──
            if cmd.startswith("__EMOTION_INTERVENTION__:"):
                emotion_zh = cmd.split(":", 1)[1]
                vision_ctx = self._build_vision_context()
                prompt = (
                    f"检测到用户当前情绪状态为：{emotion_zh}。"
                    f"请主动进行安抚和关怀对话，不要询问用户怎么了，"
                    f"直接表达关心、陪伴和支持。语气温柔自然。"
                )
                try:
                    self.get_logger().info(f"执行情绪干预: {emotion_zh}")
                    spoken, _ = generate_response(
                        prompt, vision_context=vision_ctx)
                    if spoken:
                        self.get_logger().info(f"安抚回复: {spoken}")
                        stream_play(spoken)
                except Exception as e:
                    self.get_logger().error(f"情绪干预失败: {e}")
                continue

            # ── 普通语音指令处理（结合去抖后的视觉/情绪上下文）──
            self.get_logger().info(f"收到指令: {cmd}")
            try:
                vision_ctx = self._build_vision_context()
                if vision_ctx:
                    self.get_logger().info(
                        f"视觉上下文: {vision_ctx[:100]}...")
                instructions = process_command(
                    cmd, log=self.get_logger().info,
                    vision_context=vision_ctx)
                if instructions:
                    payload = json.dumps(instructions, ensure_ascii=False)
                    self.get_logger().info(f"发布指令: {payload}")
                    self._instr_pub.publish(String(data=payload))
                else:
                    self.get_logger().info("无机械指令")
            except Exception as e:
                self.get_logger().error(f"处理指令失败: {e}")


def main():
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
