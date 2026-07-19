#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知上下文管道：缓存 → 去抖 → 格式化，为 LLM 提供稳定的视觉/情绪上下文。

供 brain_node 等 ROS 层调用，屏蔽滑动窗口、众数投票、干预冷却等内部细节。

用法:
    ctx = ContextPipeline(window_sec=3.0)
    ctx.feed_scene(json.loads(scene_msg.data))
    emotion = ctx.feed_emotion(json.loads(emotion_msg.data))
    prompt = ctx.build_prompt()                    # 注入 LLM 的上下文字符串
"""
import math
import time
import threading
from collections import deque, Counter


class ContextPipeline:
    """感知数据缓存与去抖管道。"""

    def __init__(self, window_sec: float = 3.0, stable_ratio: float = 0.5,
                 intervention_cooldown: float = 30.0):
        self._window_sec = window_sec
        self._stable_ratio = stable_ratio
        self._intervention_cooldown = intervention_cooldown
        self._last_intervention_time = 0.0
        self._prev_intervention_time = 0.0   # 供 cancel_intervention 回退

        self._scene_window   = deque()   # [(stamp, data), ...]
        self._emotion_window = deque()
        self._lock = threading.Lock()

    # ── 窗口工具（线程安全）────────────────────────────────

    def _push(self, window: deque, data: dict):
        now = time.time()
        with self._lock:
            window.append((now, data))
            while window and (now - window[0][0]) > self._window_sec:
                window.popleft()

    def _snapshot(self, window: deque) -> list:
        now = time.time()
        with self._lock:
            while window and (now - window[0][0]) > self._window_sec:
                window.popleft()
            return list(window)

    def _stable_threshold(self, total: int) -> int:
        return max(2, math.ceil(total * self._stable_ratio))

    # ── 对外接口 ──────────────────────────────────────────

    def feed_scene(self, data: dict):
        """摄入一帧物体检测结果。"""
        self._push(self._scene_window, data)

    def feed_emotion(self, data: dict) -> str | None:
        """
        摄入一帧情绪检测结果。
        若稳定情绪需要干预且冷却已过，返回情绪中文名；
        否则返回 None。
        """
        self._push(self._emotion_window, data)
        stable = self._stable_emotion()
        if not stable or not stable.get("intervention_required"):
            return None
        now = time.time()
        with self._lock:
            if now - self._last_intervention_time < self._intervention_cooldown:
                return None
            # 这里就记账是为了防止刷屏：情绪帧持续涌入时，冷却未过的帧直接返回
            # None，不会重复入队。若干预最终没送达，由 cancel_intervention 退还。
            self._prev_intervention_time = self._last_intervention_time
            self._last_intervention_time = now
        return stable.get("emotion_zh", "情绪不佳")

    def cancel_intervention(self):
        """干预未实际送达（排队过久被丢弃），退还冷却。

        feed_emotion 在检测到干预时就抢先记了账，若该干预最终没能播报出去，
        必须把冷却退回去——否则一次"未遂"的关怀会白白堵死后续 30 秒。
        """
        with self._lock:
            self._last_intervention_time = self._prev_intervention_time

    def build_prompt(self) -> str:
        """从去抖后的稳定视觉/情绪状态构建注入 LLM 的上下文字符串。"""
        parts = []

        objs = self._stable_objects()
        if objs:
            names = [self._fmt_obj(o) for o in objs["objects"]]
            if names:
                parts.append(f"【当前桌面物体】共{objs['count']}个：{', '.join(names)}")
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

    # ── 去抖算法 ──────────────────────────────────────────

    def _stable_emotion(self):
        frames = self._snapshot(self._emotion_window)
        emotions = [d.get("emotion") for (_, d) in frames if d.get("emotion")]
        if not emotions:
            return None
        dominant, count = Counter(emotions).most_common(1)[0]
        if count < self._stable_threshold(len(emotions)):
            return None
        for (_, d) in reversed(frames):
            if d.get("emotion") == dominant:
                return d
        return None

    def _stable_objects(self):
        frames = self._snapshot(self._scene_window)
        if not frames:
            return None
        total = len(frames)
        appear     = {}
        per_counts = {}
        latest     = {}
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
            rep["count"] = Counter(per_counts[c]).most_common(1)[0][0]
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

    @staticmethod
    def _fmt_obj(o: dict) -> str:
        name = o.get("name_zh", o.get("class_name", ""))
        cnt = o.get("count", 1)
        return f"{name}×{cnt}" if cnt > 1 else name
