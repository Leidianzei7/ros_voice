#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command          — 用户指令文本。
订阅 /vision/scene_objects    — 视野内物体(JSON)。
订阅 /vision/emotion_context  — 用户情绪(JSON)。
发布 /command                 — JSON 指令数组。
发布 /voice/speak             — 播报任务(JSON)：文本 + 待播歌曲 + 下一轮监听模式。

brain_node 只思考，不出声。播报交给 voice_node——它同时握有麦克风和扬声器，
能在出声期间可靠闭麦。情绪干预由视觉话题触发、不经过 voice_node，若在此处
直接播报，voice_node 无从得知，会把机器人自己的声音当成用户指令收回来。

感知预处理 → ContextPipeline，持久记忆 → UserMemory。
"""
import json
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module.context import ContextPipeline
from voice_brain_module.memory import UserMemory
from voice_brain_module.pipeline import process_command


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")
        self._instr_pub    = self.create_publisher(String, "/command", 10)
        self._speak_pub    = self.create_publisher(String, "/voice/speak", 10)
        self.create_subscription(String, "/voice/command", self._on_command, 10)
        self.create_subscription(String, "/vision/scene_objects",
                                 self._on_scene_objects, 10)
        self.create_subscription(String, "/vision/emotion_context",
                                 self._on_emotion_context, 10)

        self._ctx   = ContextPipeline(window_sec=3.0)
        self._mem   = UserMemory()
        self._last_question = ""   # 机器人上一轮问用户的问题
        self._pending_songs = []   # 待播歌曲，随播报任务一起交给 voice_node
        self._work_q = queue.Queue()
        threading.Thread(target=self._work_loop, daemon=True).start()

        self.get_logger().info("brain_node 就绪")

    # ── 视觉话题回调 ──────────────────────────────────────────

    def _on_scene_objects(self, msg: String):
        try:
            self._ctx.feed_scene(json.loads(msg.data))
        except json.JSONDecodeError:
            self.get_logger().warn("scene_objects JSON 解析失败")

    def _on_emotion_context(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("emotion_context JSON 解析失败")
            return
        emotion_zh = self._ctx.feed_emotion(data)
        if emotion_zh:
            self.get_logger().warn(f"触发情绪干预: {emotion_zh}")
            self._work_q.put(
                f"__EMOTION_INTERVENTION__:{emotion_zh}:{time.time()}"
            )

    # ── 指令处理 ─────────────────────────────────────────────

    def _on_command(self, msg: String):
        self._work_q.put(msg.data)

    def _publish_instructions(self, instructions):
        """LLM 一返回就发布机械指令（在 TTS 播放之前），让动作和语音基本同步开始。

        音箱指令（播放歌曲）不下发 /command——control_node 没有音频通路。
        此处挑出来暂存，随后与播报文本一起交给 voice_node，由它闭麦后串行播放。
        """
        self._pending_songs = [
            c.get("params", {}).get("song")
            for c in (instructions or [])
            if c.get("actuator") == "音箱" and c.get("params", {}).get("song")
        ]
        remote = [c for c in (instructions or []) if c.get("actuator") != "音箱"]

        if remote:
            payload = json.dumps(remote, ensure_ascii=False)
            self.get_logger().info(f"发布指令: {payload}")
            self._instr_pub.publish(String(data=payload))
        elif not self._pending_songs:
            self.get_logger().info("无机械指令")

    def _work_loop(self):
        _INTERVENTION_TTL = 3.0   # 干预消息在队列中最长存活时间（秒）

        while True:
            cmd = self._work_q.get()
            is_intervention = cmd.startswith("__EMOTION_INTERVENTION__:")

            if is_intervention:
                # 格式: __EMOTION_INTERVENTION__:{emotion_zh}:{timestamp}
                # 用 rsplit 从右侧切时间戳：emotion_zh 来自视觉侧的外部 JSON，
                # 内容不可控，若含冒号，split(":", 2) 会把它切进时间戳字段，
                # float() 抛 ValueError 打死整个 work_loop 线程（此处在 try 之外）。
                head, _, ts = cmd.rpartition(":")
                emotion_zh = head[len("__EMOTION_INTERVENTION__:"):]
                queued_at  = float(ts)
                age = time.time() - queued_at
                if age > _INTERVENTION_TTL:
                    self.get_logger().info(
                        f"丢弃过期干预 ({age:.1f}s 前): {emotion_zh}"
                    )
                    # 干预没送达，退还冷却，否则白白堵死后续 30 秒
                    self._ctx.cancel_intervention()
                    continue   # 跳过，不等 TTS 也不调 LLM
                self.get_logger().info(f"触发情绪干预: {emotion_zh} (排队 {age:.1f}s)")
                cmd = (
                    f"检测到用户当前情绪状态为：{emotion_zh}。"
                    f"请主动进行安抚和关怀对话，不要询问用户怎么了，"
                    f"直接表达关心、陪伴和支持。语气温柔自然。"
                )
                # 情绪干预是系统自主触发，不是用户对机器人问句的应答，
                # 因此必须清空 last_question，否则 classify_meta_response
                # 会把干预 prompt 当作用户语音来分类，导致逻辑混乱。
                self._last_question = ""
            else:
                self.get_logger().info(f"收到指令: {cmd}")

            spoken = ""
            try:
                # 拼接感知上下文 + 记忆上下文
                vision_ctx = self._ctx.build_prompt()
                memory_ctx = self._mem.get_context_for_llm()
                full_ctx_parts = []
                if vision_ctx:
                    full_ctx_parts.append(vision_ctx)
                if memory_ctx:
                    full_ctx_parts.append(memory_ctx)
                full_ctx = "\n\n".join(full_ctx_parts) if full_ctx_parts else ""

                instructions, spoken = process_command(
                    cmd, log=self.get_logger().info,
                    vision_context=full_ctx,
                    on_instructions=self._publish_instructions,
                    last_question=self._last_question)

                # 保存对话历史 + 提取记忆（仅普通对话）
                if not is_intervention and spoken:
                    self._mem.add_turn(cmd, spoken)
                    self._mem.extract_and_save(cmd, spoken)

            except Exception as e:
                self.get_logger().error(f"处理指令失败: {e}")
            finally:
                # 无论成功失败，都必须恢复监听——否则 voice_node 永久静音（卡死）
                self._dispatch_speech(spoken)

    def _dispatch_speech(self, spoken: str):
        """把播报文本 + 待播歌曲 + 下一轮监听模式，一次性交给 voice_node。

        问句可能出现在句中而非句末（如"…是不是还没说完呀？或者想聊点别的…"），
        因此扫描整段回复是否含问号，而不是只看结尾。

        本方法必须在 finally 中被无条件调用：voice_node 在识别到指令时已自行
        闭麦，只有收到这条消息才会恢复监听。哪怕 text 为空也要发，否则永久静音。

        ⚠️ next_mode 只允许 continuous/command。mute 是外部节点的硬静音，
           brain_node 一旦发出，用户就无法说话，也就永远等不到下一条指令去
           发 unmute——直接死锁。
        """
        songs, self._pending_songs = self._pending_songs, []
        text = (spoken or "").strip()

        if text and ("？" in text or "?" in text
                     or text.split("\n")[-1].strip().endswith(("吗", "呢"))):
            next_mode = "continuous"
            self._last_question = text     # 记录本次问句，供下一轮检测
            self.get_logger().info("检测到问句，开启持续监听")
        else:
            next_mode = "command"
            self._last_question = ""       # 非问句，清除上一次的问句记录

        self._speak_pub.publish(String(data=json.dumps(
            {"text": text, "songs": songs, "next_mode": next_mode},
            ensure_ascii=False)))
