#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command          — 用户指令文本。
订阅 /vision/scene_objects    — 桌面物体(JSON)。
订阅 /vision/emotion_context  — 用户情绪(JSON)。
发布 /command                 — JSON 指令数组。
发布 /voice/listen_mode       — 控制 voice_node 监听模式。

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
        self._listen_pub   = self.create_publisher(String, "/voice/listen_mode", 10)
        self.create_subscription(String, "/voice/command", self._on_command, 10)
        self.create_subscription(String, "/vision/scene_objects",
                                 self._on_scene_objects, 10)
        self.create_subscription(String, "/vision/emotion_context",
                                 self._on_emotion_context, 10)

        self._ctx   = ContextPipeline(window_sec=3.0)
        self._mem   = UserMemory()
        self._last_question = ""   # 机器人上一轮问用户的问题
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
        """LLM 一返回就发布机械指令（在 TTS 播放之前），让动作和语音基本同步开始。"""
        if instructions:
            payload = json.dumps(instructions, ensure_ascii=False)
            self.get_logger().info(f"发布指令: {payload}")
            self._instr_pub.publish(String(data=payload))
        else:
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
                self._check_question_mode(spoken)

    def _check_question_mode(self, spoken: str):
        """如果口语回复中包含问句，通知 voice_node 进入持续监听模式。

        问句可能出现在句中而非句末（如"…是不是还没说完呀？或者想聊点别的…"），
        因此扫描整段回复是否含问号，而不是只看结尾。

        同时记录机器人问的最后一个问题，供下一轮 relevance 检测使用。

        ⚠️ 本方法只允许发 "continuous"/"command"，绝不可发 "mute"。
           mute 是外部节点的硬静音，一旦 brain_node 发出，用户就无法说话，
           brain_node 也永远等不到下一条指令去发 unmute —— 直接死锁。
        """
        if not spoken:
            self._listen_pub.publish(String(data="command"))
            return
        text = spoken.strip()
        last = text.split("\n")[-1].strip()
        if "？" in text or "?" in text or last.endswith("吗") or last.endswith("呢"):
            self.get_logger().info("检测到问句，开启持续监听")
            self._listen_pub.publish(String(data="continuous"))
            self._last_question = text   # 记录本次问句，供下一轮检测
        else:
            self._listen_pub.publish(String(data="command"))
            self._last_question = ""     # 非问句，清除上一次的问句记录


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
