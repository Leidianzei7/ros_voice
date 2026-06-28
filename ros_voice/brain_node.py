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
            self._work_q.put(f"__EMOTION_INTERVENTION__:{emotion_zh}")

    # ── 指令处理 ─────────────────────────────────────────────

    def _on_command(self, msg: String):
        self._work_q.put(msg.data)

    def _work_loop(self):
        while True:
            cmd = self._work_q.get()
            is_intervention = cmd.startswith("__EMOTION_INTERVENTION__:")

            if is_intervention:
                emotion_zh = cmd.split(":", 1)[1]
                self.get_logger().info(f"触发情绪干预: {emotion_zh}")
                cmd = (
                    f"检测到用户当前情绪状态为：{emotion_zh}。"
                    f"请主动进行安抚和关怀对话，不要询问用户怎么了，"
                    f"直接表达关心、陪伴和支持。语气温柔自然。"
                )
            else:
                self.get_logger().info(f"收到指令: {cmd}")

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

                if full_ctx:
                    self.get_logger().info(
                        f"上下文: {full_ctx[:120]}...")

                # TTS 播报期间静音麦克风，防止把自己的回答当指令
                self._listen_pub.publish(String(data="mute"))
                instructions, spoken = process_command(
                    cmd, log=self.get_logger().info,
                    vision_context=full_ctx)

                # 发布机械指令
                if instructions:
                    payload = json.dumps(instructions, ensure_ascii=False)
                    self.get_logger().info(f"发布指令: {payload}")
                    self._instr_pub.publish(String(data=payload))
                else:
                    self.get_logger().info("无机械指令")

                # 保存对话历史 + 提取记忆（仅普通对话）
                if not is_intervention and spoken:
                    self._mem.add_turn(cmd, spoken)
                    self._mem.extract_and_save(cmd, spoken)

                # 检测问句 → 开启持续监听
                self._check_question_mode(spoken)

            except Exception as e:
                self.get_logger().error(f"处理指令失败: {e}")

    def _check_question_mode(self, spoken: str):
        """如果口语回复以问句结尾，通知 voice_node 进入持续监听模式。"""
        if not spoken:
            self._listen_pub.publish(String(data="command"))
            return
        last = spoken.strip().split("\n")[-1].strip()
        if last.endswith("？") or last.endswith("?") or last.endswith("吗") or last.endswith("呢"):
            self.get_logger().info(f"检测到问句结尾，开启持续监听")
            self._listen_pub.publish(String(data="continuous"))
        else:
            self._listen_pub.publish(String(data="command"))


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
