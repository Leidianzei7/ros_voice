#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command          (std_msgs/String) — 用户指令文本。
订阅 /vision/scene_objects    (std_msgs/String) — 机械臂视觉返回的当前可见物体(JSON)。
订阅 /vision/emotion_context  (std_msgs/String) — 当前情绪(JSON)。
发布 /command                 (std_msgs/String) — JSON 指令数组。

感知数据预处理（缓存/去抖/格式化）委托给 voice_brain_module.context.ContextPipeline。
"""
import json
import queue
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module.context import ContextPipeline
from voice_brain_module.pipeline import process_command


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

        self._ctx = ContextPipeline(window_sec=3.0)
        self._work_q = queue.Queue()
        threading.Thread(target=self._work_loop, daemon=True).start()

        self.get_logger().info(
            "brain_node 就绪，等待 /voice/command 及 /vision/* 话题")

    # ── 视觉话题回调（仅缓存）─────────────────────────────────

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

            # 情绪干预：把触发信号转为安抚 prompt，其余流程与普通指令一致
            if cmd.startswith("__EMOTION_INTERVENTION__:"):
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
                vision_ctx = self._ctx.build_prompt()
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
