#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command      (std_msgs/String) — 用户指令文本。
发布 /command (std_msgs/String) — JSON 数组，每项 {"cmd": ..., "params": {...}}。
处理委托给 realtime_asr.pipeline.process_command。
"""
import sys
import json
import queue
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realtime_asr.pipeline import process_command


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")
        self._instr_pub = self.create_publisher(String, "/command", 10)
        self.create_subscription(String, "/voice/command", self._on_command, 10)

        self._work_q = queue.Queue()
        threading.Thread(target=self._work_loop, daemon=True).start()

        self.get_logger().info("brain_node 就绪，等待 /voice/command")

    def _on_command(self, msg: String):
        self._work_q.put(msg.data)

    def _work_loop(self):
        while True:
            cmd = self._work_q.get()
            self.get_logger().info(f"收到指令: {cmd}")
            try:
                instructions = process_command(cmd, log=self.get_logger().info)
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
