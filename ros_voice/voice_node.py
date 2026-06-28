#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_node: 纯 ROS 层。
发布 /voice/command — 用户指令文本。
订阅 /voice/listen_mode — brain_node 控制监听模式。
"""
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module.pipeline import listen_for_commands


class VoiceNode(Node):
    def __init__(self):
        super().__init__("voice_node")
        self._cmd_pub = self.create_publisher(String, "/voice/command", 10)
        self.create_subscription(String, "/voice/listen_mode", self._on_listen_mode, 10)

        self._listen_mode = {"wake_required": True, "muted": False}
        self._running = threading.Event()
        self._running.set()

    def _on_listen_mode(self, msg: String):
        """brain_node 控制监听模式。"""
        if msg.data == "mute":
            self._listen_mode["muted"] = True
        elif msg.data == "continuous":
            self._listen_mode["wake_required"] = False
            self._listen_mode["muted"] = False
        else:  # "command"
            self._listen_mode["wake_required"] = True
            self._listen_mode["muted"] = False

    def _on_command(self, cmd: str):
        self.get_logger().info(f"指令: {cmd}")
        self._listen_mode["wake_required"] = True  # 发出后恢复默认
        self._cmd_pub.publish(String(data=cmd))

    def start(self):
        threading.Thread(
            target=listen_for_commands,
            kwargs={
                "on_command":  self._on_command,
                "log":         self.get_logger().info,
                "running":     self._running,
                "wake_required_ref": self._listen_mode,
            },
            daemon=True,
        ).start()

    def stop(self):
        self._running.clear()


def main():
    rclpy.init()
    node = VoiceNode()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
