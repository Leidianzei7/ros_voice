#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_node: 纯 ROS 层。
发布 /voice/command   — 用户指令文本。
订阅 /voice/listen_mode — brain_node 控制监听模式 (mute/continuous/command)。
订阅 /voice/mute        — 任意节点可发布，收到即静音麦克风（与 TTS 时静音相同）。
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
        self.create_subscription(String, "/voice/mute", self._on_mute, 10)

        self._listen_mode = {"wake_required": True}
        self._running = threading.Event()
        self._active  = threading.Event()
        self._running.set()
        self._active.set()

    def _on_mute(self, msg: String):
        """收到 /voice/mute 话题即静音麦克风，与 TTS 播放时静音机制相同。"""
        self.get_logger().info(f"收到静音指令: {msg.data}")
        self._active.clear()

    def _on_listen_mode(self, msg: String):
        """brain_node 控制监听模式。"""
        if msg.data == "mute":
            self._active.clear()                          # 停止 VAD+ASR
        elif msg.data == "continuous":
            self._listen_mode["wake_required"] = False
            self._active.set()                            # 恢复
        else:  # "command"
            self._listen_mode["wake_required"] = True
            self._active.set()                            # 恢复

    def _on_command(self, cmd: str):
        self.get_logger().info(f"指令: {cmd}")
        self._cmd_pub.publish(String(data=cmd))

    def start(self):
        threading.Thread(
            target=listen_for_commands,
            kwargs={
                "on_command":  self._on_command,
                "log":         self.get_logger().info,
                "running":     self._running,
                "active":      self._active,
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
