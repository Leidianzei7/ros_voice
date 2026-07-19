#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_node: 纯 ROS 层。
发布 /voice/command   — 用户指令文本。

两大控制话题（均直接开关麦克风，后到者生效）：
  /voice/listen_mode  — brain_node 控制 (mute/continuous/command)
  /voice/mute         — 外部节点静音开关 (mute/unmute)
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

    # ── /voice/mute：外部节点静音/解除 ────────────────────────

    def _on_mute(self, msg: String):
        """外部节点（底盘/机械臂等）请求静音或解除。

        与 TTS 播放期间的静音走同一条通路：直接开关 _active。
        - "unmute"：解除静音
        - 其他任意内容（含 "mute"）：静音
        """
        if msg.data == "unmute":
            self.get_logger().info("解除静音")
            self._active.set()
        else:
            self.get_logger().info(f"静音: {msg.data}")
            self._active.clear()

    # ── /voice/listen_mode：brain_node 控制 ──────────────────

    def _on_listen_mode(self, msg: String):
        """brain_node 对 voice_node 的监听模式控制。

        三种模式：
          "mute"       — 停止 VAD+ASR，麦克风完全静音（如 TTS 播放时）
          "continuous" — 持续监听，不需要唤醒词，每句话都作为指令
          "command"    — 命令模式，需要先说唤醒词才能触发指令
        """
        if msg.data == "mute":
            self._active.clear()
        elif msg.data == "continuous":
            self._listen_mode["wake_required"] = False
            self._active.set()
        else:  # "command"
            self._listen_mode["wake_required"] = True
            self._active.set()

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
