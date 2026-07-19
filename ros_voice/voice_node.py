#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_node: 纯 ROS 层。
发布 /voice/command   — 用户指令文本。

唯一控制话题 /voice/listen_mode，四种模式，分属两个正交维度：

  轮次维度（brain_node 每轮处理完发一次，表示"下一句话怎么收"）：
    continuous — 下一句免唤醒词，直接当指令
    command    — 下一句需要先说唤醒词

  硬静音维度（外部节点任意时刻可发，独立于轮次）：
    mute       — 麦克风彻底关闭，且此后 continuous/command 一律无法开启
    unmute     — 解除硬静音

硬静音是"粘性"的：一旦 mute，只有 unmute 能解除。即使 mute 到达时
brain_node 正思考到一半，它随后发回的 continuous/command 也不会把麦克风打开。

⚠️ brain_node 只能发 continuous/command，永远不可以发 mute —— 因为静音后
   用户无法说话，brain_node 也就永远等不到下一条指令去触发 unmute，直接死锁。
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

        self._listen_mode = {"wake_required": True}
        self._running = threading.Event()
        self._active  = threading.Event()
        self._running.set()
        self._active.set()

        # 硬静音标志。为 True 时 continuous/command 无权开启麦克风。
        self._hard_muted = False

    # ── /voice/listen_mode：唯一控制入口 ─────────────────────

    def _on_listen_mode(self, msg: String):
        """四种模式，见模块 docstring。

        硬静音（mute/unmute）优先于轮次模式（continuous/command）：
        _hard_muted 为 True 时，轮次模式只更新 wake_required，不碰麦克风。
        """
        mode = msg.data

        if mode == "mute":
            self._hard_muted = True
            self._active.clear()
            self.get_logger().info("硬静音，仅 unmute 可解除")
            return

        if mode == "unmute":
            self._hard_muted = False
            self._active.set()
            self.get_logger().info("解除硬静音")
            return

        if mode == "continuous":
            self._listen_mode["wake_required"] = False
        elif mode == "command":
            self._listen_mode["wake_required"] = True
        else:
            # 未知值按 command 处理：宁可多要一次唤醒词，也不能把麦克风锁死
            self.get_logger().warn(f"未知模式 {mode!r}，按 command 处理")
            self._listen_mode["wake_required"] = True

        # 硬静音期间轮次模式无权开麦，但 wake_required 已记下，
        # unmute 后即以该模式恢复。
        if self._hard_muted:
            self.get_logger().info(f"硬静音中，忽略 {mode} 的开麦请求")
        else:
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
