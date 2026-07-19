#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_node: 纯 ROS 层。
发布 /voice/command   — 用户指令文本。

麦克风由三个**互相独立**的闸控制，全开才收音（见 _MicGate）：

  轮次闸   pipeline 识别到指令后自锁，brain 发 continuous/command 放开
  硬静音闸 外部节点发 /voice/listen_mode 的 mute/unmute
  播报闸   brain 发 /voice/speaking 的 start/end

播报闸是必需的：TTS 与放歌都在 brain_node 进程内进行，voice_node 无从感知。
尤其情绪干预由视觉话题触发、不经过 voice_node，没有这条信号就会让机器人
把自己的声音当成用户指令收回来。

三闸独立意味着互不干扰：brain 播报结束不会解掉外部节点的硬静音，
外部 unmute 也不会在 brain 说到一半时把麦克风打开。

/voice/listen_mode 四种模式，分属两个维度：

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
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module.pipeline import listen_for_commands


class _MicGate:
    """麦克风闸门：三个互相独立的条件全部放行，麦克风才开。

        round_open   轮次闸——pipeline 识别到指令后自锁，brain 发 continuous/command 放开
        hard_muted   硬静音——外部节点（机械臂/底盘）的粘性静音，只有 unmute 能解
        speaking     播报闸——brain 正在 TTS 或放歌

    三者独立是必需的：brain 播报结束不能把外部节点的硬静音一起解掉，
    外部 unmute 也不该在 brain 说话说到一半时把麦克风打开。

    对外暴露 threading.Event 的接口（is_set/wait/set/clear），因此
    audio.py 与 pipeline.py 可以原样使用，无需改动。
    """

    def __init__(self):
        self._ev = threading.Event()
        self._ev.set()
        self._round_open = True
        self._hard_muted = False
        self._speaking   = False
        self._closed_since = None       # 闸门关闭的起始时刻，供看门狗判断
        self._lock = threading.Lock()   # 回调线程与看门狗线程会并发改这些标志

    def _refresh(self):
        """调用者需持有 _lock。"""
        if self._round_open and not self._hard_muted and not self._speaking:
            self._ev.set()
            self._closed_since = None
        else:
            self._ev.clear()
            if self._closed_since is None:
                self._closed_since = time.monotonic()

    def stuck_seconds(self) -> float:
        """闸门已连续关闭多久。

        硬静音不计入——那是外部节点有意为之的无限期静音，
        看门狗无权干涉，否则机械臂抓取途中麦克风会被擅自打开。
        """
        with self._lock:
            if self._hard_muted or self._closed_since is None:
                return 0.0
            return time.monotonic() - self._closed_since

    def force_release(self):
        """看门狗兜底：松开 brain 负责的两个闸，不碰硬静音。"""
        with self._lock:
            self._round_open = True
            self._speaking   = False
            self._refresh()

    # ── Event 接口：供 audio.py / pipeline.py 直接使用 ──
    def is_set(self):        return self._ev.is_set()
    def wait(self, t=None):  return self._ev.wait(t)

    def clear(self):
        """pipeline 识别到指令后自锁。"""
        with self._lock:
            self._round_open = False
            self._refresh()

    def set(self):
        with self._lock:
            self._round_open = True
            self._refresh()

    # ── 各条件独立开关 ──
    def set_hard_muted(self, v: bool):
        with self._lock:
            self._hard_muted = v
            self._refresh()

    def set_speaking(self, v: bool):
        with self._lock:
            self._speaking = v
            self._refresh()

    @property
    def hard_muted(self):    return self._hard_muted
    @property
    def speaking(self):      return self._speaking
    @property
    def round_open(self):    return self._round_open


class VoiceNode(Node):
    def __init__(self):
        super().__init__("voice_node")
        self._cmd_pub = self.create_publisher(String, "/voice/command", 10)
        self.create_subscription(String, "/voice/listen_mode", self._on_listen_mode, 10)
        self.create_subscription(String, "/voice/speaking", self._on_speaking, 10)

        self._listen_mode = {"wake_required": True}
        self._running = threading.Event()
        self._running.set()
        self._stop_ev = threading.Event()   # 初始未置位，供看门狗真正睡眠
        self._active = _MicGate()           # 三闸合一，见 _MicGate

        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    # ── 看门狗：brain 挂掉时兜底解锁 ──────────────────────────

    # 闸门连续关闭超过此秒数即强制松开。取值要高于任何合法播报的上限：
    # LLM 思考 ~15s + TTS ~20s + 单曲上限 60s ≈ 95s，故留足余量到 150s。
    # 正常流程永远碰不到这个阈值，只有 brain_node 崩溃/线程死掉才会触发。
    _WATCHDOG_SEC = 150.0

    def _watchdog_loop(self):
        """brain_node 负责的两个闸（轮次闸、播报闸）都依赖它发消息来松开。

        try/finally 挡得住 Python 异常，但挡不住进程被杀、OOM，或 work_loop
        线程整个死掉——那些情况下麦克风会永久关闭，整机哑掉。此处兜底。

        硬静音不在兜底范围：那是外部节点有意为之的无限期静音，
        擅自打开会让机械臂抓取途中的静音失效。
        """
        # 注意：不能用 self._running.wait()——_running 是"已置位"的运行标志，
        # Event.wait() 对已置位的事件立刻返回，循环会空转吃满一个 CPU 核。
        # 这里用一个初始为"未置位"的停止事件，wait 才会真正睡够 5 秒。
        while not self._stop_ev.wait(5.0):
            stuck = self._active.stuck_seconds()
            if stuck > self._WATCHDOG_SEC:
                self.get_logger().error(
                    f"麦克风已关闭 {stuck:.0f}s 超过阈值 {self._WATCHDOG_SEC:.0f}s，"
                    f"brain_node 可能已崩溃——强制恢复监听"
                )
                self._active.force_release()

    # ── /voice/speaking：brain_node 播报期间闭麦 ──────────────

    def _on_speaking(self, msg: String):
        """brain_node 播报（TTS / 放歌）的成对信号：start / end。

        brain_node 在自己的进程里出声，voice_node 无从感知，因此由它显式告知。
        情绪干预由视觉话题触发、不经过 voice_node，只有这条信号能让麦克风闭合。

        与硬静音相互独立：播报结束只清播报闸，不会解掉外部节点的硬静音。
        """
        speaking = (msg.data == "start")
        self._active.set_speaking(speaking)
        self.get_logger().info("播报开始，闭麦" if speaking else "播报结束")

    # ── /voice/listen_mode：轮次模式 + 硬静音 ────────────────

    def _on_listen_mode(self, msg: String):
        """四种模式，见模块 docstring。

        硬静音与轮次模式是独立的闸，由 _MicGate 做与运算——硬静音期间
        轮次模式只更新 wake_required，不会把麦克风打开。
        """
        mode = msg.data

        if mode == "mute":
            self._active.set_hard_muted(True)
            self.get_logger().info("硬静音，仅 unmute 可解除")
            return

        if mode == "unmute":
            self._active.set_hard_muted(False)
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

        self._active.set()             # 放开轮次闸；另两闸仍可能压着
        if not self._active.is_set():
            self.get_logger().info(f"{mode} 已记下，但硬静音/播报中，暂不开麦")

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
        self._stop_ev.set()          # 唤醒看门狗，立即退出


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
