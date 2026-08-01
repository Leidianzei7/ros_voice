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

/voice/listen_mode 六种模式，分属三个维度：

  轮次维度（brain_node 每轮处理完发一次，表示"下一句话怎么收"）：
    continuous — 下一句免唤醒词，直接当指令
    command    — 下一句需要先说唤醒词

  硬静音维度（外部节点任意时刻可发，独立于轮次）：
    mute       — 麦克风彻底关闭，且此后 continuous/command 一律无法开启
    unmute     — 解除硬静音

  紧急确认维度（发起方在联络前开窗确认"用户是否要叫停"，优先级最高）：
    emergency_confirm     — 强制开麦 + 免唤醒词，压过轮次闸与硬静音
    emergency_confirm_end — 退出确认态，恢复进入前的轮次模式与硬静音状态

硬静音是"粘性"的：一旦 mute，只有 unmute 能解除。即使 mute 到达时
brain_node 正思考到一半，它随后发回的 continuous/command 也不会把麦克风打开。

确认态比硬静音还高一级：机械臂抓取途中老人情绪异常、麦克风正被 mute 压着，
若不允许 emergency_confirm 越过它，用户喊"不用发了"根本传不进来。因此
确认期间硬静音只是被**暂时旁路**（状态仍记着），emergency_confirm_end 后
原样恢复。播报闸不在旁路之列——机器人自己说话时照旧闭麦。

⚠️ brain_node 只能发 continuous/command/emergency_confirm_end，永远不可以
   发 mute —— 静音后用户无法说话，brain_node 也就永远等不到下一条指令去
   触发 unmute，直接死锁。
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

    此外还有一个凌驾其上的 emergency 旁路：紧急呼叫期间必须听得见用户说
    "不用打了"，因此它压过轮次闸和硬静音（但不压播报闸——机器人自己说话时
    照旧闭麦，否则会把自己的声音当成用户指令收回来）。

    对外暴露 threading.Event 的接口（is_set/wait/set/clear），因此
    audio.py 与 pipeline.py 可以原样使用，无需改动。
    """

    def __init__(self):
        self._ev = threading.Event()
        self._ev.set()
        self._round_open = True
        self._hard_muted = False
        self._speaking   = False
        self._emergency  = False
        self._closed_since = None       # 闸门关闭的起始时刻，供看门狗判断
        self._lock = threading.Lock()   # 回调线程与看门狗线程会并发改这些标志

    def _refresh(self):
        """调用者需持有 _lock。"""
        want_open = self._emergency or (self._round_open and not self._hard_muted)
        if want_open and not self._speaking:
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

    def set_emergency(self, v: bool):
        """紧急旁路开关。

        退出时顺带把轮次闸复位：紧急期间 pipeline 每识别一句仍会调 clear()
        自锁（被旁路压着，麦克风照开），若退出时不复位，轮次闸就停在关闭态，
        而紧急流程里 brain_node 未必会再发 continuous/command 来放开它，
        麦克风要一直哑到 150 秒后看门狗兜底才恢复。
        """
        with self._lock:
            self._emergency = v
            if not v:
                self._round_open = True
            self._refresh()

    @property
    def hard_muted(self):    return self._hard_muted
    @property
    def speaking(self):      return self._speaking
    @property
    def round_open(self):    return self._round_open
    @property
    def emergency(self):     return self._emergency


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

        # 紧急态：进入时暂存轮次模式，emergency_end 后原样恢复
        self._emergency = False
        self._emergency_since = 0.0
        self._saved_wake_required = True
        self._emg_lock = threading.Lock()   # 回调线程与看门狗线程都会切换紧急态

        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    # ── 看门狗：brain 挂掉时兜底解锁 ──────────────────────────

    # 闸门连续关闭超过此秒数即强制松开。取值要高于任何合法播报的上限：
    # LLM 思考 ~15s + TTS ~20s + 单曲上限 60s ≈ 95s，故留足余量到 150s。
    # 正常流程永远碰不到这个阈值，只有 brain_node 崩溃/线程死掉才会触发。
    _WATCHDOG_SEC = 150.0

    # 紧急态最长持续时间。emergency/emergency_end 同样要求成对发送，漏发
    # emergency_end 的后果比漏发 unmute 更糟：麦克风被钉在"强制开 + 免唤醒词"，
    # 机械臂/底盘的 mute 也压不住它，且每句话都会被送去做中止意图判定。
    # 取 180s：紧急电话拨号+接通+通话通常在两三分钟内出结果。
    _EMERGENCY_MAX_SEC = 180.0

    def _watchdog_loop(self):
        """brain_node 负责的两个闸（轮次闸、播报闸）都依赖它发消息来松开。

        try/finally 挡得住 Python 异常，但挡不住进程被杀、OOM，或 work_loop
        线程整个死掉——那些情况下麦克风会永久关闭，整机哑掉。此处兜底。

        硬静音不在兜底范围：那是外部节点有意为之的无限期静音，
        擅自打开会让机械臂抓取途中的静音失效。

        紧急态则相反，超时必须**收**（而不是放）：它是个强制开麦的旁路，
        滞留只会让麦克风失控，因此单独设一个 _EMERGENCY_MAX_SEC 兜底退出。
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

            if self._emergency:
                held = time.monotonic() - self._emergency_since
                if held > self._EMERGENCY_MAX_SEC:
                    self.get_logger().error(
                        f"紧急态已持续 {held:.0f}s 超过阈值 "
                        f"{self._EMERGENCY_MAX_SEC:.0f}s，紧急侧可能漏发 "
                        f"emergency_confirm_end——自动退出确认态"
                    )
                    self._exit_emergency(reason="超时兜底")

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

    # ── /voice/listen_mode：轮次模式 + 硬静音 + 紧急态 ────────

    def _on_listen_mode(self, msg: String):
        """六种模式，见模块 docstring。

        硬静音与轮次模式是独立的闸，由 _MicGate 做与运算——硬静音期间
        轮次模式只更新 wake_required，不会把麦克风打开。
        紧急态再高一级，压过前两者。
        """
        mode = msg.data

        if mode == "emergency_confirm":
            self._enter_emergency()
            return

        if mode == "emergency_confirm_end":
            self._exit_emergency(reason="收到 emergency_confirm_end")
            return

        if mode == "mute":
            self._active.set_hard_muted(True)
            self.get_logger().info("硬静音，仅 unmute 可解除")
            return

        if mode == "unmute":
            self._active.set_hard_muted(False)
            self.get_logger().info("解除硬静音")
            return

        if mode == "continuous":
            wake_required = False
        elif mode == "command":
            wake_required = True
        else:
            # 未知值按 command 处理：宁可多要一次唤醒词，也不能把麦克风锁死
            self.get_logger().warn(f"未知模式 {mode!r}，按 command 处理")
            wake_required = True

        # 紧急期间免唤醒词是硬性的：brain_node 每播完一句就会发一次 command，
        # 若直接写进生效值，用户第二次喊"不用打了"就得先说唤醒词才收得到。
        # 这里只暂存，emergency_end 后再恢复——与硬静音的粘性同理。
        if self._emergency:
            self._saved_wake_required = wake_required
            self.get_logger().info(f"{mode} 已记下，紧急态结束后再生效")
            return

        self._listen_mode["wake_required"] = wake_required
        self._active.set()             # 放开轮次闸；另两闸仍可能压着
        if not self._active.is_set():
            self.get_logger().info(f"{mode} 已记下，但硬静音/播报中，暂不开麦")

    # ── 紧急态进出 ───────────────────────────────────────────

    def _enter_emergency(self):
        """紧急呼叫模块开始拨打紧急电话/发紧急短信时进入。

        强制开麦 + 免唤醒词，让 brain_node 能听到用户的中止意图。
        重复收到 emergency 只刷新计时，不覆盖已暂存的轮次模式。
        """
        with self._emg_lock:
            if self._emergency:
                self._emergency_since = time.monotonic()   # 续期，不重复进入
                return
            self._saved_wake_required = self._listen_mode["wake_required"]
            self._emergency = True
            self._emergency_since = time.monotonic()

        self._listen_mode["wake_required"] = False
        self._active.set_emergency(True)
        muted = "（硬静音被暂时旁路）" if self._active.hard_muted else ""
        self.get_logger().warn(f"进入紧急态：强制开麦 + 免唤醒词{muted}")

    def _exit_emergency(self, reason: str):
        """退出紧急态，恢复进入前的轮次模式与硬静音状态。"""
        with self._emg_lock:
            if not self._emergency:
                return          # 未进入过/已退出，幂等
            self._emergency = False
            wake_required = self._saved_wake_required

        self._listen_mode["wake_required"] = wake_required
        self._active.set_emergency(False)
        restored = "command" if wake_required else "continuous"
        self.get_logger().warn(f"退出紧急态（{reason}），恢复 {restored} 模式")
        if not self._active.is_set():
            self.get_logger().info("硬静音/播报仍压着，暂不开麦")

    def _on_command(self, cmd: str):
        # 紧急态下这句话不是普通指令，brain_node 会拿它去判定中止意图，
        # 日志里标出来，事后复盘"当时到底听到了什么"时省事
        tag = "[紧急态] " if self._emergency else ""
        self.get_logger().info(f"{tag}指令: {cmd}")
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
