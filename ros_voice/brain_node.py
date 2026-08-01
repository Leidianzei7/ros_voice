#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_node: 纯 ROS 层。
订阅 /voice/command          — 用户指令文本。
订阅 /vision/scene_objects    — 视野内物体(JSON)。
订阅 /vision/emotion_context  — 用户情绪(JSON)。
订阅 /voice/listen_mode       — 只关心 emergency_confirm/emergency_confirm_end，用于跟踪确认态。
发布 /command                 — JSON 指令数组。
发布 /voice/listen_mode       — 轮次模式 continuous/command + emergency_confirm_end。
发布 /voice/speaking          — 播报期间闭麦的成对信号 start/end。

TTS 与放歌都在本进程内进行，voice_node 无从感知，因此必须由 brain_node
显式发 /voice/speaking 告知。情绪干预由视觉话题触发、不经过 voice_node，
只有这条信号能让它在干预播报期间闭麦。

紧急态（紧急呼叫模块正在拨打紧急电话/发紧急短信）走**独立于工作队列的快车道**：
_work_loop 是串行的，一次 LLM+TTS 动辄十几秒，放歌更是能占到一分钟，把中止
判定排在后面等于让用户喊了"不用打了"却要等一首歌唱完才生效。因此紧急态下
_on_command 在 ROS 回调线程里当场判定并下发中止指令，只有事后的口头确认才
回到队列里排队（那时电话已经撤了，说话晚几秒无妨）。

感知预处理 → ContextPipeline，持久记忆 → UserMemory。
"""
import json
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_brain_module import emergency as emg
from voice_brain_module.commands import build_abort_command
from voice_brain_module.config import (
    EMERGENCY_ABORT_WINDOW_SEC, EMERGENCY_ASK_AGAIN_TEXT,
    EMERGENCY_ASK_TEXT, EMERGENCY_CANCELLED_TEXT,
    EMERGENCY_CHANNEL_BY_EMOTION, EMERGENCY_COOLDOWN_SEC,
    EMERGENCY_SENT_TEXT,
)
from voice_brain_module.context import ContextPipeline
from voice_brain_module.memory import UserMemory
from voice_brain_module.pipeline import process_command
from voice_brain_module.player import play_song

# 队列里的特殊消息前缀：只播一句固定话术，不走 LLM
_SPEAK_PREFIX = "__SPEAK__:"



class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")
        self._instr_pub    = self.create_publisher(String, "/command", 10)
        self._listen_pub   = self.create_publisher(String, "/voice/listen_mode", 10)
        self._speaking_pub = self.create_publisher(String, "/voice/speaking", 10)
        self._initiate_pub = self.create_publisher(String, "/emergency/initiate", 10)
        self.create_subscription(String, "/voice/command", self._on_command, 10)
        self.create_subscription(String, "/vision/scene_objects",
                                 self._on_scene_objects, 10)
        self.create_subscription(String, "/vision/emotion_context",
                                 self._on_emotion_context, 10)
        self.create_subscription(String, "/voice/listen_mode",
                                 self._on_listen_mode, 10)

        self._ctx   = ContextPipeline(window_sec=3.0,
                                      intervention_cooldown=EMERGENCY_COOLDOWN_SEC)
        self._mem   = UserMemory()
        self._last_question = ""   # 机器人上一轮问用户的问题
        self._pending_songs = []   # 待播歌曲，TTS 说完后由 _play_pending_songs 消费
        self._work_q = queue.Queue()

        # 紧急态：与 voice_node 各自独立跟踪同一个话题，互不依赖
        self._emergency = False
        self._emg_lock  = threading.Lock()
        self._emg_llm_busy = False   # 同时最多一个后台判定线程在跑
        self._emg_confirm_flag = threading.Event()  # 确认窗口内用户说"需要"
        self._emg_reask_flag  = threading.Event()   # 判定为听不清，需要追问

        threading.Thread(target=self._work_loop, daemon=True).start()
        threading.Thread(target=self._prewarm_llm, daemon=True).start()

        self.get_logger().info("brain_node 就绪")

    def _prewarm_llm(self):
        """后台预热 llm 模块的导入。

        llm 一直是惰性导入的（第一条用户指令来时才加载），openai + 提示词构建
        首次要好几秒。常规对话里这几秒混在 LLM 推理里看不出来，但紧急中止判定
        等不起——开机后第一次交互就是紧急呼叫时，一句规则拿不准的话要多等三四秒
        才出结论。这里在启动时就把导入做掉。

        失败无所谓（缺 API Key 等）：真用到时会再导一次并走各自的兜底逻辑。
        """
        try:
            import voice_brain_module.llm   # noqa: F401
        except Exception as e:
            self.get_logger().warn(f"LLM 模块预热失败，将在首次使用时重试: {e}")

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
        stable = self._ctx.feed_emotion(data)
        if not stable:
            return

        emotion    = stable.get("emotion", "")
        emotion_zh = stable.get("emotion_zh", "情绪不佳")
        channel    = EMERGENCY_CHANNEL_BY_EMOTION.get(emotion)

        if channel:
            # 映射表里的情绪 → 走紧急联络流程（先问，再决定打电话/发短信）
            self.get_logger().warn(
                f"触发紧急联络询问: {emotion_zh}({emotion}) → {channel}"
            )
            self._work_q.put({
                "type":       "emergency_ask",
                "channel":    channel,
                "emotion":    emotion,
                "emotion_zh": emotion_zh,
            })
        else:
            # 未纳入映射表的情绪 → 保持原来的大模型安抚，不联系家属
            self.get_logger().warn(f"触发情绪干预: {emotion_zh}")
            self._work_q.put(
                f"__EMOTION_INTERVENTION__:{emotion_zh}:{time.time()}"
            )

    # ── 紧急态：中止意图判定 ──────────────────────────────────

    def _on_listen_mode(self, msg: String):
        """只关心 emergency_confirm / emergency_confirm_end。

        本节点自己也往这个话题发 continuous/command，ROS 2 默认会把消息投递回
        发布者自身——因此**必须**只认这两个值，否则每轮对话结束时自己发的那句
        command 会把刚进入的紧急态立刻清掉。
        """
        if msg.data == "emergency_confirm":
            with self._emg_lock:
                already = self._emergency
                self._emergency = True
            if not already:
                self.get_logger().warn("进入确认态：开始监听中止意图")
        elif msg.data == "emergency_confirm_end":
            with self._emg_lock:
                was_in = self._emergency
                self._emergency = False
            # 中止成功时本节点自己会发一次 emergency_confirm_end，绕回来正好命中这里，
            # 只在真正发生状态跳变时打日志，免得日志里出现两条"退出紧急态"
            if was_in:
                self.get_logger().warn("退出确认态：恢复常规对话")

    def _handle_emergency_utterance(self, text: str) -> bool:
        """确认窗口内的用户语音。返回 True 表示已被消费。

        三分类：CONFIRM → 即刻发起 / ABORT → 即刻取消 / ASK_AGAIN → 追问。
        """
        with self._emg_lock:
            if self._emg_llm_busy:
                # 上一句还在判，这句直接放过。判定是幂等的，用户没被理解时会
                # 自己重复；排队反而让结论晚到，不如让下一句重新触发。
                self.get_logger().info(f"判定进行中，跳过本句: {text}")
                return True
            self._emg_llm_busy = True

        self.get_logger().info(f"确认窗口收到: {text}")
        threading.Thread(target=self._emergency_llm_arbitrate,
                         args=(text,), daemon=True).start()
        return True

    def _emergency_llm_arbitrate(self, text: str):
        """后台线程：三分类判定。"""
        try:
            from voice_brain_module.llm import classify_confirm_intent
            decision = classify_confirm_intent(text)
        except Exception as e:
            self.get_logger().error(f"判定异常，按听不清处理: {e}")
            decision = "ASK_AGAIN"
        finally:
            with self._emg_lock:
                self._emg_llm_busy = False

        if decision == emg.ABORT:
            self.get_logger().warn(f"判定中止: {text}")
            self._publish_abort(text, emg.DETECTOR_LLM)
        elif decision == emg.CONFIRM:
            self.get_logger().warn(f"判定确认: {text}")
            self._emg_confirm_flag.set()
        else:
            self.get_logger().info(f"听不清，追问: {text}")
            self._emg_reask_flag.set()

    # ── 紧急联络发起（负面情绪触发）───────────────────────────

    def _run_emergency_ask(self, task: dict):
        """播报询问 → 即刻开确认窗口 → 等 EMERGENCY_ABORT_WINDOW_SEC 秒。

        播完立刻发 emergency_confirm（voice_node 强制开麦、brain_node 切到
        中止判定）。窗口期内用户说的每句话都走中止判定——规则层离线判定"不用"、
        "我没事"、"别发"等拒绝语，拿不准的交大模型仲裁。窗口到期未被中止才发
        /emergency/initiate。和外部发起方的流程完全相同。

        「为什么无应答照发」：老人可能已经痛得说不出话或失去意识——那恰恰是
        最需要联络的情况。
        """
        self._speak_fixed(EMERGENCY_ASK_TEXT)
        self._emg_confirm_flag.clear()
        self._emg_reask_flag.clear()
        self._open_confirm_window()

        remaining = EMERGENCY_ABORT_WINDOW_SEC
        tick = time.time()
        while remaining > 0:
            with self._emg_lock:
                if not self._emergency:
                    # 已被 _publish_abort 中止：窗口早已关闭、取消话术也已排进
                    # 队列，这里直接退出即可——本函数一返回，工作线程就腾出来
                    # 播那句话。在这里再播一遍会变成前后两句意思重复的话。
                    return
            if self._emg_confirm_flag.is_set():
                self.get_logger().warn("用户确认联络，即刻发起")
                break
            # 大模型要求追问 → 播追问话术，重置窗口继续等
            if self._emg_reask_flag.is_set():
                self._emg_reask_flag.clear()
                self._speak_fixed(EMERGENCY_ASK_AGAIN_TEXT)
                tick = time.time()
                continue
            # 大模型在判 → 这一轮不计时。必须在 wait 之前取 busy、在 wait 之后
            # 才决定扣不扣：若像之前那样只在 wait 前重置 tick，wait 的 0.2 秒
            # 照样会被 now - tick 算进去，等于没暂停。
            with self._emg_lock:
                busy = self._emg_llm_busy
            self._emg_confirm_flag.wait(timeout=0.2)
            now = time.time()
            if not busy:
                remaining -= (now - tick)
            tick = now

        # 窗口到期或用户确认 → 发起
        self._close_confirm_window()
        self._publish_initiate(task)
        self._speak_fixed(EMERGENCY_SENT_TEXT.get(task["channel"], "已发送"))

    def _open_confirm_window(self):
        """发布确认信号。voice_node 收到后强制开麦免唤醒词，brain_node 的
        _on_listen_mode 回调也把本节点切成中止判定状态。

        不设定时器——调用方负责到时关窗。当前调用方 _run_emergency_ask 在
        窗口到期后调用 _close_confirm_window；将来其他发起方开窗时同理。
        """
        with self._emg_lock:
            self._emergency = True
        self._listen_pub.publish(String(data="emergency_confirm"))
        self.get_logger().warn(
            f"开启确认窗口 {EMERGENCY_ABORT_WINDOW_SEC:.0f}s")

    def _close_confirm_window(self):
        """窗口到期。若期间已中止过，_publish_abort 早已收尾，这里是空操作。"""
        with self._emg_lock:
            if not self._emergency:
                return          # 已经中止过，不必重复发
            self._emergency = False
        self.get_logger().info("确认窗口到期，联络照常发起")
        self._listen_pub.publish(String(data="emergency_confirm_end"))

    def _publish_initiate(self, task: dict):
        payload = json.dumps({
            "event":      "emergency_initiate",
            "channel":    task["channel"],          # call = 打电话；sms = 发短信
            "reason":     "negative_emotion",
            "emotion":    task["emotion"],
            "emotion_zh": task["emotion_zh"],
            "stamp_sec":  time.time(),
        }, ensure_ascii=False)
        self.get_logger().warn(f"发起紧急联络: {payload}")
        self._initiate_pub.publish(String(data=payload))

    def _publish_abort(self, utterance: str, detector: str):
        """下发"中止紧急情况"，并收尾紧急态。

        三步顺序不能换：先把指令发出去（用户等的是电话停下，不是机器人说话），
        再让 voice_node 退出紧急态，最后才把口头确认排进队列慢慢说。
        """
        with self._emg_lock:
            if not self._emergency:
                return          # 已经撤过了，别重复下发
            self._emergency = False

        payload = json.dumps([build_abort_command(utterance, detector)],
                             ensure_ascii=False)
        self._instr_pub.publish(String(data=payload))
        self.get_logger().warn(f"发布中止指令: {payload}")

        # 发起方职责：中止即关窗。⚠️ 必须**直接发**，不能调 _close_confirm_window()
        # ——上面已经把 _emergency 置 False，那个函数开头的守卫会直接 return，
        # emergency_confirm_end 就永远发不出去，麦克风要等 180 秒兜底才恢复。
        self.get_logger().info("中止成功，关闭确认窗口")
        self._listen_pub.publish(String(data="emergency_confirm_end"))

        self._work_q.put(_SPEAK_PREFIX + EMERGENCY_CANCELLED_TEXT)

    # ── 指令处理 ─────────────────────────────────────────────

    def _on_command(self, msg: String):
        # 紧急态走快车道：当场判定并下发，不进工作队列。
        with self._emg_lock:
            in_emergency = self._emergency
        if in_emergency:
            self._handle_emergency_utterance(msg.data)
            return

        self._work_q.put(msg.data)

    def _publish_instructions(self, instructions):
        """LLM 一返回就发布机械指令（在 TTS 播放之前），让动作和语音基本同步开始。

        音箱指令（播放歌曲）由 brain_node 本地执行，不下发 /command——control_node
        没有音频通路。此处只把它挑出来暂存，等 TTS 说完再放，避免抢占扬声器。
        """
        self._pending_songs = [
            c.get("params", {}).get("song")
            for c in (instructions or [])
            if c.get("actuator") == "音箱" and c.get("params", {}).get("song")
        ]
        remote = [c for c in (instructions or []) if c.get("actuator") != "音箱"]

        if remote:
            payload = json.dumps(remote, ensure_ascii=False)
            self.get_logger().info(f"发布指令: {payload}")
            self._instr_pub.publish(String(data=payload))
        elif not self._pending_songs:
            self.get_logger().info("无机械指令")

    def _play_pending_songs(self):
        """播放暂存的歌曲。在 TTS 播完之后调用，阻塞直到放完或达上限。

        期间麦克风由 /voice/speaking 的 start 信号压着（见 _work_loop 的
        try/finally），因此不会把歌声当成用户指令收回来。
        """
        songs, self._pending_songs = self._pending_songs, []
        for name in songs:
            self.get_logger().info(f"播放歌曲: {name}")
            try:
                if not play_song(name):
                    self.get_logger().warn(f"播放失败: {name}")
            except Exception as e:
                self.get_logger().error(f"播放异常 {name}: {e}")

    def _work_loop(self):
        _INTERVENTION_TTL = 3.0   # 干预消息在队列中最长存活时间（秒）

        while True:
            item = self._work_q.get()

            # 结构化任务（紧急联络询问）用 dict，避免把字段拼进字符串再切开——
            # emotion_zh 来自外部 JSON，含冒号就会把解析切错
            if isinstance(item, dict):
                if item.get("type") == "emergency_ask":
                    try:
                        self._run_emergency_ask(item)
                    except Exception as e:
                        self.get_logger().error(f"紧急联络询问失败: {e}")
                continue

            cmd = item
            if cmd.startswith(_SPEAK_PREFIX):
                self._speak_fixed(cmd[len(_SPEAK_PREFIX):])
                continue

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
            # 出声前先闭麦。放在 try 之外确保与 finally 的 end 严格配对；
            # LLM 思考期间一并闭着，用户指令那条路本来就已自锁，无差别。
            self._speaking_pub.publish(String(data="start"))
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

                # TTS 已播完，此时才放歌，避免与语音抢扬声器
                self._play_pending_songs()

                # 保存对话历史 + 提取记忆（仅普通对话）
                if not is_intervention and spoken:
                    self._mem.add_turn(cmd, spoken)
                    self._mem.extract_and_save(cmd, spoken)

            except Exception as e:
                self.get_logger().error(f"处理指令失败: {e}")
            finally:
                # 两条都必须无条件发出，否则 voice_node 永久静音（卡死）：
                # end 松开播报闸，_check_question_mode 松开轮次闸，缺一不可。
                self._speaking_pub.publish(String(data="end"))
                self._check_question_mode(spoken)

    def _speak_fixed(self, text: str):
        """播一句固定话术，不走 LLM。

        必须和 _work_loop 共用同一个线程（都在队列里排队），否则两条播报路径
        会各自发一对 /voice/speaking start/end：嵌套的那对 end 一到，
        voice_node 就在另一条还在出声时把麦克风打开，机器人听自己说话。
        """
        self.get_logger().info(f"语音回复: {text}")
        self._speaking_pub.publish(String(data="start"))
        try:
            from voice_brain_module.tts import stream_play
            stream_play(text)
        except Exception as e:
            self.get_logger().error(f"播报失败: {e}")
        finally:
            self._speaking_pub.publish(String(data="end"))
            # 把话术原文交给 _check_question_mode：紧急联络的询问句以"吗"结尾，
            # 它会据此发 continuous，让接下来的等待期免唤醒词直接收音。
            # 陈述句（如中止确认）走 command 分支，与原行为一致。
            self._check_question_mode(text)

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
