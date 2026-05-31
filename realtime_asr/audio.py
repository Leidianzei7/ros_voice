#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频管道：开麦 → 重采样到 16k → VAD → ASR → 通过回调吐出文本。

run_audio_pipeline(on_asr_text, log, running)
    on_asr_text: Callable[[str], None] — 每段识别结果（VAD 切完后）的回调
    log:         Callable[[str], None] — 日志函数（print 或 ros logger）
    running:     threading.Event — 为 None 时使用 state.running

handle_asr_result / _dispatch_command 是 main.py 用的"唤醒词 + LLM 派发"逻辑，
ros_voice/voice_node.py 不应使用，它有自己的 ROS topic 派发逻辑。
"""
import sys
import json
import queue
import threading
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
from .config import (
    DEVICE_INDEX, SAMPLE_RATE, HW_SAMPLE_RATE, CHANNELS, CHUNK,
    NOISE_INIT_SEC, SPEECH_DELTA, VAD_MODE,
)
from . import state as _state
from .asr import recognize
from .vad import run_vad


def _drain(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _calibrate_noise(audio_q, log):
    from .tts import stream_play
    log("🔇 即将校准底噪，请保持安静...")
    stream_play("请保持安静，正在校准底噪")
    _drain(audio_q)  # 丢掉 TTS 期间采集到的回声

    log(f"🔇 校准中（{NOISE_INIT_SEC:.1f}秒）...")
    init_chunks = int(NOISE_INIT_SEC * SAMPLE_RATE / CHUNK)
    samples = []
    for _ in range(init_chunks):
        try:
            chunk = audio_q.get(timeout=1.0)
            rms   = np.sqrt(np.mean(chunk.astype(np.float32) ** 2) * (32768 ** 2))
            samples.append(rms)
        except queue.Empty:
            pass

    noise_floor = float(np.median(samples)) if samples else 500.0
    threshold   = noise_floor + SPEECH_DELTA
    db          = lambda r: 20 * np.log10(max(r, 1))
    log(f"📊 底噪 {db(noise_floor):.1f} dB，检测阈值 {db(threshold):.1f} dB")

    stream_play("校准完成，可以开始说话了")
    _drain(audio_q)
    return noise_floor


def run_audio_pipeline(on_asr_text, log=print, running=None):
    if running is None:
        running = _state.running

    # raw_q：回调写入 48kHz 原始帧（轻量，不做任何计算）
    # audio_q：重采样线程写入 16kHz 帧，供 VAD 消费
    raw_q   = queue.Queue(maxsize=100)
    audio_q = queue.Queue()

    import time as _time
    status_q = queue.Queue()           # 非 overflow 的真实错误，丢给工作线程打印
    cb_total    = [0]                  # 总回调次数
    cb_overflow = [0]                  # 其中 overflow 次数
    last_report = [_time.monotonic()]

    def _callback(indata, frames, time_info, status):
        cb_total[0] += 1
        if status:
            if status.input_overflow:
                cb_overflow[0] += 1   # 只计数，回调内不打印
            else:
                status_q.put_nowait(str(status))
        try:
            raw_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass   # 重采样线程跟不上时丢帧，优先保证回调不阻塞

    def _resample_worker():
        while running.is_set():
            # 真实错误立即打印
            while not status_q.empty():
                log(f"[音频状态] {status_q.get_nowait()}")
            # overflow 每 10 秒汇总一次，报百分比让用户自己判断
            now = _time.monotonic()
            if now - last_report[0] >= 10.0:
                total, oflw = cb_total[0], cb_overflow[0]
                if oflw > 0 and total > 0:
                    pct = 100.0 * oflw / total
                    tip = "（<5% 可忽略）" if pct < 5.0 else ""
                    log(f"[音频] 10s 内丢帧 {oflw}/{total} 回调 ({pct:.1f}%) {tip}")
                cb_total[0] = 0
                cb_overflow[0] = 0
                last_report[0] = now

            try:
                raw = raw_q.get(timeout=0.1)
                chunk = scipy_signal.resample_poly(raw, up=1, down=3).astype(np.float32)
                audio_q.put(chunk)
            except queue.Empty:
                continue

    threading.Thread(target=_resample_worker, daemon=True).start()

    with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=HW_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK * 3,
        latency='high',        # 让驱动保留更大内部缓冲，减少 overflow
        callback=_callback,
    ):
        noise_floor = 0.0
        if VAD_MODE != "webrtc":
            noise_floor = _calibrate_noise(audio_q, log)

        def _on_speech(audio):
            log("⏳ 识别中...")
            text = recognize(audio)
            if text.strip():
                log(f"🗣  {text}")
                on_asr_text(text)
            else:
                log("（未识别到有效内容）")

        run_vad(audio_q, running, _on_speech, log, noise_floor)


# ── 仅 main.py 使用：唤醒词 + LLM 派发 ───────────────────────
from .wake_word import find_wake_word
from .llm import generate_response


def _dispatch_command(cmd):
    print(f"📋 指令原文: {cmd}", flush=True)
    print("⏳ 解析中...", flush=True)
    spoken, commands = generate_response(cmd)
    if spoken:
        print(f"🔊 语音回复：{spoken}", flush=True)
        _state.tts_text_q.put(spoken)
    if commands:
        print(f"\n✅ 标准化指令:\n{json.dumps(commands, ensure_ascii=False, indent=2)}\n", flush=True)
    elif not spoken:
        print("[警告] LLM 回复解析失败", flush=True)


def handle_asr_result(text):
    pos, ww_len = find_wake_word(text)
    if pos >= 0:
        cmd = text[pos + ww_len:].strip("，。,.： ")
        if cmd:
            _dispatch_command(cmd)
        else:
            print("👂 已唤醒，请说出指令...", flush=True)
            _state.waiting_for_command = True
    elif _state.waiting_for_command:
        _state.waiting_for_command = False
        _dispatch_command(text)
