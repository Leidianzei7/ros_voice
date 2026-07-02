#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频管道：开麦 → 重采样到 16k → VAD → ASR → 通过回调吐出文本。

run_audio_pipeline(on_asr_text, log, running)
    on_asr_text: Callable[[str], None] — 每段识别结果（VAD 切完后）的回调
    log:         Callable[[str], None] — 日志函数（print 或 ros logger）
    running:     threading.Event — 主循环控制信号（必须传入）

使用 PortAudio 回调 + 外层重试恢复 ALSA xrun。
"""
import queue
import threading
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
from .config import (
    DEVICE_NAME, SAMPLE_RATE, HW_SAMPLE_RATE, CHANNELS, CHUNK,
    NOISE_INIT_SEC, SPEECH_DELTA, VAD_MODE,
    resolve_device,
)
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
    _drain(audio_q)

    log(f"🔇 校准中（{NOISE_INIT_SEC:.1f}秒）...")
    init_chunks = int(NOISE_INIT_SEC * SAMPLE_RATE / CHUNK)
    samples = []
    for _ in range(init_chunks):
        try:
            chunk = audio_q.get(timeout=1.0)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2) * (32768 ** 2))
            samples.append(rms)
        except queue.Empty:
            pass

    noise_floor = float(np.median(samples)) if samples else 500.0
    threshold = noise_floor + SPEECH_DELTA
    db = lambda r: 20 * np.log10(max(r, 1))
    log(f"📊 底噪 {db(noise_floor):.1f} dB，检测阈值 {db(threshold):.1f} dB")

    stream_play("校准完成，可以开始说话了")
    _drain(audio_q)
    return noise_floor


def run_audio_pipeline(on_asr_text, log=print, running=None, active=None):
    assert running is not None, "running 参数必须传入 threading.Event"
    if active is None:
        active = running

    import time as _time

    while running.is_set():
        raw_q    = queue.Queue(maxsize=100)
        audio_q  = queue.Queue()
        status_q = queue.Queue()
        stream_dead = threading.Event()   # 看门狗：流无响应时由 _resample_worker 设置

        cb_total    = [0]
        cb_overflow = [0]
        last_report = [_time.monotonic()]

        def _callback(indata, frames, time_info, status):
            cb_total[0] += 1
            if status:
                if status.input_overflow:
                    cb_overflow[0] += 1
                else:
                    status_q.put_nowait(str(status))
            try:
                raw_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        def _resample_worker():
            last_frame = _time.monotonic()
            while running.is_set() and not stream_dead.is_set():
                while not status_q.empty():
                    try:
                        log(f"[音频状态] {status_q.get_nowait()}")
                    except queue.Empty:
                        break
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
                # TTS 期间：从源头截断，排空两个队列，VAD 自然得不到数据
                if not active.is_set():
                    _drain(raw_q)
                    _drain(audio_q)
                    active.wait(0.1)
                    last_frame = _time.monotonic()  # mute 期间重置看门狗
                    continue
                try:
                    raw = raw_q.get(timeout=0.1)
                    last_frame = _time.monotonic()
                    chunk = scipy_signal.resample_poly(raw, up=1, down=3).astype(np.float32)
                    audio_q.put(chunk)
                except queue.Empty:
                    # 3 秒收不到帧：ALSA 流已在 C 层崩溃，Python 感知不到异常
                    if _time.monotonic() - last_frame > 3.0:
                        log("[音频] 3 秒无音频帧，ALSA 流可能已崩溃，触发重启...")
                        stream_dead.set()

        threading.Thread(target=_resample_worker, daemon=True).start()

        try:
            with sd.InputStream(
                device=resolve_device(DEVICE_NAME, "input"),
                samplerate=HW_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK * 3,
                latency='high',
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

                run_vad(audio_q, running, _on_speech, log, noise_floor,
                        stream_dead=stream_dead)
        except Exception as e:
            log(f"[音频] 流异常: {e}，1 秒后重试...")
        running.wait(1.0)
