#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频管道：开麦 → 重采样到 16k → VAD → ASR → 通过回调吐出文本。

run_audio_pipeline(on_asr_text, log, running)
    on_asr_text: Callable[[str], None] — 每段识别结果（VAD 切完后）的回调
    log:         Callable[[str], None] — 日志函数（print 或 ros logger）
    running:     threading.Event — 主循环控制信号（必须传入）

使用阻塞 read() 而非 PortAudio callback：回调模式下 ALSA xrun 可能在 C
层卡死 WaitForFrames 且 Python 无法中断；阻塞模式每次 read 都有超时，
卡住即可被检测并重试恢复。
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


def _calibrate_noise(stream, log):
    from .tts import stream_play
    log("🔇 即将校准底噪，请保持安静...")
    stream_play("请保持安静，正在校准底噪")
    # 丢掉 TTS 期间采集到的回声
    try:
        while True:
            stream.read(CHUNK * 3)
    except sd.PortAudioError:
        pass

    log(f"🔇 校准中（{NOISE_INIT_SEC:.1f}秒）...")
    init_chunks = int(NOISE_INIT_SEC * SAMPLE_RATE / CHUNK)
    samples = []
    for _ in range(init_chunks):
        try:
            indata, _ = stream.read(CHUNK * 3)
        except sd.PortAudioError:
            continue
        chunk = scipy_signal.resample_poly(indata[:, 0], up=1, down=3).astype(np.float32)
        rms = np.sqrt(np.mean(chunk ** 2) * (32768 ** 2))
        samples.append(rms)

    noise_floor = float(np.median(samples)) if samples else 500.0
    threshold = noise_floor + SPEECH_DELTA
    db = lambda r: 20 * np.log10(max(r, 1))
    log(f"📊 底噪 {db(noise_floor):.1f} dB，检测阈值 {db(threshold):.1f} dB")

    stream_play("校准完成，可以开始说话了")
    return noise_floor


def _stream_worker(stream, audio_q, running, log):
    """阻塞读取音频帧，重采样后放入 audio_q，供 VAD 消费。"""
    import time as _time
    overflow_count = 0
    last_report = _time.monotonic()

    while running.is_set():
        try:
            indata, _ = stream.read(CHUNK * 3)
        except sd.PortAudioError as e:
            log(f"[音频] 读取异常: {e}")
            continue

        chunk = scipy_signal.resample_poly(indata[:, 0], up=1, down=3).astype(np.float32)
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            overflow_count += 1

        now = _time.monotonic()
        if now - last_report >= 10.0 and overflow_count > 0:
            log(f"[音频] 10s 内丢帧 {overflow_count} 次")
            overflow_count = 0
            last_report = now


def run_audio_pipeline(on_asr_text, log=print, running=None):
    assert running is not None, "running 参数必须传入 threading.Event"

    while running.is_set():
        try:
            with sd.InputStream(
                device=resolve_device(DEVICE_NAME, "input"),
                samplerate=HW_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK * 3,
                latency='high',
            ) as stream:
                audio_q = queue.Queue(maxsize=200)

                noise_floor = 0.0
                if VAD_MODE != "webrtc":
                    noise_floor = _calibrate_noise(stream, log)

                worker = threading.Thread(
                    target=_stream_worker,
                    args=(stream, audio_q, running, log),
                    daemon=True,
                )
                worker.start()

                def _on_speech(audio):
                    log("⏳ 识别中...")
                    text = recognize(audio)
                    if text.strip():
                        log(f"🗣  {text}")
                        on_asr_text(text)
                    else:
                        log("（未识别到有效内容）")

                run_vad(audio_q, running, _on_speech, log, noise_floor)
        except Exception as e:
            log(f"[音频] 流异常: {e}，1 秒后重试...")
            running.wait(1.0)
