#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD（语音活动检测）核心循环，供 audio.py 和 ros_voice/voice_node.py 共用。

run_vad(audio_q, running, on_speech, log, noise_floor)
    audio_q:     queue.Queue  — 提供 float32 音频块
    running:     threading.Event — 为 False 时退出循环
    on_speech:   Callable[[np.ndarray], None] — 收到完整语音段时回调
    log:         Callable[[str], None] — 状态日志（print 或 ros logger）
    noise_floor: float — 预测量的底噪 RMS（energy 模式用；webrtc 模式忽略）
"""
import queue
import collections
import numpy as np
from .config import (
    SAMPLE_RATE, CHUNK, SPEECH_HOLD_SEC, MIN_SPEECH_SEC, VAD_MODE,
    NOISE_ALPHA, SPEECH_DELTA,
    VAD_AGGRESSIVENESS, VAD_FRAME_SAMPLES, VAD_FRAME_MS, VAD_SPEECH_TRIGGER,
)


def run_vad(audio_q, running, on_speech, log, noise_floor=500.0):
    if VAD_MODE == "webrtc":
        _run_webrtc(audio_q, running, on_speech, log)
    else:
        _run_energy(audio_q, running, on_speech, log, noise_floor)


def _run_energy(audio_q, running, on_speech, log, noise_floor):
    threshold     = noise_floor + SPEECH_DELTA
    silence_limit = int(SPEECH_HOLD_SEC * SAMPLE_RATE / CHUNK)

    speech_buf  = np.array([], dtype=np.float32)
    in_speech   = False
    silence_cnt = 0

    while running.is_set():
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        chunk = chunk.astype(np.float32)
        rms   = np.sqrt(np.mean(chunk ** 2) * (32768 ** 2))

        if rms > threshold:
            if not in_speech:
                in_speech  = True
                speech_buf = chunk.copy()
                log("正在聆听...")
            else:
                speech_buf = np.concatenate([speech_buf, chunk])
            silence_cnt = 0
        else:
            if not in_speech:
                noise_floor = NOISE_ALPHA * rms + (1 - NOISE_ALPHA) * noise_floor
                threshold   = noise_floor + SPEECH_DELTA
            if in_speech:
                speech_buf  = np.concatenate([speech_buf, chunk])
                silence_cnt += 1
                if silence_cnt >= silence_limit:
                    in_speech   = False
                    silence_cnt = 0
                    if len(speech_buf) / SAMPLE_RATE >= MIN_SPEECH_SEC:
                        on_speech(speech_buf)
                    speech_buf = np.array([], dtype=np.float32)


def _run_webrtc(audio_q, running, on_speech, log):
    import webrtcvad
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    pre_roll      = int(200 / VAD_FRAME_MS)
    pre_buf       = collections.deque(maxlen=pre_roll)
    silence_limit = int(SPEECH_HOLD_SEC * 1000 / VAD_FRAME_MS)

    speech_frames = []
    in_speech     = False
    silence_cnt   = 0
    speech_cnt    = 0
    sample_buf    = np.array([], dtype=np.float32)

    log("WebRTC VAD 就绪，开始监听")

    while running.is_set():
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        sample_buf = np.concatenate([sample_buf, chunk.astype(np.float32)])

        while len(sample_buf) >= VAD_FRAME_SAMPLES:
            frame      = sample_buf[:VAD_FRAME_SAMPLES]
            sample_buf = sample_buf[VAD_FRAME_SAMPLES:]
            pcm        = (frame * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
            is_speech  = vad.is_speech(pcm, SAMPLE_RATE)

            if is_speech:
                speech_cnt += 1
                silence_cnt = 0
                if not in_speech:
                    pre_buf.append(frame)  # 暂存触发窗口内的语音帧，防止起始音节丢失
                    if speech_cnt >= VAD_SPEECH_TRIGGER:
                        in_speech     = True
                        speech_frames = list(pre_buf)
                        log("正在聆听...")
                else:
                    speech_frames.append(frame)
            else:
                speech_cnt = 0
                if in_speech:
                    speech_frames.append(frame)
                    silence_cnt += 1
                    if silence_cnt >= silence_limit:
                        in_speech     = False
                        silence_cnt   = 0
                        audio         = np.concatenate(speech_frames)
                        speech_frames = []
                        if len(audio) / SAMPLE_RATE >= MIN_SPEECH_SEC:
                            on_speech(audio)
                else:
                    pre_buf.append(frame)
