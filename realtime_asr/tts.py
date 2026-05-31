#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback
from .config import TTS_VOICE, TTS_SAMPLE_RATE, HW_SAMPLE_RATE, OUTPUT_DEVICE_INDEX, LLM_API_KEY, CHUNK
from . import state as _state

dashscope.api_key = LLM_API_KEY


class _Callback(ResultCallback):
    def __init__(self, pcm_q):
        self._q = pcm_q

    def on_open(self): pass
    def on_close(self): pass

    def on_complete(self):
        self._q.put(None)

    def on_error(self, response):
        print(f"[TTS 错误] {response}", file=sys.stderr)
        self._q.put(None)

    def on_data(self, data: bytes):
        if data:
            self._q.put(data)


def stream_play(text):
    pcm_q = queue.Queue()
    syn = SpeechSynthesizer(
        model="cosyvoice-v2",
        voice=TTS_VOICE,
        format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        callback=_Callback(pcm_q),
    )

    def _safe_call():
        try:
            syn.call(text)
        except Exception as e:
            print(f"[TTS 错误] {e}", file=sys.stderr)
            pcm_q.put(None)  # 仅在异常时手动解锁，正常结束由 on_complete 负责

    threading.Thread(target=_safe_call, daemon=True).start()

    with sd.OutputStream(
        samplerate=HW_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=OUTPUT_DEVICE_INDEX,
        blocksize=CHUNK * 3,
    ) as stream:
        while True:
            chunk = pcm_q.get()
            if chunk is None:
                break
            pcm     = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            pcm_48k = scipy_signal.resample_poly(pcm, up=3, down=1)
            pcm_48k = pcm_48k.clip(-32768, 32767).astype(np.int16)
            stream.write(pcm_48k)


def tts_playback_thread():
    while _state.running.is_set():
        try:
            text = _state.tts_text_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            stream_play(text)
        except Exception as e:
            print(f"[TTS 错误] {e}", file=sys.stderr)
