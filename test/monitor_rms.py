#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监测麦克风 RMS 能量值
用于校准 config.py 中的 SPEECH_DELTA 参数

运行：python3 monitor_rms.py
退出：Ctrl+C
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
from realtime_asr.config import (
    DEVICE_INDEX, SAMPLE_RATE, HW_SAMPLE_RATE, CHUNK,
    NOISE_INIT_SEC, NOISE_ALPHA, SPEECH_DELTA,
)

BAR_WIDTH       = 40
RMS_DISPLAY_MAX = 20000


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2) * (32768 ** 2)))


def db(r: float) -> float:
    return 20 * np.log10(max(r, 1.0))


def draw_bar(value: float, max_value: float, width: int, threshold: float) -> str:
    filled = int(min(value / max_value, 1.0) * width)
    bar = "#" * filled + "-" * (width - filled)
    t_pos = int(min(threshold / max_value, 1.0) * width)
    bar = bar[:t_pos] + "|" + bar[t_pos + 1:]
    return bar


def calibrate(stream) -> float:
    print(f"🔇 校准底噪，请保持安静（{NOISE_INIT_SEC:.1f} 秒）...")
    init_chunks = int(NOISE_INIT_SEC * SAMPLE_RATE / CHUNK)
    samples = []
    for _ in range(init_chunks):
        data, _ = stream.read(CHUNK * 3)
        chunk_16k = scipy_signal.resample_poly(data[:, 0], up=1, down=3).astype(np.float32)
        samples.append(rms(chunk_16k))
    return float(np.median(samples)) if samples else 0.0


def main():
    print(f"设备 Index={DEVICE_INDEX}，硬件采样率={HW_SAMPLE_RATE}Hz → 重采样至 {SAMPLE_RATE}Hz")
    print(f"SPEECH_DELTA={SPEECH_DELTA}（config.py），NOISE_ALPHA={NOISE_ALPHA}")
    print("按 Ctrl+C 退出\n")

    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=1,
        samplerate=HW_SAMPLE_RATE,
        dtype="float32",
        blocksize=CHUNK * 3,
    ) as stream:
        noise_floor = calibrate(stream)
        threshold   = noise_floor + SPEECH_DELTA
        print(f"📊 底噪 RMS={noise_floor:.0f}（{db(noise_floor):.1f} dB）")
        print(f"📊 检测阈值 RMS={threshold:.0f}（{db(threshold):.1f} dB）\n")

        while True:
            data, _ = stream.read(CHUNK * 3)
            chunk_16k = scipy_signal.resample_poly(data[:, 0], up=1, down=3).astype(np.float32)
            r = rms(chunk_16k)

            if r <= threshold:
                noise_floor = NOISE_ALPHA * r + (1 - NOISE_ALPHA) * noise_floor
                threshold   = noise_floor + SPEECH_DELTA

            bar    = draw_bar(r, RMS_DISPLAY_MAX, BAR_WIDTH, threshold)
            status = "语音" if r > threshold else "静音"
            print(
                f"\r[{bar}] RMS={r:6.0f} ({db(r):5.1f}dB)  "
                f"阈={threshold:5.0f} 底噪={noise_floor:5.0f}  {status}   ",
                end="", flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
