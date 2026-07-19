#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
歌曲播放：从曲库读取音频文件，重采样后送扬声器。

list_songs()        — 扫描曲库，返回歌名列表（文件名去扩展名）
find_song(name)     — 按歌名查文件，支持模糊匹配
play_song(name)     — 播放（阻塞），最长 SONG_MAX_SEC，超时淡出

设计要点：
- 与 TTS 共用同一个输出设备与 sounddevice 通路，避免设备争用
- 播放期间麦克风本就处于静音态（brain_node 处理完才会发 listen_mode 解除），
  因此不会听见自己放的歌，无需额外静音逻辑
- 播放是阻塞的：一分钟上限由 SONG_MAX_SEC 兜底，不做打断
"""
import os
import sys

import numpy as np
import sounddevice as sd

from .config import (
    SONG_DIR, SONG_MAX_SEC, HW_SAMPLE_RATE, OUTPUT_DEVICE_NAME,
    CHUNK, resolve_device,
)

_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")


def list_songs() -> list[str]:
    """扫描曲库返回歌名（文件名去扩展名），按名称排序。目录不存在时返回空表。"""
    if not os.path.isdir(SONG_DIR):
        return []
    names = []
    for fn in os.listdir(SONG_DIR):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in _EXTS and stem.strip():
            names.append(stem)
    return sorted(set(names))


def find_song(name: str) -> str | None:
    """按歌名找文件路径。先精确匹配，再子串模糊匹配。找不到返回 None。"""
    if not name or not os.path.isdir(SONG_DIR):
        return None
    files = [f for f in os.listdir(SONG_DIR)
             if os.path.splitext(f)[1].lower() in _EXTS]
    target = name.strip().strip("《》\"' ")

    for f in files:                                  # 精确
        if os.path.splitext(f)[0] == target:
            return os.path.join(SONG_DIR, f)
    for f in files:                                  # 模糊（互为子串）
        stem = os.path.splitext(f)[0]
        if target and (target in stem or stem in target):
            return os.path.join(SONG_DIR, f)
    return None


def _load(path: str) -> np.ndarray | None:
    """解码为 HW_SAMPLE_RATE 单声道 float32；失败返回 None。"""
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)                           # 混为单声道
    except Exception as e:
        print(f"[播放] 解码失败 {os.path.basename(path)}: {e}", file=sys.stderr)
        return None

    if sr != HW_SAMPLE_RATE:
        try:
            from scipy import signal as scipy_signal
            from math import gcd
            g = gcd(int(sr), int(HW_SAMPLE_RATE))
            y = scipy_signal.resample_poly(
                y, up=int(HW_SAMPLE_RATE) // g, down=int(sr) // g
            ).astype(np.float32)
        except Exception as e:
            print(f"[播放] 重采样失败: {e}", file=sys.stderr)
            return None

    limit = int(SONG_MAX_SEC * HW_SAMPLE_RATE)
    if len(y) > limit:                               # 超时截断并淡出，避免爆音
        fade = min(int(1.5 * HW_SAMPLE_RATE), limit)
        y = y[:limit].copy()
        y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return y


def play_song(name: str) -> bool:
    """播放指定歌曲（阻塞至放完或达上限）。成功返回 True。"""
    path = find_song(name)
    if not path:
        print(f"[播放] 曲库中没有《{name}》", file=sys.stderr)
        return False

    y = _load(path)
    if y is None or len(y) == 0:
        return False

    try:
        with sd.OutputStream(
            samplerate=HW_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=resolve_device(OUTPUT_DEVICE_NAME, "output"),
            blocksize=CHUNK * 3,
        ) as stream:
            step = CHUNK * 3
            for i in range(0, len(y), step):
                stream.write(y[i:i + step])
        return True
    except Exception as e:
        print(f"[播放] 输出失败: {e}", file=sys.stderr)
        return False
