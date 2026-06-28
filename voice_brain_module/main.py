#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone 运行入口（无 ROS）。

模块内聚：
- 全局状态（running, tts_text_q, waiting_for_command）仅在此文件
- 唤醒词 + LLM 派发逻辑（handle_asr_result, _dispatch_command）仅此文件
- TTS 消费线程（tts_playback_thread）仅此文件

ROS 路径（voice_node / brain_node）有自己的状态管理和派发逻辑，
不走此文件。
"""
import sys
import json
import queue
import threading

from .config import (
    DEVICE_NAME, OUTPUT_DEVICE_NAME, SAMPLE_RATE,
    WAKE_WORD, TTS_VOICE, resolve_device,
)
from .audio import run_audio_pipeline
from .wake_word import find_wake_word
from .llm import generate_response
from .tts import stream_play

# ── 全局状态（仅 standalone 模式使用）─────────────────────────
tts_text_q = queue.Queue()
running = threading.Event()
running.set()
waiting_for_command = False


# ── TTS 消费线程 ──────────────────────────────────────────────

def tts_playback_thread():
    while running.is_set():
        try:
            text = tts_text_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            stream_play(text)
        except Exception as e:
            print(f"[TTS 错误] {e}", file=sys.stderr)


# ── 唤醒词 + LLM 派发 ────────────────────────────────────────

def _dispatch_command(cmd):
    print(f"📋 指令原文: {cmd}", flush=True)
    print("⏳ 解析中...", flush=True)
    spoken, commands = generate_response(cmd)
    if spoken:
        print(f"🔊 语音回复：{spoken}", flush=True)
        tts_text_q.put(spoken)
    if commands:
        print(f"\n✅ 标准化指令:\n{json.dumps(commands, ensure_ascii=False, indent=2)}\n", flush=True)
    elif not spoken:
        print("[警告] LLM 回复解析失败", flush=True)


def handle_asr_result(text):
    global waiting_for_command
    pos, ww_len = find_wake_word(text)
    if pos >= 0:
        cmd = text[pos + ww_len:].strip("，。,.： ")
        if cmd:
            _dispatch_command(cmd)
        else:
            print("👂 已唤醒，请说出指令...", flush=True)
            waiting_for_command = True
    elif waiting_for_command:
        waiting_for_command = False
        _dispatch_command(text)


# ── 入口 ─────────────────────────────────────────────────────

def main():
    mic_idx = resolve_device(DEVICE_NAME, "input")
    spk_idx = resolve_device(OUTPUT_DEVICE_NAME, "output")
    print(f"麦克风: [{mic_idx}] {DEVICE_NAME} | 扬声器: [{spk_idx}] {OUTPUT_DEVICE_NAME}")
    print(f"采样率: {SAMPLE_RATE} Hz | 阈值: 动态自适应")
    print(f"唤醒词: 【{WAKE_WORD}】 | TTS 音色: {TTS_VOICE}")
    print("─" * 50)
    print("说出唤醒词后下达指令，按 Ctrl+C 停止...\n")

    tts_thread = threading.Thread(target=tts_playback_thread, daemon=True)
    tts_thread.start()

    try:
        run_audio_pipeline(on_asr_text=handle_asr_result, running=running)
    except KeyboardInterrupt:
        print("\n\n停止识别。")
    except Exception as e:
        print(f"\n音频设备错误: {e}")
    finally:
        running.clear()
        tts_thread.join(timeout=3)


if __name__ == "__main__":
    main()
