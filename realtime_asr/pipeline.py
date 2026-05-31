#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高层管道接口：供 ros_voice 等纯 ROS 层调用，屏蔽 ASR/LLM/TTS 内部细节。

run_command_pipeline(on_command, log, running)
    采音 + VAD + ASR + 唤醒词识别一体化。每检测到一条用户指令（唤醒词后的内容
    或唤醒后单独说出的整句）通过 on_command(cmd_text) 回调。

process_command(cmd_text, log)
    LLM 推理 + 语音回复播报一体化。返回机械指令列表（可能为空）。
"""
from .audio import run_audio_pipeline
from .wake_word import find_wake_word
from .llm import generate_response
from .tts import stream_play


def run_command_pipeline(on_command, log=print, running=None):
    waiting = {"flag": False}

    def _on_text(text):
        pos, ww_len = find_wake_word(text)
        if pos >= 0:
            cmd = text[pos + ww_len:].strip("，。,.： ")
            if cmd:
                on_command(cmd)
            else:
                log("已唤醒，等待指令...")
                waiting["flag"] = True
        elif waiting["flag"]:
            waiting["flag"] = False
            on_command(text)

    run_audio_pipeline(on_asr_text=_on_text, log=log, running=running)


def process_command(cmd_text, log=print):
    spoken, commands = generate_response(cmd_text)
    if spoken:
        log(f"语音回复: {spoken}")
        stream_play(spoken)
    return commands
