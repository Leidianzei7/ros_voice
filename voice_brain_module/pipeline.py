#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高层管道接口：供 ros_voice 等纯 ROS 层调用，屏蔽 ASR/LLM/TTS 内部细节。

listen_for_commands(on_command, log, running, wake_required_ref)
    采音 + VAD + ASR + 唤醒词识别一体化。
    wake_required_ref: {"wake_required": bool} — 为 False 时跳过唤醒词，
                       每句话都直接回调 on_command。

process_command(cmd_text, log, vision_context)
    LLM 推理 + 语音回复播报一体化。返回 (指令列表, 口语回复文本)。

注意：各函数内部按需导入，避免 brain_node 导入 pipeline 时连带加载 ASR 模型。
"""
# 模块级不 import，按需在各函数内惰性导入


def listen_for_commands(on_command, log=print, running=None,
                        wake_required_ref=None, active=None):
    from .audio import run_audio_pipeline
    from .wake_word import find_wake_word

    if wake_required_ref is None:
        wake_required_ref = {"wake_required": True}

    waiting = {"flag": False}

    def _on_text(text):
        # 持续监听模式：每句话直接当指令
        if not wake_required_ref["wake_required"]:
            on_command(text)
            return

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

    run_audio_pipeline(on_asr_text=_on_text, log=log, running=running,
                       active=active)


def process_command(cmd_text, log=print, vision_context=""):
    from .llm import generate_response
    from .tts import stream_play

    spoken, commands = generate_response(cmd_text, vision_context=vision_context)
    if spoken:
        log(f"语音回复: {spoken}")
        stream_play(spoken)
    return commands, spoken
