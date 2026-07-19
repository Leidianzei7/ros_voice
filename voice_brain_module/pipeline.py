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
    from .config import is_meaningful

    if wake_required_ref is None:
        wake_required_ref = {"wake_required": True}

    waiting = {"flag": False}

    def _on_text(text):
        # 兜底：正常情况下 audio 层已在 _on_speech 里过滤掉纯标点/单字碎片
        if not is_meaningful(text):
            return

        # 持续监听模式：每句话直接当指令
        if not wake_required_ref["wake_required"]:
            active.clear()   # 立刻静音，不等 brain_node 的 mute 话题回来
            on_command(text)
            return

        pos, ww_len = find_wake_word(text)
        if pos >= 0:
            cmd = text[pos + ww_len:].strip("，。,.： ")
            if cmd:
                active.clear()
                on_command(cmd)
            else:
                log("已唤醒，等待指令...")
                waiting["flag"] = True
        elif waiting["flag"]:
            waiting["flag"] = False
            active.clear()
            on_command(text)

    run_audio_pipeline(on_asr_text=_on_text, log=log, running=running,
                       active=active)


def process_command(cmd_text, log=print, vision_context="", on_instructions=None,
                    last_question=""):
    """LLM 推理 + 语音回复播报一体化。

    on_instructions: 可选回调，LLM 一返回就调用（在 TTS 播放之前），
                     让机械动作和语音播报基本同时开始，而不必等一整句话说完。
    last_question:   机器人上一轮问用户的问题。若设置了该参数，会先用轻量 LLM
                     判断用户的语音是否在回答机器人；若判定用户在对别人说话，
                     则直接回复"请问您在和我讲话吗？"并跳过主 LLM 调用。
    """
    from .llm import generate_response, is_addressing_robot, classify_meta_response
    from .tts import stream_play

    _META_QUESTION = "请问您在和我讲话吗？"

    # 元对话处理：机器人刚问了"请问您在和我讲话吗？"
    # 使用三分类器（单次 LLM 调用）避免 is_addressing_robot 的悖论问题：
    # "没有，你一边去" — 用户在"对机器人"说"没在跟机器人说话"
    # 最多问一遍：NOT_ADDRESSING 和 REJECTING 都直接结束，避免反复追问。
    if last_question == _META_QUESTION:
        intent = classify_meta_response(cmd_text)
        if intent in ("NOT_ADDRESSING", "REJECTING"):
            spoken = "看起来您有些忙，等您有空了再叫我吧。"
            log(f"元对话结束 (意图={intent}): {spoken}")
            stream_play(spoken)
            return [], spoken
        # intent == "ADDRESSING": 用户确认在对话，走正常 LLM 流程

    # 常规检查：机器人问了普通问题但用户回答不像是给自己的
    elif last_question and not is_addressing_robot(cmd_text, last_question):
        spoken = _META_QUESTION
        log(f"检测到用户可能不在对话，回复: {spoken}")
        stream_play(spoken)
        return [], spoken

    spoken, commands = generate_response(cmd_text, vision_context=vision_context)
    if on_instructions:
        on_instructions(commands)
    if spoken:
        log(f"语音回复: {spoken}")
        stream_play(spoken)
    return commands, spoken
