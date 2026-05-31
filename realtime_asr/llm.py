#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
from openai import OpenAI
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .commands import build_system_prompt, validate_commands

llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

_SYSTEM_PROMPT = build_system_prompt()
_SPOKEN_PREFIX = "[口语回复]："
_INSTR_PREFIX  = "[执行指令]："

_MAX_RETRIES = 2   # 校验失败时最多让 LLM 重新生成的次数


def generate_response(
    text: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> tuple[str, list[dict]]:
    """
    调用 LLM，返回 (口语回复, 指令列表)。
    若 LLM 输出的指令未通过 commands.validate_commands 校验，会把错误反馈给
    LLM 让其重新生成；重试耗尽后丢弃指令并在口语回复中告知用户。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]

    spoken, commands, last_err = "", [], ""

    for attempt in range(_MAX_RETRIES + 1):
        raw      = _call_llm(messages)
        spoken   = _parse_spoken(raw)
        commands = _parse_commands(raw)

        ok, last_err = validate_commands(commands)
        if ok:
            if not spoken:
                print(f"[LLM 原始输出]\n{raw}\n", flush=True)
                print("[警告] LLM 未生成口语回复，检查提示词格式", file=sys.stderr)
            return spoken, commands

        if attempt < _MAX_RETRIES:
            print(f"[校验失败,重试 {attempt+1}/{_MAX_RETRIES}] {last_err}",
                  file=sys.stderr, flush=True)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"你上次输出的执行指令未通过校验：{last_err}。"
                f"请保持完全相同的两段式输出格式，但执行机构、动作名称和参数值"
                f"必须严格使用【可用指令集】中列出的；若没有任何匹配项，"
                f"[执行指令] 输出空数组 []。"
            )})

    print(f"[警告] 重试 {_MAX_RETRIES} 次后仍校验失败,丢弃 commands: {last_err}",
          file=sys.stderr)
    suffix = "另外，我刚才没能正确生成动作指令，可以再说一遍吗？"
    spoken = f"{spoken}\n{suffix}" if spoken else suffix
    return spoken, []


def _call_llm(messages: list) -> str:
    """单次调用 LLM（流式累积），返回完整原始文本。启用千问联网搜索。"""
    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        extra_body={"enable_search": True},   # Qwen / DashScope 内置联网搜索
    )
    raw = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        raw += delta
    return raw


def _parse_spoken(raw: str) -> str:
    """提取 [口语回复]： 与 [执行指令]： 之间的文本（可多行）。"""
    s_idx = raw.find(_SPOKEN_PREFIX)
    if s_idx == -1:
        return ""
    start = s_idx + len(_SPOKEN_PREFIX)
    c_idx = raw.find(_INSTR_PREFIX, start)
    end   = c_idx if c_idx != -1 else len(raw)
    return raw[start:end].strip()


def _parse_commands(raw: str) -> list[dict]:
    """从 LLM 原始输出中提取 [执行指令] 后的 JSON 数组。"""
    idx = raw.find(_INSTR_PREFIX)
    if idx == -1:
        return []
    after = raw[idx + len(_INSTR_PREFIX):].strip()
    first_line = after.split("\n")[0].strip()
    try:
        result = json.loads(first_line)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        start = after.find("[")
        end   = after.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(after[start: end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        print(f"[警告] 执行指令 JSON 解析失败: {after[:200]}", file=sys.stderr)
    return []
