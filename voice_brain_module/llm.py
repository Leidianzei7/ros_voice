#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
from openai import OpenAI
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .commands import COMMANDS, validate_commands

# ── LLM 提示词构建 ────────────────────────────────────────────

def _compact_param(name: str, p: dict) -> str:
    if p.get("options"):
        opts = " | ".join(p["options"])
        return f'{name}=[{opts}]'
    if p.get("range"):
        lo, hi = p["range"]
        unit = p.get("unit", "")
        return f"{name}={lo}~{hi}{unit}"
    return name


def build_command_reference() -> str:
    """生成紧凑的指令参考字符串，嵌入 LLM 系统提示词。"""
    lines = []
    for cmd in COMMANDS:
        params = cmd.get("params", {})
        param_str = ", ".join(_compact_param(k, v) for k, v in params.items()) if params else "无"
        lines.append(
            f"- 执行机构：{cmd['actuator']}；"
            f"执行动作：{cmd['action']}；"
            f"参数：{param_str}  ({cmd['desc']})"
        )
    return "\n".join(lines)


def build_system_prompt() -> str:
    ref = build_command_reference()
    return (
        "你是小智，一个智能陪护机器人助手。用户通过语音和你交互，"
        "一句话里可能同时包含对话/问询（需要你口头回答）和动作请求（需要机器人执行），"
        "你要分别处理这两部分。\n\n"
        "机器人具备以下感知能力，感知数据会以【标记】形式出现在用户消息中：\n"
        "- 视觉识别：可通过摄像头识别桌面物体，区分可抓取物体"
        "（苹果、香蕉、瓶子、蛋糕）和仅可对话物体（如杯子、书本等）；"
        "桌面可能同时有多个不同物体、且每类可能有多个，"
        "上下文中以「物体×数量」标注（如「苹果×2」表示有 2 个苹果）\n"
        "- 情绪识别：可识别用户的情绪状态\n\n"
        "⚠️ 关键规则：如果用户消息中**没有**出现【当前桌面物体】/【用户情绪】等标记，"
        "说明当前**没有任何感知数据**。此时若被问到\"看到了什么\"，必须诚实回答"
        "目前没有看到任何物体，**严禁凭示例或猜测编造物体名称**。\n\n"
        "严格按以下格式回复，两个标签都必须出现：\n"
        "[口语回复]：\n"
        "<对用户的对话或问询给出自然、完整的口语回答；若用户同时请求执行动作，"
        "在回答末尾用一句话确认动作。可多行，不限字数。>\n"
        "[执行指令]：\n"
        "<JSON 数组，每项为三段式：{\"actuator\": ..., \"action\": ..., \"params\": {...}}>\n\n"
        "示例 1（可执行 - 单个动作）：\n"
        "用户：帮我抓一下苹果\n"
        "[口语回复]：\n"
        "好的，我这就用机械臂抓取苹果。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"机械臂\",\"action\":\"抓取\",\"params\":{\"target\":\"苹果\"}}]\n\n"
        "示例 2（多个指令组合）：\n"
        "用户：前进一米然后抓香蕉\n"
        "[口语回复]：\n"
        "好的，我先前进一米再用机械臂抓取香蕉。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"底盘\",\"action\":\"前进\",\"params\":{\"speed\":0.2,\"distance\":1.0}},{\"actuator\":\"机械臂\",\"action\":\"抓取\",\"params\":{\"target\":\"香蕉\"}}]\n\n"
        "示例 3（有视觉数据 - 如实描述所见）：\n"
        "【当前桌面物体】共3个：苹果×2, 杯子\n【可抓取物体】苹果×2\n\n---\n用户说：桌子上有什么\n"
        "[口语回复]：\n"
        "我看到桌上有两个苹果和一个杯子。需要我帮你抓取什么吗？\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 4（无视觉数据 - 诚实说没看到）：\n"
        "用户：桌子上有什么\n"
        "[口语回复]：\n"
        "抱歉，我现在看不到桌面上的东西，视觉系统还没有数据。你可以告诉我你想找什么吗？\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 5（不在指令集，必须拒绝执行）：\n"
        "用户：打开窗帘\n"
        "[口语回复]：\n"
        "抱歉，无法执行这个动作。我目前可以控制底盘前进、后退、左转、右转和停止，"
        "以及用机械臂抓取苹果、香蕉、瓶子、蛋糕、小黄鸭、绿色药盒或大樱桃。\n"
        "[执行指令]：\n"
        "[]\n\n"
        "规则（必须严格遵守）：\n"
        "- 你只能从下方【可用指令集】中选取指令，"
        "**禁止编造任何不在列表里的执行机构、动作或参数值**\n"
        "- 若用户的请求无法用列表中任何一条指令完成，"
        "[执行指令] 输出空数组 []，"
        "[口语回复] 必须明确告知用户「无法执行」，并简短说明你目前能做什么\n"
        "- 若用户没有任何动作意图（纯聊天/问询，如\"桌子上有什么\"、\"你看到了什么\"），"
        "[执行指令] 输出 []，[口语回复] 正常回答即可\n"
        "- 若用户消息中没有【当前桌面物体】标记，说明没有视觉数据，"
        "回答物体相关问题时**必须诚实说没看到**，不要编造任何物体名称\n"
        "- 若用户谈及桌面物体但未指定抓取，应先用口语回复描述所见，"
        "不要自动生成抓取指令，等待用户明确指令\n"
        "- 若用户只有动作没有问询，[口语回复] 仍要简短确认（例如 \"好的，马上去\"）\n"
        "- 参数值必须严格匹配选项之一，不要做近义替换或省略\n"
        "- 不要输出这两个标签之外的任何内容\n"
        "- 当视觉系统反馈用户情绪低落或负面痛苦时，"
        "即使没有动作请求，[口语回复] 也应主动进行安抚和关怀对话，"
        "[执行指令] 输出 []\n\n"
        f"【可用指令集】：\n{ref}"
    )


# ── LLM 客户端 ────────────────────────────────────────────────

llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

_SYSTEM_PROMPT = build_system_prompt()
_SPOKEN_PREFIX = "[口语回复]："
_INSTR_PREFIX  = "[执行指令]："

_MAX_RETRIES = 2   # 校验失败时最多让 LLM 重新生成的次数


def generate_response(
    text: str,
    system_prompt: str = _SYSTEM_PROMPT,
    vision_context: str = "",
) -> tuple[str, list[dict]]:
    """
    调用 LLM，返回 (口语回复, 指令列表)。
    若 LLM 输出的指令未通过 commands.validate_commands 校验，会把错误反馈给
    LLM 让其重新生成；重试耗尽后丢弃指令并在口语回复中告知用户。

    vision_context: 可选的视觉/情绪上下文，会注入到用户消息中，
                    供 LLM 回答"桌上有什么"或进行情绪关怀。
    """
    user_content = text
    if vision_context:
        user_content = f"{vision_context}\n\n---\n用户说：{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
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


def _call_llm_simple(messages: list) -> str:
    """轻量 LLM 调用（非流式，无搜索），供记忆提取等后台任务使用。"""
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


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
