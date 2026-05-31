#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
陪护机器人指令集（三段式）
每条指令格式：
  {
      "actuator": "执行机构（中文，如 机械臂 / 底盘 / 表情屏）",
      "action":   "执行动作（中文，如 抓取 / 前进 / 表达情绪）",
      "desc":     "整体描述",
      "params": {
          "param_name": {
              "options": [...]        # 枚举型参数
              "range":   [min, max],  # 数值型参数
              "default": ...,
              "unit":    "m/s" 等,    # 可选
              "desc":    "参数说明",
          }
      }
  }
LLM 输出格式（JSON 数组）：
  [{"actuator": "机械臂", "action": "抓取", "params": {"target": "小黄鸭"}}, ...]
新增/修改指令只需改此文件，LLM 提示词与话题消息自动跟随。
"""

COMMANDS: list = [
    # ══════════════════════════════════════════
    # 底盘运动（对应 /cmd_vel Twist 消息）
    # ══════════════════════════════════════════
    {
        "actuator": "底盘",
        "action":   "前进",
        "desc":     "向前直行指定距离",
        "params": {
            "speed":    {"type": float, "range": [0.05, 0.5],  "default": 0.2, "unit": "m/s",   "desc": "行进速度"},
            "distance": {"type": float, "range": [0.1, 10.0],  "default": 1.0, "unit": "m",     "desc": "行进距离"},
        },
    },
    {
        "actuator": "底盘",
        "action":   "后退",
        "desc":     "向后直行指定距离",
        "params": {
            "speed":    {"type": float, "range": [0.05, 0.3],  "default": 0.15, "unit": "m/s", "desc": "行进速度"},
            "distance": {"type": float, "range": [0.1, 5.0],   "default": 0.5,  "unit": "m",   "desc": "行进距离"},
        },
    },
    {
        "actuator": "底盘",
        "action":   "左转",
        "desc":     "原地左转指定角度",
        "params": {
            "angle": {"type": float, "range": [5.0, 360.0], "default": 90.0, "unit": "度", "desc": "转动角度"},
            "speed": {"type": float, "range": [0.1, 1.0],   "default": 0.5,  "unit": "rad/s", "desc": "转动角速度"},
        },
    },
    {
        "actuator": "底盘",
        "action":   "右转",
        "desc":     "原地右转指定角度",
        "params": {
            "angle": {"type": float, "range": [5.0, 360.0], "default": 90.0, "unit": "度", "desc": "转动角度"},
            "speed": {"type": float, "range": [0.1, 1.0],   "default": 0.5,  "unit": "rad/s", "desc": "转动角速度"},
        },
    },
    {
        "actuator": "底盘",
        "action":   "停止",
        "desc":     "立即停止所有运动",
        "params": {},
    },
    # ══════════════════════════════════════════
    # 机械臂
    # ══════════════════════════════════════════
    {
        "actuator": "机械臂",
        "action":   "抓取",
        "desc":     "机械臂抓取指定物品",
        "params": {
            "target": {
                "options": ["小黄鸭", "苹果", "绿色药盒"],
                "desc":    "抓取目标，必须严格从三个选项中选一",
            },
        },
    },
]


# ── 指令校验 ──────────────────────────────────────────────────────

_LOOKUP = {(c["actuator"], c["action"]): c["params"] for c in COMMANDS}


def validate_commands(commands: list) -> tuple[bool, str]:
    """
    校验 LLM 产出的指令列表是否符合 COMMANDS schema。
    返回 (是否通过, 错误说明)。空数组视为通过（表示用户无机械动作请求）。
    """
    if not isinstance(commands, list):
        return False, "指令必须是 JSON 数组"

    for i, cmd in enumerate(commands, start=1):
        if not isinstance(cmd, dict):
            return False, f"第 {i} 条指令不是 JSON 对象"

        actuator = cmd.get("actuator")
        action   = cmd.get("action")
        params   = cmd.get("params", {})

        key = (actuator, action)
        if key not in _LOOKUP:
            return False, (
                f'第 {i} 条指令的执行机构/动作 "{actuator}/{action}" 不在指令集'
            )

        spec = _LOOKUP[key]
        if not isinstance(params, dict):
            return False, f"第 {i} 条指令的 params 必须是 JSON 对象"

        for pname, pspec in spec.items():
            if pname not in params:
                return False, f'第 {i} 条指令缺少参数 "{pname}"'
            value = params[pname]

            if "options" in pspec and value not in pspec["options"]:
                return False, (
                    f'第 {i} 条指令参数 "{pname}" 值 "{value}" '
                    f"不在选项 {pspec['options']} 中"
                )
            if "range" in pspec:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return False, f'第 {i} 条指令参数 "{pname}" 不是数值'
                lo, hi = pspec["range"]
                if not (lo <= v <= hi):
                    return False, (
                        f'第 {i} 条指令参数 "{pname}" 值 {v} '
                        f"超出范围 [{lo}, {hi}]"
                    )

        for pname in params:
            if pname not in spec:
                return False, f'第 {i} 条指令含未定义参数 "{pname}"'

    return True, ""


# ── 自动生成 LLM 提示词 ────────────────────────────────────────────

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
        "严格按以下格式回复，两个标签都必须出现：\n"
        "[口语回复]：\n"
        "<对用户的对话或问询给出自然、完整的口语回答；若用户同时请求执行动作，"
        "在回答末尾用一句话确认动作。可多行，不限字数。>\n"
        "[执行指令]：\n"
        "<JSON 数组，每项为三段式：{\"actuator\": ..., \"action\": ..., \"params\": {...}}>\n\n"
        "示例 1（可执行）：\n"
        "用户：帮我抓一下苹果\n"
        "[口语回复]：\n"
        "好的，我这就用机械臂抓取苹果。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"机械臂\",\"action\":\"抓取\",\"params\":{\"target\":\"苹果\"}}]\n\n"
        "示例 2（多个指令组合）：\n"
        "用户：前进一米然后左转\n"
        "[口语回复]：\n"
        "好的，我先前进一米然后左转。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"底盘\",\"action\":\"前进\",\"params\":{\"speed\":0.2,\"distance\":1.0}},{\"actuator\":\"底盘\",\"action\":\"左转\",\"params\":{\"angle\":90.0,\"speed\":0.5}}]\n\n"
        "示例 3（不在指令集，必须拒绝执行）：\n"
        "用户：打开窗帘\n"
        "[口语回复]：\n"
        "抱歉，无法执行这个动作。我目前可以控制底盘前进、后退、左转、右转和停止，"
        "以及用机械臂抓取小黄鸭、苹果或绿色药盒。\n"
        "[执行指令]：\n"
        "[]\n\n"
        "规则（必须严格遵守）：\n"
        "- 你只能从下方【可用指令集】中选取指令，"
        "**禁止编造任何不在列表里的执行机构、动作或参数值**\n"
        "- 若用户的请求无法用列表中任何一条指令完成，"
        "[执行指令] 输出空数组 []，"
        "[口语回复] 必须明确告知用户「无法执行」，并简短说明你目前能做什么\n"
        "- 若用户没有任何动作意图（纯聊天/问询），"
        "[执行指令] 输出 []，[口语回复] 正常回答即可\n"
        "- 若用户只有动作没有问询，[口语回复] 仍要简短确认（例如 \"好的，马上去\"）\n"
        "- 参数值必须严格匹配选项之一，不要做近义替换或省略\n"
        "- 不要输出这两个标签之外的任何内容\n\n"
        f"【可用指令集】：\n{ref}"
    )
