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
    {
        "actuator": "底盘",
        "action":   "人体跟踪",
        "desc":     "启动或停止人体跟随模式",
        "params": {
            "状态": {
                "options": ["开始", "停止"],
                "desc":    "开始或停止人体跟随",
            },
        },
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
                "options": ["苹果", "香蕉", "瓶子", "蛋糕", "小黄鸭", "绿色药盒", "大樱桃"],
                "desc":    "抓取目标，必须严格从七个选项中选一",
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

