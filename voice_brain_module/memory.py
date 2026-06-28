#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持久化记忆管理：用户画像 / 对话历史 / 重要记忆。

数据存储在 ~/.ros_voice/ 目录下，供 brain_node 在每次 LLM 推理时注入上下文，
实现跨对话的连贯性和个性化。

用法:
    mem = UserMemory()
    context = mem.get_context_for_llm()              # 注入 LLM 的记忆文本
    mem.add_turn(user_text, assistant_text)           # 保存一轮对话
    mem.extract_and_save(user_text, assistant_text)   # 调 LLM 提取重要记忆
"""
import json
import os
import threading
from pathlib import Path
from datetime import datetime


class UserMemory:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.ros_voice")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._profile_path  = self._dir / "user_profile.json"
        self._history_path  = self._dir / "conversation_history.json"
        self._memory_path   = self._dir / "important_memories.json"
        self._lock = threading.Lock()

        self._ensure_files()

    def _ensure_files(self):
        for p in [self._profile_path, self._history_path, self._memory_path]:
            if not p.exists():
                p.write_text("[]" if p != self._profile_path else "{}", encoding="utf-8")

    # ── 读取 ──────────────────────────────────────────────

    def _read_json(self, path: Path, default):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return default

    def _write_json(self, path: Path, data):
        with self._lock:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(path)

    def get_profile(self) -> dict:
        return self._read_json(self._profile_path, {})

    def get_history(self, n: int = 10) -> list:
        return self._read_json(self._history_path, [])[-n:]

    def get_memories(self) -> list:
        return self._read_json(self._memory_path, [])

    # ── 上下文拼装 ────────────────────────────────────────

    def get_context_for_llm(self) -> str:
        """生成注入 LLM 的完整记忆上下文。"""
        parts = []

        profile = self.get_profile()
        if profile:
            name = profile.get("name", "")
            prefs = profile.get("preferences", [])
            desc  = profile.get("description", "")
            if name:
                parts.append(f"【用户名称】{name}")
            if desc:
                parts.append(f"【用户描述】{desc}")
            if prefs:
                parts.append(f"【用户偏好】{', '.join(prefs)}")

        memories = self.get_memories()
        if memories:
            mem_lines = [m.get("content", "") for m in memories]
            parts.append(f"【重要记忆】\n" + "\n".join(f"- {x}" for x in mem_lines if x))

        history = self.get_history(10)
        if history:
            hist_lines = []
            for h in history:
                u = h.get("user", "")
                a = h.get("assistant", "")
                if u or a:
                    hist_lines.append(f"用户：{u}\n小智：{a}")
            if hist_lines:
                parts.append(f"【最近对话】\n" + "\n---\n".join(hist_lines))

        return "\n\n".join(parts) if parts else ""

    # ── 写入 ──────────────────────────────────────────────

    def add_turn(self, user_text: str, assistant_text: str):
        history = self._read_json(self._history_path, [])
        history.append({
            "user": user_text,
            "assistant": assistant_text,
            "time": datetime.now().isoformat(),
        })
        if len(history) > 50:
            history = history[-50:]
        self._write_json(self._history_path, history)

    def extract_and_save(self, user_text: str, assistant_text: str):
        """用 LLM 从对话中提取画像更新和重要记忆（在独立线程中运行）。"""
        def _work():
            try:
                from .llm import _call_llm_simple
                self._update_profile(user_text, assistant_text, _call_llm_simple)
                self._extract_memories(user_text, assistant_text, _call_llm_simple)
            except Exception:
                pass  # 记忆提取失败不影响主流程

        threading.Thread(target=_work, daemon=True).start()

    def _update_profile(self, user_text: str, assistant_text: str, call_llm):
        profile = self.get_profile()
        prompt = (
            f"当前用户画像：{json.dumps(profile, ensure_ascii=False)}\n\n"
            f"用户说：{user_text}\n"
            f"助手回复：{assistant_text}\n\n"
            f"从以上对话中提取用户的新信息（姓名、偏好、习惯、特征等）。"
            f"只需输出一个 JSON 对象，合并到现有画像中。"
            f"不要编造内容。如果没有新信息，输出空对象 {{}}。"
        )
        try:
            raw = call_llm([
                {"role": "system", "content": "你是一个用户画像提取器。只输出 JSON，不要任何解释。"},
                {"role": "user", "content": prompt},
            ])
            # 提取 JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                update = json.loads(raw[start:end + 1])
                if update:
                    profile.update(update)
                    self._write_json(self._profile_path, profile)
        except Exception:
            pass

    def _extract_memories(self, user_text: str, assistant_text: str, call_llm):
        existing = self.get_memories()
        prompt = (
            f"已有重要记忆：{json.dumps(existing, ensure_ascii=False)}\n\n"
            f"用户说：{user_text}\n"
            f"助手回复：{assistant_text}\n\n"
            f"判断对话中是否包含需要长期记住的信息"
            f"（例如：计划、偏好变化、重要日期、人际关系等）。"
            f"只需输出一个 JSON 数组，每项含 content 字段。"
            f"不要记住闲聊内容。如果没有，输出空数组 []。"
        )
        try:
            raw = call_llm([
                {"role": "system", "content": "你是一个记忆提取器。只输出 JSON 数组，不要任何解释。"},
                {"role": "user", "content": prompt},
            ])
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                new_mems = json.loads(raw[start:end + 1])
                if new_mems:
                    for m in new_mems:
                        m["source_date"] = datetime.now().isoformat()
                    existing.extend(new_mems)
                    if len(existing) > 30:
                        existing = existing[-30:]
                    self._write_json(self._memory_path, existing)
        except Exception:
            pass
