"""人格提示词组装（机制层）。

私有产物：private/persona.md（bootstrap 生成）、private/scenarios/*.md。
机制：人格底稿 + 按场景选风格叠加层 + 结构化上下文 + 输出 schema。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ConfigError, resolve

# 场景文件选择映射：case 的 chat_type → 场景文件（无则不加叠加层）
_SCENARIO_BY_CHAT_TYPE = {
    "group": "group_chat.md",
    "private": "friend.md",
}
CONTEXT_IDENTITY_POLICY = "explicit-self-v1"


class PersonaPromptBuilder:
    def __init__(self, settings: dict[str, Any], prompt_root: Path, *,
                 context_identity: str | None = None):
        """prompt_root：生成器版本目录（自包含 persona.md + scenarios/）。
        版本目录由 versions.py 保证存在——没有静默回退，调用方必须显式给出。"""
        if context_identity not in (None, CONTEXT_IDENTITY_POLICY):
            raise ConfigError(f"未知的 context_identity 策略: {context_identity!r}")
        self._context_identity = context_identity
        self._persona_path = prompt_root / "persona.md"
        self._scenarios_dir = prompt_root / "scenarios"
        if not self._persona_path.exists():
            raise ConfigError(f"版本目录缺少人格文件: {self._persona_path}")
        self._persona = self._persona_path.read_text(encoding="utf-8")

    def _history_line(self, message: dict[str, Any]) -> str:
        sender = str(message.get("sender", "?"))
        if self._context_identity == CONTEXT_IDENTITY_POLICY:
            # Only a typed source flag establishes identity; names and truthy
            # strings/numbers must never turn another participant into self.
            flag = message.get("is_self")
            role = "本人" if flag is True else "其他人" if flag is False else "身份未标明"
            sender = f"[{role}] {sender}"
        return f"{sender}: {message.get('text', '')}"

    def scenario_layer(self, case: dict[str, Any]) -> str:
        name = case.get("scenario") or _SCENARIO_BY_CHAT_TYPE.get(
            case.get("chat_type", ""), ""
        )
        if not name:
            return ""
        path: Path = self._scenarios_dir / name
        if path.exists():
            return f"\n<scenario_rules>\n{path.read_text(encoding='utf-8')}\n</scenario_rules>\n"
        return ""

    def build_messages(
        self,
        case: dict[str, Any],
        style_block: str,
        forced_reply: bool,
    ) -> list[dict[str, str]]:
        history = "\n".join(self._history_line(m) for m in case.get("context", []))
        identity_rule = (
            "<speaker_identity>历史消息的[本人]表示你要代为回复的人，[其他人]表示聊天对象；"
            "[身份未标明]表示缺少身份标记，不要根据姓名猜测。只以本人的身份接话。</speaker_identity>\n"
            if self._context_identity == CONTEXT_IDENTITY_POLICY else ""
        )
        unread = case.get("unread") or ""
        if not unread and case.get("context"):
            unread = str(case["context"][-1].get("text", ""))
        force_rule = (
            "\n<force_reply>评测模式：本题必须给出回复，不得输出空 replies。</force_reply>"
            if forced_reply
            else ""
        )
        user = (
            f"{identity_rule}<history>\n{history}\n</history>\n"
            f"<unread>\n{unread}\n</unread>\n"
            f"{self.scenario_layer(case)}"
            f"<style_examples>\n{style_block}\n</style_examples>\n"
            f"{force_rule}\n"
            "按 system 中的 output_schema 只输出一个合法 JSON 对象。"
        )
        return [
            {"role": "system", "content": self._persona},
            {"role": "user", "content": user},
        ]
