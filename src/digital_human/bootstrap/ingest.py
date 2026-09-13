"""聊天记录导入（机制层）：统一消息 schema 的校验与加载。

机制层只认统一 schema；各 IM 导出格式的适配器写在这里（数据层无关）。
统一消息：
  {"chat_id": str, "chat_type": "group"|"private", "chat_name": str,
   "sender": str, "is_self": bool, "timestamp": int, "text": str}
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"chat_id", "chat_type", "sender", "is_self", "timestamp", "text"}
VALID_CHAT_TYPES = {"group", "private"}


def load_unified(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"聊天数据不存在: {path}")
    messages = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}:{lineno} 缺少字段 {missing}")
        if row["chat_type"] not in VALID_CHAT_TYPES:
            raise ValueError(f"{path}:{lineno} chat_type 非法: {row['chat_type']}")
        messages.append(row)
    return messages


def load_chat_export(path: Path) -> list[dict[str, Any]]:
    """按文件类型分发到适配器。当前支持统一 jsonl 与 WeFlow 导出目录。"""
    if path.suffix == ".jsonl":
        return load_unified(path)
    if path.is_dir():
        return load_weflow_dir(path)
    raise ValueError(
        f"暂不支持的导出格式: {path}；可用统一 jsonl（schema 见 docstring）或 WeFlow 导出目录"
    )


def load_weflow_dir(root: Path) -> list[dict[str, Any]]:
    """WeFlow 导出目录 → 统一消息。

    chat_id 用 session.wxid（非文件名，避免 b/main 同名文件被拼错）；同一 chat_id 的
    多个导出文件先合并，按时间排序，再按 (timestamp, sender, content) 去重——
    实际数据存在同名文件时间倒退与数百条重复记录，不清理会弄乱测试题上下文。
    只保留文本消息（localType 1）；isSend=1 视为本人。
    """
    by_chat: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(root.rglob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"跳过无法解析的导出 {f}: {exc}")
            continue
        session = data.get("session") or {}
        chat_type = "group" if session.get("type") == "群聊" else "private"
        # session.wxid 是联系人/群 ID，不是账号；账号 = 本方发送者 ID（isSend=1 的 senderUsername）
        senders = [m.get("senderUsername") for m in data.get("messages") or [] if m.get("isSend")]
        account = str(senders[0]) if senders else "unknown"
        peer = str(session.get("wxid") or f.stem)
        chat_id = f"{chat_type}:{account}:{peer}"
        chat_name = str(session.get("displayName") or f.stem)
        source_name = re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", f.stem).strip()
        source_chat_id = "chat_" + hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:10]
        rows = by_chat.setdefault(chat_id, [])
        for m in data.get("messages") or []:
            if m.get("localType") != 1:
                continue
            text = str(m.get("content") or "").strip()
            if not text:
                continue
            rows.append({
                "chat_id": chat_id,
                "chat_type": chat_type,
                "chat_name": chat_name,
                "source_chat_id": source_chat_id,
                "sender": str(m.get("senderDisplayName") or m.get("senderUsername") or "?"),
                "is_self": bool(m.get("isSend")),
                "timestamp": int(m.get("createTime") or 0),
                "text": text,
            })
    messages = []
    dup = 0
    for rows in by_chat.values():
        rows.sort(key=lambda m: m["timestamp"])
        seen = set()
        for m in rows:
            key = (m["timestamp"], m["sender"], m["text"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            messages.append(m)
    if dup:
        print(f"跨导出合并去重：剔除重复消息 {dup} 条")
    return messages
