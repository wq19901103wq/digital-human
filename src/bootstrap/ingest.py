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
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import ConfigError

REQUIRED_FIELDS = {"chat_id", "chat_type", "sender", "is_self", "timestamp", "text"}
VALID_CHAT_TYPES = {"group", "private"}


def load_unified(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"聊天数据不存在: {path}")
    messages = []
    # JSONL records end at LF. Unicode line separators inside a JSON string
    # are valid text and must not be treated as record boundaries.
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split('\n'), 1):
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


EVENT_POLICY = 'all_events_text_targets_v1'
_KINDS = {'文本消息': 'text', '引用消息': 'quote', '图片消息': 'image',
          '语音消息': 'voice', '视频消息': 'video', '动画表情': 'sticker',
          '系统消息': 'system', '红包消息': 'red_packet', '转账消息': 'transfer',
          '文件消息': 'file', '名片消息': 'contact', '位置消息': 'location',
          '链接消息': 'link', '合并转发消息': 'forward', '通话消息': 'call'}
_TYPES = {'1': 'text', '3': 'image', '34': 'voice', '43': 'video', '47': 'sticker',
          '42': 'contact', '48': 'location', '10000': 'system', '10002': 'system',
          '50': 'call', '244813135921': 'quote'}
_APP_TYPES = {'57': 'quote', '2001': 'red_packet', '2000': 'transfer', '6': 'file',
              '5': 'link', '19': 'forward', '33': 'miniapp', '36': 'miniapp',
              '3': 'music', '4': 'video_share', '51': 'channels', '53': 'channels',
              '24': 'note'}
_LABELS = {'image': '图片', 'voice': '语音', 'video': '视频', 'sticker': '表情包',
           'red_packet': '红包', 'transfer': '转账', 'file': '文件', 'contact': '名片',
           'location': '位置', 'link': '链接', 'forward': '合并转发', 'system': '系统事件',
           'quote': '引用消息', 'text': '空文本', 'unknown': '未解析消息',
           'call': '通话', 'miniapp': '小程序', 'music': '音乐分享',
           'video_share': '视频分享', 'channels': '视频号', 'note': '笔记'}


def message_kind(message):
    """Accept exporter names and app subtypes; unknown types remain real events."""
    app = str(message.get('appMsgType') or '')
    # Exporter app metadata can be more accurate than the outer localType (e.g.
    # quoted messages exported as sticker/other). Check quote evidence first.
    if (app == '57' or message.get('appMsgKind') == 'quote'
            or message.get('type') == '引用消息' or str(message.get('localType')) == '244813135921'):
        return 'quote'
    if app in _APP_TYPES:
        return _APP_TYPES[app]
    # WeChat packs app subtype in the high 32 bits and base type 49 in the low
    # bits. Older exports omit appMsgType, including all red packets/transfers.
    local = str(message.get('localType', ''))
    if local.isdecimal() and int(local) & 0xffffffff == 49:
        subtype = str(int(local) >> 32)
        if subtype in _APP_TYPES:
            return _APP_TYPES[subtype]
    name = str(message.get('type') or '')
    for needle, kind in [('红包', 'red_packet'), ('转账', 'transfer'), ('撤回', 'system')]:
        if needle in name:
            return kind
    return _KINDS.get(name, _TYPES.get(str(message.get('localType')), 'unknown'))


def normalize_event(message, occurrence=0):
    kind = message_kind(message)
    content = message.get('content')
    if content is not None and not isinstance(content, str):
        raise ConfigError('导出消息 content 不是字符串或 null')
    content = content or ''
    text = content.strip()
    # Never infer media contents. Preserve the raw export separately and expose
    # its type in context, including events whose content is null.
    eligible = kind in {'text', 'quote'} and bool(text)
    if not eligible:
        text = f'[{_LABELS[kind]}：导出未提供可读内容]'
        if content.strip() and not content.lstrip().startswith('<'):
            text = f'[{_LABELS[kind]}] {content.strip()}'
    platform_id = str(message.get('platformMessageId') or '')
    if platform_id and platform_id != '0':
        identity = 'platform:' + platform_id
    else:
        # No reliable server ID: retain repeated identical events within a file.
        # Matching occurrences across overlapping exports may still deduplicate.
        payload = {k: message.get(k) for k in ('createTime', 'senderUsername', 'isSend',
                   'localType', 'type', 'content', 'replyToMessageId', 'quotedContent')}
        identity = 'fallback:' + hashlib.sha256(json.dumps(payload, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest() + ':' + str(occurrence)
    event = dict(schema=1, id=identity, kind=kind, source_type=str(message.get('localType', '')),
        source_type_name=str(message.get('type') or ''), sender_id=str(message.get('senderUsername') or ''),
        reply_eligible=eligible, context_boundary=kind in {'system', 'unknown'})
    # Preserve exported attachment/reference metadata, even when no media bytes
    # or readable contents were exported. It is not rendered as model input.
    metadata = {k: v for k, v in message.items() if k not in {
        'content', 'createTime', 'formattedTime', 'localType', 'type', 'isSend',
        'senderUsername', 'senderDisplayName', 'senderAvatarKey', 'platformMessageId',
        'localId', 'source', 'replyToMessageId', 'quotedContent', 'quotedSender', 'quotedType'}}
    if metadata:
        event['metadata'] = metadata
    if content != text:
        event['original_content'] = content
    if kind == 'quote':
        event['quote'] = {k: message[k] for k in
            ('replyToMessageId', 'quotedContent', 'quotedSender', 'quotedType') if k in message}
    return text, event


def load_weflow_dir(root: Path, *, statistics=None) -> list[dict[str, Any]]:
    return load_weflow_files(sorted(root.rglob('*.json')), statistics=statistics)


def load_weflow_files(paths, *, statistics=None) -> list[dict[str, Any]]:
    """Keep every source event; only deduplicate matching source identities.

    Parsing errors and conflicting server IDs stop ingestion, never skip a file.
    The same path list can be taken from a frozen manifest for reproducible repair.
    """
    by_chat: dict[str, list[dict[str, Any]]] = {}
    counts, duplicates, source_types = Counter(), Counter(), Counter()
    seen = {}
    for f in sorted(map(Path, paths)):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f'无法解析导出 {f}') from exc
        if not isinstance(data, dict) or not isinstance(data.get('messages'), list):
            raise ConfigError(f'导出缺少消息列表: {f}')
        if any(not isinstance(m, dict) for m in data['messages']):
            raise ConfigError(f'导出消息不是对象: {f}')
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
        occurrences = Counter()
        for m in data.get("messages") or []:
            text, event = normalize_event(m)
            base_id = event['id']
            if base_id.startswith('fallback:'):
                event['id'] = base_id.rsplit(':', 1)[0] + ':' + str(occurrences[base_id])
                occurrences[base_id] += 1
            counts[event['kind']] += 1
            source_types[(event['kind'], event['source_type'], event['source_type_name'],
                          str(m.get('appMsgType') or ''), str(m.get('appMsgKind') or ''))] += 1
            timestamp = m.get('createTime')
            if type(timestamp) is not int or timestamp <= 0:
                raise ConfigError(f'导出消息时间戳无效: {f}')
            row = {
                "chat_id": chat_id,
                "chat_type": chat_type,
                "chat_name": chat_name,
                "source_chat_id": source_chat_id,
                "sender": str(m.get("senderDisplayName") or m.get("senderUsername") or "?"),
                "is_self": bool(m.get("isSend")),
                "timestamp": timestamp,
                "text": text,
                "event": event,
            }
            key = (chat_id, event['id'])
            if key in seen:
                if seen[key] != row:
                    raise ConfigError(f'同一来源消息 ID 内容冲突: {f}')
                duplicates[event['kind']] += 1
            else:
                seen[key] = row
                rows.append(row)
    messages = []
    for rows in by_chat.values():
        rows.sort(key=lambda m: m["timestamp"])
        messages.extend(rows)
    if statistics is not None:
        statistics.update(policy=EVENT_POLICY, raw_by_type=dict(counts),
                          duplicates_by_type=dict(duplicates), retained_by_type=dict(counts - duplicates),
                          raw_source_types=[dict(kind=k[0], local_type=k[1], name=k[2],
                              app_type=k[3], app_kind=k[4], count=n)
                              for k, n in sorted(source_types.items())])
    return messages


def message_identity(message):
    keys = ['chat_id', 'sender', 'timestamp', 'is_self', 'text']
    if 'event' in message:
        keys.append('event')
    return hashlib.sha256(json.dumps({k: message[k] for k in keys},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
