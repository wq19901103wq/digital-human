"""RPA 已采用 Judge 的纯推理函数快照。

从 rpa_runtime_sources.json 所列源函数逐字提取；不含训练、数据集或 RPA 运行依赖。
更换算法须新建 runtime 版本，避免改变已冻结 Judge 的行为。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

RECENT_SPEAKER_POSITIONS = 5


IDENTIFIED_GROUP_SPEAKER_POSITIONS = 3


GROUP_PATTERN_RECENT_MESSAGE_LIMIT = 20


OBSERVABLE_OPTION_BOOLEAN_NAMES = (
    "reply_bubble_count>=2",
    "reply_bubble_count>=3",
    "reply_char_count>=8",
    "reply_char_count>=16",
    "reply_char_count>=32",
    "reply_contains_question_mark",
    "reply_contains_at_mention",
    "reply_contains_digit",
    "reply_mentions_latest_speaker",
    "reply_contains_laughter_marker",
)


GROUP_PATTERN_OPTION_BOOLEAN_NAMES = (
    "reply_reuses_active_group_pattern",
)


INTEGER_FIELDS = (
    "context_fit",
    "addressee_fit",
    "timeline_continuity",
    "stance_continuity",
    "specificity",
    "conversational_naturalness",
    "ai_template_signal",
    "unsupported_fact_signal",
    "over_explanation_signal",
    "group_pattern_alignment",
)


BOOLEAN_FIELDS = (
    "depends_on_unstated_fact",
    "repeats_context_needlessly",
)


ENUM_FIELDS = {
    "reply_action": (
        "answer_now", "defer", "accept", "refuse", "confirm", "joke_tease",
        "agree", "disagree", "ask", "clarify", "supplement", "acknowledge_close",
        "empathize_support", "evaluate_criticize", "correct", "coordinate", "other",
    ),
    "tone": (
        "plain", "warm", "playful", "teasing", "supportive", "serious",
        "impatient", "formal", "other",
    ),
    "length_band": ("very_short", "short", "medium", "long"),
}


OPTION_FIELDS = set(INTEGER_FIELDS) | set(BOOLEAN_FIELDS) | set(ENUM_FIELDS)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def feature_extractor_prompt(case: dict[str, Any]) -> str:
    allowed = {
        "integer_fields": {field: "integer 0..3" for field in INTEGER_FIELDS},
        "boolean_fields": {field: "boolean" for field in BOOLEAN_FIELDS},
        "enum_fields": ENUM_FIELDS,
    }
    safe_case = {
        "relationship": case["relationship"],
        "context": case["context_original"],
        "option_A": case["option_A"],
        "option_B": case["option_B"],
    }
    forbidden_fields = {"human_option", "source_case_id", "answer", "label", "case_id"}
    if set(safe_case) & forbidden_fields:
        raise ValueError("feature request contains a forbidden answer-bearing field")
    prompt = (
        "你是回复特征提取器，不是分类器。你每次只看到一道匿名题。"
        "分别、独立、使用完全相同的字段提取 A 和 B 的可观察特征。"
        "不得判断或暗示哪项是真人，不得推荐 A/B，不得输出自由文本理由。"
        "上下文没有提供的事实必须视为未知。group_pattern_alignment 仅在群聊里衡量复读、队形、接力调侃等承接，"
        "私聊固定填 0。所有评分都只用 0、1、2、3。只输出一个 JSON 对象，顶层只能是 option_A、option_B。\n"
        "<schema>\n" + json.dumps(allowed, ensure_ascii=False, sort_keys=True) + "\n</schema>\n"
        "<case>\n" + json.dumps(safe_case, ensure_ascii=False) + "\n</case>"
    )
    return prompt


def validate_option_features(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != OPTION_FIELDS:
        raise ValueError("feature option does not match the fixed schema")
    result: dict[str, Any] = {}
    for field in INTEGER_FIELDS:
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or not 0 <= field_value <= 3:
            raise ValueError(f"{field} must be an integer from 0 through 3")
        result[field] = field_value
    for field in BOOLEAN_FIELDS:
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
        result[field] = value[field]
    for field, allowed in ENUM_FIELDS.items():
        if value[field] not in allowed:
            raise ValueError(f"{field} has an unsupported enum value")
        result[field] = value[field]
    return result


def parse_feature_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("feature extractor did not return JSON")
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict) or set(payload) != {"option_A", "option_B"}:
        raise ValueError("feature response top level must contain only option_A and option_B")
    return {
        "option_A": validate_option_features(payload["option_A"]),
        "option_B": validate_option_features(payload["option_B"]),
    }


def feature_response_json_schema() -> dict[str, Any]:
    option_properties: dict[str, Any] = {
        field: {"type": "integer", "minimum": 0, "maximum": 3}
        for field in INTEGER_FIELDS
    }
    option_properties.update({field: {"type": "boolean"} for field in BOOLEAN_FIELDS})
    option_properties.update({
        field: {"type": "string", "enum": list(allowed)}
        for field, allowed in ENUM_FIELDS.items()
    })
    option_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(OPTION_FIELDS),
        "properties": option_properties,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["option_A", "option_B"],
        "properties": {"option_A": option_schema, "option_B": option_schema},
    }


def vectorize_boolean_option(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    option = validate_option_features(features)
    values: list[float] = []
    names: list[str] = []
    for field in INTEGER_FIELDS:
        for threshold in (1, 2, 3):
            values.append(float(option[field] >= threshold))
            names.append(f"{field}>={threshold}")
    for field in BOOLEAN_FIELDS:
        values.append(float(option[field]))
        names.append(field)
    for field, allowed in ENUM_FIELDS.items():
        for choice in allowed:
            values.append(float(option[field] == choice))
            names.append(f"{field}={choice}")
    vector = np.asarray(values, dtype=float)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("boolean option encoding produced a non-boolean value")
    return vector, names


def vectorize_boolean_options(
    features: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    left, left_names = vectorize_boolean_option(features["option_A"])
    right, right_names = vectorize_boolean_option(features["option_B"])
    if left_names != right_names:
        raise ValueError("A/B boolean option feature schemas differ")
    return left, right, left_names


_SPEAKER_PREFIX = re.compile(
    r"^(.*?)[（(]\d{4}-\d{2}-\d{2}[^）)]*[）)][:：]",
    flags=re.DOTALL,
)


def speaker_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    speakers: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        for message in root.iter("message"):
            role = str(message.attrib.get("role") or "")
            if role == "self":
                speakers.append("__self__")
                continue
            text_value = "".join(message.itertext()).strip()
            matched = _SPEAKER_PREFIX.match(text_value)
            speaker = matched.group(1).strip() if matched else "__unknown_other__"
            speakers.append(speaker or "__unknown_other__")
    return tuple(speakers)


def message_text_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    messages: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        messages.extend("".join(message.itertext()).strip() for message in root.iter("message"))
    return tuple(messages)


def group_other_text_timeline(context_original: Any) -> tuple[str, ...]:
    if not isinstance(context_original, list):
        raise ValueError("context_original must be a list of XML fragments")
    messages: list[str] = []
    for fragment in context_original:
        if not isinstance(fragment, str):
            raise ValueError("context XML fragment must be text")
        root = ET.fromstring(fragment)
        messages.extend(
            "".join(message.itertext()).strip()
            for message in root.iter("message")
            if message.attrib.get("role") == "other"
            and message.attrib.get("type") == "text"
        )
    return tuple(messages)


def _private_category_name(prefix: str, value: str) -> str:
    return f"{prefix}={_hash(value)[:12]}"


def vectorize_context_booleans(
    metadata: dict[str, Any], schema: dict[str, tuple[Any, ...]]
) -> tuple[np.ndarray, list[str]]:
    relationship = str(metadata["relationship"])
    chat_id = str(metadata["source_chat_id"])
    group_name = str(metadata["group_name"])
    recent = tuple(str(value) for value in metadata["recent_speakers"])
    known_group_names = tuple(str(value) for value in schema["known_group_names"])
    known_group_members = tuple(
        (str(chat), str(speaker)) for chat, speaker in schema["known_group_members"]
    )
    values: list[float] = [
        float(relationship == "private"),
        float(relationship == "group"),
        float(bool(metadata.get("latest_message_has_question_mark"))),
        float(bool(metadata.get("latest_message_has_at_mention"))),
    ]
    names: list[str] = [
        "is_private",
        "is_group",
        "latest_message_has_question_mark",
        "latest_message_has_at_mention",
    ]
    for known_group_name in known_group_names:
        values.append(float(
            relationship == "group" and group_name == known_group_name
        ))
        names.append(_private_category_name("group_name", known_group_name))
    values.append(float(
        relationship == "group" and group_name not in known_group_names
    ))
    names.append("group_name=unknown")
    for position in range(RECENT_SPEAKER_POSITIONS):
        exists = position < len(recent)
        speaker = recent[position] if exists else ""
        values.extend((
            float(exists),
            float(exists and speaker == "__self__"),
            float(exists and speaker != "__self__"),
        ))
        names.extend((
            f"recent_{position + 1}_exists",
            f"recent_{position + 1}_is_self",
            f"recent_{position + 1}_is_other",
        ))
    for left_position in range(RECENT_SPEAKER_POSITIONS):
        for right_position in range(left_position + 1, RECENT_SPEAKER_POSITIONS):
            values.append(float(
                right_position < len(recent)
                and recent[left_position] == recent[right_position]
            ))
            names.append(
                f"recent_{left_position + 1}_same_speaker_as_{right_position + 1}"
            )
    values.append(float(len(recent) >= 2 and recent[0] != recent[1]))
    names.append("latest_turn_switches_speaker")
    known_member_set = set(known_group_members)
    for position in range(IDENTIFIED_GROUP_SPEAKER_POSITIONS):
        speaker = recent[position] if position < len(recent) else ""
        current_member = (chat_id, speaker)
        for known_chat_id, known_speaker in known_group_members:
            values.append(float(
                relationship == "group"
                and current_member == (known_chat_id, known_speaker)
            ))
            names.append(_private_category_name(
                f"recent_{position + 1}_group_member",
                f"{known_chat_id}\0{known_speaker}",
            ))
        values.append(float(
            relationship == "group"
            and bool(speaker)
            and speaker != "__self__"
            and current_member not in known_member_set
        ))
        names.append(f"recent_{position + 1}_group_member=unknown")
    vector = np.asarray(values, dtype=float)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise ValueError("context encoding produced a non-boolean value")
    return vector, names


_GROUP_PATTERN_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")


_GROUP_PATTERN_QUOTE = re.compile(r"\[引用.*$", flags=re.DOTALL)


_GROUP_PATTERN_PUNCTUATION = re.compile(
    r"[\s，,、…~～—\-：:。！？!?；;\"'“”‘’（）()\[\]【】]+"
)


def _group_pattern_sentences(value: str) -> tuple[str, ...]:
    matched = _SPEAKER_PREFIX.match(value)
    body = value[matched.end():] if matched else value
    body = _GROUP_PATTERN_QUOTE.sub("", body).strip()
    while body.startswith("@") and "\u2005" in body:
        body = body.split("\u2005", 1)[1].lstrip()
    body = re.sub(r"^同(?=\s)", "", body).strip()
    sentences = (
        _GROUP_PATTERN_PUNCTUATION.sub("", sentence)
        for sentence in _GROUP_PATTERN_SENTENCE_SPLIT.split(body)
    )
    return tuple(sentence for sentence in sentences if len(sentence) >= 2)


def _has_distinctive_group_pattern_match(left: str, right: str) -> bool:
    if min(len(left), len(right)) < 8:
        return False
    blocks = [
        block.size
        for block in difflib.SequenceMatcher(
            None, left, right, autojunk=False
        ).get_matching_blocks()
        if block.size >= 2
    ]
    return (
        len(blocks) >= 2
        and max(blocks) >= 3
        and sum(blocks) >= 6
        and sum(blocks) / min(len(left), len(right)) >= 0.65
    )


def reply_reuses_active_group_pattern(
    reply: Any, metadata: dict[str, Any]
) -> bool:
    if metadata.get("relationship") != "group":
        return False
    if not isinstance(reply, list) or not all(isinstance(item, str) for item in reply):
        raise ValueError("candidate reply must be a list of strings")
    candidate_sentences = tuple(
        sentence
        for bubble in reply
        for sentence in _group_pattern_sentences(bubble)
    )
    history_by_message = tuple(
        _group_pattern_sentences(str(message))
        for message in metadata.get("recent_group_other_texts") or ()
    )
    exact_counts: Counter[str] = Counter()
    for sentences in history_by_message:
        exact_counts.update(set(sentences))
    if any(exact_counts[sentence] >= 2 for sentence in candidate_sentences):
        return True
    history_sentences = tuple(
        sentence for sentences in history_by_message for sentence in sentences
    )
    return any(
        _has_distinctive_group_pattern_match(candidate, historical)
        for candidate in candidate_sentences
        for historical in history_sentences
    )


def vectorize_group_pattern_option_boolean(
    reply: Any, metadata: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(
        [reply_reuses_active_group_pattern(reply, metadata)], dtype=float
    )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("group pattern encoding produced a non-boolean value")
    return values, list(GROUP_PATTERN_OPTION_BOOLEAN_NAMES)


def vectorize_observable_option_booleans(
    reply: Any, metadata: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(reply, list) or not all(isinstance(item, str) for item in reply):
        raise ValueError("candidate reply must be a list of strings")
    text = "".join(reply)
    compact_text = "".join(character for character in text if not character.isspace())
    recent = tuple(str(value) for value in metadata.get("recent_speakers") or ())
    latest_speaker = recent[0] if recent else ""
    mentions_latest_speaker = (
        latest_speaker not in {"", "__self__", "__unknown_other__"}
        and latest_speaker in text
    )
    contains_laughter_marker = any(
        marker in text.lower() for marker in ("哈", "笑", "hh", "lol")
    )
    values = np.asarray(
        [
            len(reply) >= 2,
            len(reply) >= 3,
            len(compact_text) >= 8,
            len(compact_text) >= 16,
            len(compact_text) >= 32,
            any(marker in text for marker in ("?", "？")),
            "@" in text,
            any(character.isdigit() for character in text),
            mentions_latest_speaker,
            contains_laughter_marker,
        ],
        dtype=float,
    )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("observable reply encoding produced a non-boolean value")
    return values, list(OBSERVABLE_OPTION_BOOLEAN_NAMES)


def refined_boolean_context_crosses(
    option_names: list[str], context_names: list[str]
) -> list[tuple[str, str]]:
    option_set = set(option_names)
    context_set = set(context_names)
    crosses: list[tuple[str, str]] = []

    def add(context_name: str, option_name: str) -> None:
        pair = (context_name, option_name)
        if (
            context_name in context_set
            and option_name in option_set
            and pair not in crosses
        ):
            crosses.append(pair)

    relation_features = (
        "reply_action=answer_now",
        "reply_action=defer",
        "reply_action=accept",
        "reply_action=refuse",
        "reply_action=confirm",
        "reply_action=joke_tease",
        "reply_action=ask",
        "reply_action=clarify",
        "reply_action=supplement",
        "reply_action=acknowledge_close",
        "reply_action=coordinate",
        "tone=plain",
        "tone=warm",
        "tone=playful",
        "tone=teasing",
        "tone=serious",
        "tone=formal",
        "conversational_naturalness>=2",
        "conversational_naturalness>=3",
        "ai_template_signal>=2",
        "ai_template_signal>=3",
        "unsupported_fact_signal>=1",
        "unsupported_fact_signal>=2",
        "over_explanation_signal>=2",
        "over_explanation_signal>=3",
        "depends_on_unstated_fact",
        "repeats_context_needlessly",
        "reply_bubble_count>=2",
        "reply_bubble_count>=3",
        "reply_char_count>=8",
        "reply_char_count>=16",
        "reply_char_count>=32",
        "reply_contains_question_mark",
        "reply_contains_at_mention",
        "reply_contains_digit",
        "reply_contains_laughter_marker",
    )
    for context_name in ("is_private", "is_group"):
        for option_name in relation_features:
            add(context_name, option_name)
    for option_name in (
        "group_pattern_alignment>=1",
        "group_pattern_alignment>=2",
        "group_pattern_alignment>=3",
    ):
        add("is_group", option_name)

    for context_name in (
        "recent_1_is_self",
        "recent_1_is_other",
        "latest_turn_switches_speaker",
    ):
        for option_name in (
            "addressee_fit>=2",
            "addressee_fit>=3",
            "timeline_continuity>=2",
            "timeline_continuity>=3",
            "stance_continuity>=2",
            "stance_continuity>=3",
            "reply_mentions_latest_speaker",
        ):
            add(context_name, option_name)
    for option_name in (
        "reply_action=answer_now",
        "reply_action=defer",
        "reply_action=confirm",
        "reply_action=ask",
        "reply_action=clarify",
        "reply_contains_question_mark",
    ):
        add("latest_message_has_question_mark", option_name)
    for option_name in (
        "addressee_fit>=2",
        "addressee_fit>=3",
        "reply_mentions_latest_speaker",
        "reply_contains_at_mention",
    ):
        add("latest_message_has_at_mention", option_name)
    for context_name in (
        "recent_1_same_speaker_as_2",
        "recent_1_same_speaker_as_3",
        "recent_2_same_speaker_as_3",
    ):
        for option_name in (
            "group_pattern_alignment>=2",
            "group_pattern_alignment>=3",
            "reply_action=joke_tease",
        ):
            add(context_name, option_name)

    for context_name in context_names:
        if context_name.startswith("group_name="):
            for option_name in (
                "group_pattern_alignment>=2",
                "tone=playful",
                "reply_bubble_count>=2",
            ):
                add(context_name, option_name)
        elif context_name.startswith("recent_1_group_member="):
            for option_name in (
                "addressee_fit>=2",
                "reply_mentions_latest_speaker",
            ):
                add(context_name, option_name)
        elif context_name.startswith("recent_2_group_member="):
            add(context_name, "timeline_continuity>=2")
        elif context_name.startswith("recent_3_group_member="):
            add(context_name, "stance_continuity>=2")
    return crosses


def expand_boolean_option_with_refined_context(
    option: np.ndarray,
    option_names: list[str],
    context: np.ndarray,
    context_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if len(option) != len(option_names) or len(context) != len(context_names):
        raise ValueError("boolean values and feature names are not aligned")
    if not np.isin(option, (0.0, 1.0)).all() or not np.isin(context, (0.0, 1.0)).all():
        raise ValueError("context crosses require boolean inputs")
    option_indexes = {name: index for index, name in enumerate(option_names)}
    context_indexes = {name: index for index, name in enumerate(context_names)}
    crosses = refined_boolean_context_crosses(option_names, context_names)
    crossed_values = np.asarray(
        [
            context[context_indexes[context_name]] * option[option_indexes[option_name]]
            for context_name, option_name in crosses
        ],
        dtype=float,
    )
    crossed_names = [
        f"{context_name}*{option_name}" for context_name, option_name in crosses
    ]
    return np.concatenate((option, crossed_values)), [*option_names, *crossed_names]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


@dataclass(frozen=True)
class FormalJudge:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    context_schema: dict[str, tuple[Any, ...]]
    schema_hash: str
    artifact_sha256: str


def load_formal_judge(path: Path) -> FormalJudge:
    """Load the frozen pairwise Judge and fail closed on any schema drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("final_model") or {}
    feature_system = payload.get("feature_system") or {}
    names = tuple(str(value) for value in model.get("feature_names") or ())
    coefficients = np.asarray(model.get("coefficients") or (), dtype=float)
    raw_schema = feature_system.get("context_schema")
    if (
        payload.get("selected_model_name") != "symmetric_logistic_refined_boolean_crosses"
        or model.get("model") != "LogisticRegression"
        or model.get("fit_intercept") is not False
        or not names
        or len(names) != len(coefficients)
        or int(model.get("input_feature_count") or -1) != len(names)
        or not isinstance(raw_schema, dict)
    ):
        raise ValueError("formal Judge artifact is not the frozen pairwise boolean model")
    context_schema = {
        str(key): tuple(value) for key, value in raw_schema.items()
        if isinstance(value, list)
    }
    if set(context_schema) != {"known_group_names", "known_group_members"}:
        raise ValueError("formal Judge context schema is incomplete")
    schema_payload = {
        "feature_names": names,
        "context_schema": context_schema,
        "model": model.get("model"),
        "fit_intercept": model.get("fit_intercept"),
    }
    return FormalJudge(
        feature_names=names,
        coefficients=coefficients,
        context_schema=context_schema,
        schema_hash=_json_sha256(schema_payload),
        artifact_sha256=_sha256_bytes(path.read_bytes()),
    )


def probability_from_vectors(diff: np.ndarray, coefficients: np.ndarray) -> float:
    if diff.ndim != 1 or coefficients.ndim != 1 or len(diff) != len(coefficients):
        raise ValueError("Judge vector and coefficient dimensions differ")
    logit = float(np.dot(diff, coefficients))
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def context_metadata(
    target: Mapping[str, Any],
    output_metadata: Mapping[str, Any],
    schema: Mapping[str, tuple[Any, ...]],
    context_original: list[str],
) -> dict[str, Any]:
    """Match the frozen Judge's metadata construction from the exact XML context."""
    message_texts = message_text_timeline(context_original)
    if not message_texts:
        raise ValueError("C0 returned no textual context for Judge metadata")
    latest = message_texts[-1]
    return {
        "relationship": str(target.get("relationship") or ""),
        "source_chat_id": str(target.get("chat_id") or "__unknown_chat__"),
        "group_name": str(output_metadata.get("chat_name") or "__unknown_group_name__"),
        "recent_speakers": tuple(reversed(speaker_timeline(context_original)))[:RECENT_SPEAKER_POSITIONS],
        "latest_message_has_question_mark": any(marker in latest for marker in ("?", "？")),
        "latest_message_has_at_mention": "@" in latest,
        "recent_group_other_texts": group_other_text_timeline(context_original)[-GROUP_PATTERN_RECENT_MESSAGE_LIMIT:],
        # Passing schema through makes the construction's frozen dependency explicit.
        "_frozen_schema_keys": tuple(sorted(schema)),
    }


def score_formal_judge_pair(
    judge: FormalJudge,
    llm_features: Mapping[str, Any],
    reply_a: list[str],
    reply_b: list[str],
    metadata: Mapping[str, Any],
) -> float:
    """Return P(A is the more human-like reply), with strict A/B symmetry."""
    features = {
        "option_A": validate_option_features(llm_features.get("option_A")),
        "option_B": validate_option_features(llm_features.get("option_B")),
    }
    option_a, option_b, names = vectorize_boolean_options(features)
    observed_a, observed_names = vectorize_observable_option_booleans(reply_a, dict(metadata))
    observed_b, other_observed_names = vectorize_observable_option_booleans(reply_b, dict(metadata))
    group_a, group_names = vectorize_group_pattern_option_boolean(reply_a, dict(metadata))
    group_b, other_group_names = vectorize_group_pattern_option_boolean(reply_b, dict(metadata))
    if observed_names != other_observed_names or group_names != other_group_names:
        raise ValueError("A/B observable feature schemas differ")
    option_names = [*names, *observed_names, *group_names]
    vector_a = np.concatenate((option_a, observed_a, group_a))
    vector_b = np.concatenate((option_b, observed_b, group_b))
    context, context_names = vectorize_context_booleans(dict(metadata), judge.context_schema)
    expanded_a, expanded_names = expand_boolean_option_with_refined_context(
        vector_a, option_names, context, context_names
    )
    expanded_b, swapped_names = expand_boolean_option_with_refined_context(
        vector_b, option_names, context, context_names
    )
    if expanded_names != swapped_names or tuple(expanded_names) != judge.feature_names:
        raise ValueError("formal Judge artifact and current feature schema differ")
    probability = probability_from_vectors(expanded_a - expanded_b, judge.coefficients)
    swapped_probability = probability_from_vectors(expanded_b - expanded_a, judge.coefficients)
    if not math.isclose(probability + swapped_probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("formal Judge A/B probabilities are not complementary")
    return probability


def _event_has_tool_call(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    event_type = str(event.get("type", "")).lower()
    item_type = str(item.get("type", "")).lower()
    blocked = ("tool_call", "function_call", "command_execution", "mcp_tool", "web_search")
    return any(marker in event_type or marker in item_type for marker in blocked)


def build_api_judge_messages(
    case: dict[str, Any],
    reference: dict[str, Any] | None = None,
    judge_profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    reference = reference or {"facts": [], "examples": []}
    profile_mode = str((judge_profile or {}).get("mode") or "replace")
    if judge_profile is not None and profile_mode not in {"replace", "append_to_builtin"}:
        raise ValueError(f"unsupported Judge profile mode: {profile_mode}")
    if judge_profile is None or profile_mode == "append_to_builtin":
        appended_instruction = str(judge_profile["instructions"]) if judge_profile is not None else ""
        system = (
            "你是独立盲测 Judge。判断匿名回复 A、B 中哪个更像该聊天中真人本人回复。"
            "你只会收到当前这一道题，不能参考其他测试题。"
            "参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。"
            "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作测试答案。"
            "参考资料和当前聊天没有提供的个人事实一律视为未知，不得自行判定真实或编造。"
            + appended_instruction
            + "不得调用工具或外部资料。只输出一个 JSON 对象："
            "{\"human_option\":\"A或B\",\"confidence\":0到1,\"reason\":\"简短理由\"}。\n\n"
            "<reference_pack>\n"
            + json.dumps(
                {"facts": reference.get("facts", []), "examples": reference.get("examples", [])},
                ensure_ascii=False,
            )
            + "\n</reference_pack>"
        )
    else:
        system = (
            "你是独立盲测 Judge。你只会收到当前这一道题，不能参考其他测试题。"
            "参考资料来自与测试聊天隔离的 2026-04-01 前真人聊天，以及用户确认的本人事实。"
            "参考资料只用于理解稳定事实和表达习惯，不得把参考示例当作测试答案。"
            + str(judge_profile["instructions"])
            + "不得调用工具或外部资料。只输出一个 JSON 对象："
            "{\"option_a_thread\":\"A具体承接的问题、状态或并行话题\","
            "\"option_a_counterevidence\":\"A不像真人的最强反证\","
            "\"option_b_thread\":\"B具体承接的问题、状态或并行话题\","
            "\"option_b_counterevidence\":\"B不像真人的最强反证\","
            "\"human_option\":\"A或B\",\"confidence\":0到1,\"reason\":\"简短理由\"}。\n\n"
            "<reference_pack>\n"
            + json.dumps(
                {"facts": reference.get("facts", []), "examples": reference.get("examples", [])},
                ensure_ascii=False,
            )
            + "\n</reference_pack>"
        )
    user = "<blind_case>\n" + json.dumps(case, ensure_ascii=False) + "\n</blind_case>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_shared_single_case_prompt(
    case: dict[str, Any],
    reference: dict[str, Any] | None = None,
    judge_profile: dict[str, Any] | None = None,
) -> str:
    messages = build_api_judge_messages(case, reference, judge_profile)
    return (
        "<judge_instructions>\n"
        + messages[0]["content"]
        + "\n</judge_instructions>\n\n"
        + messages[1]["content"]
    )


def _parse_api_judge_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        try:
            if start < 0 or end <= start:
                raise json.JSONDecodeError("missing JSON object", text, 0)
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            option = re.search(r'"human_option"\s*:\s*"([AB])"', text)
            if not option:
                raise ValueError("API Judge did not return an unambiguous human_option")
            confidence = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text)
            reason = re.search(r'"reason"\s*:\s*"(.*)"\s*\}?\s*$', text, re.DOTALL)
            payload = {
                "human_option": option.group(1),
                "confidence": float(confidence.group(1)) if confidence else None,
                "reason": reason.group(1) if reason else "（原始响应 JSON 格式损坏，详见审计）",
            }
    if payload.get("human_option") not in {"A", "B"}:
        raise ValueError("API Judge returned an invalid human_option")
    parsed = {
        "human_option": payload["human_option"],
        "confidence": payload.get("confidence"),
        "reason": str(payload.get("reason") or ""),
    }
    if "selected_reply" in payload:
        parsed["selected_reply"] = payload.get("selected_reply")
    for field in (
        "option_a_thread",
        "option_a_person_specific_evidence",
        "option_a_counterevidence",
        "option_b_thread",
        "option_b_person_specific_evidence",
        "option_b_counterevidence",
        "decisive_evidence",
        "uncertainty",
    ):
        if field in payload:
            parsed[field] = str(payload.get(field) or "")
    return parsed
