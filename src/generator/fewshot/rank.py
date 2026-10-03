"""fewshot.rank：逐题打分的纯函数层，由 src/generator/few_shot.py 逐字搬迁。

代码与旧模块逐字一致（本模块自含全部依赖，无需任何相对 import），行为零变化
由 tests/test_fewshot_package.py 锁定。
"""

import hashlib
import re
from collections import Counter
from typing import Any

_SKILL_SITUATIONS = {
    "handling_praise": "praise",
    "handling_vent": "complaint",
    "answering_questions": "question",
    "receiving_share": "share",
}

_PROFILE_SITUATIONS = {
    "praise": "praise",
    "hostile_teasing": "teasing",
    "teasing": "teasing",
    "vent": "complaint",
    "complaint": "complaint",
    "question": "question",
    "invitation": "invitation_request",
    "request": "invitation_request",
    "self_deprecation": "self_deprecation",
    "share": "share",
}



def _chat_id(chat_name: str) -> str:
    return f"chat_{hashlib.sha256(chat_name.encode('utf-8')).hexdigest()[:10]}"


def _terms(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    chars = [char for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff"]
    singles = chars if len(chars) <= 12 else chars[:12]
    return singles + ["".join(chars[i:i + 2]) for i in range(max(0, len(chars) - 1))]


def _is_sparse_turn(text: str) -> bool:
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())
    return 0 < len(compact) <= 4 and not re.search(r"[?？]|为什么|为啥|怎么|咋|什么|啥|谁|哪|多少", text)


def _is_brief_exclamation(text: str) -> bool:
    compact = re.sub(r"[\s，,。.!！~～…]+", "", text.lower())
    return bool(re.fullmatch(r"(?:我操|我擦|卧槽|我靠|擦|艹|草|妈呀|天哪|天呐|啊这|绝了)", compact))


def _latest_turn_form(text: str) -> str:
    """Classify the latest incoming turn by its concrete conversational form."""
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return ""
    if _is_brief_exclamation(compact):
        return "brief_exclamation"
    if re.search(r"(?:还是|或者|或是).{0,18}[?？]$", compact):
        return "choice_question"
    if re.search(r"(?:为什么|为啥|怎么|咋|什么|啥|谁|哪|多少|几).{0,24}[?？]?$", compact):
        return "wh_question"
    if re.search(r"(?:是不是|有没有|能不能|可不可以|要不要|行不行|对不对|吗|么)[?？]?$", compact):
        return "yes_no_question"
    if compact.endswith(("?", "？")):
        return "other_question"
    if len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)) <= 6:
        return "brief_statement"
    return "statement"


def _group_participant_profile(messages: list[dict[str, str]]) -> tuple[bool, bool, bool, bool, bool]:
    cleaned = [
        {"sender": str(message.get("sender") or "").strip(), "text": str(message.get("text") or "").strip()}
        for message in messages
        if str(message.get("text") or "").strip()
    ]
    if not cleaned:
        return (False, False, False, False, False)
    latest = cleaned[-1]
    previous = cleaned[-2] if len(cleaned) >= 2 else None
    other_senders = {message["sender"] for message in cleaned if message["sender"] not in {"", "我"}}
    mentions_other = "@" in latest["text"] or any(
        sender != latest["sender"] and sender in latest["text"]
        for sender in other_senders
        if len(sender) >= 2
    )
    return (
        bool(previous and previous["sender"] == "我" and latest["sender"] != "我"),
        mentions_other,
        bool(previous and previous["sender"] != latest["sender"]),
        any(message["sender"] == "我" for message in cleaned[:-1]),
        len(other_senders) >= 2,
    )


def _has_strong_concrete_payload(row: dict[str, Any]) -> bool:
    messages = [str(value).strip() for value in row.get("context") or [] if str(value).strip()]
    text = "".join(messages)
    compact = re.sub(r"\s+", "", text)
    return (
        len(messages) > 1
        or len(compact) > 10
        or bool(re.search(r"\d|[a-zA-Z]{2,}", text))
    )


def _has_context_bound_reply(row: dict[str, Any]) -> bool:
    replies = [str(value).strip() for value in row.get("reply") or [] if str(value).strip()]
    return any(
        len(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", reply.lower())) > 5
        or bool(re.search(r"\d|[a-zA-Z]{2,}", reply))
        for reply in replies
    )


def _has_portable_sparse_reply(row: dict[str, Any]) -> bool:
    replies = [str(value).strip() for value in row.get("reply") or [] if str(value).strip()]
    if not replies:
        return False
    portable = re.compile(
        r"(?:"
        r"怎么了|咋了|咋啦|咋回事|什么情况|真的假的|为啥|为什么|比如|然后呢|后来呢|你呢|"
        r"惨|太惨了|难受|确实|唉|哎|哎呀|抱抱|没事吧|还好吗|辛苦了|"
        r"卧槽|我操|我擦|我靠|笑死|哈哈+|hhh+|牛逼|牛|离谱|绝了|"
        r"可怕|这么可怕|少来|哪有|别闹|可还行|行吧|好吧|可以|羡慕|我也羡慕|"
        r"问问|问下|确认下|看看|试试|在|在呢|来了"
        r")"
    )
    return all(
        portable.fullmatch(re.sub(r"[\s，,。.!！?？~～…]+", "", reply.lower()))
        for reply in replies
    )


def _needs_clarification(compact: str) -> bool:
    vague_subject = re.search(
        r"(?:有些|有的|某些|一些)(?:个)?(?:词|话|东西|地方|问题|内容|部分|情况|事)",
        compact,
    )
    unresolved = re.search(
        r"(?:不知道|不清楚|不确定|不会|没想好|不好)(?:该|要)?(?:怎么|如何|咋|翻|说|弄|处理|选|写|回)",
        compact,
    )
    if vague_subject and unresolved:
        return True
    if re.search(r"(?:我感觉|我觉得).{0,18}(?:没必要|没有必要|很多(?:种)?(?:办法|方法|解决办法))[啊呀吧。！？]*$", compact):
        return True
    if re.search(r"我(?:现在)?跟以前.{0,8}(?:不一样|不同|变了)(?:的)?[啊呀吧。！？]*$", compact):
        return True
    if re.fullmatch(r"(?:也)?太坑了?[啊呀吧]*", compact):
        return True
    return bool(re.fullmatch(r"(?:我)?(?:今天|刚刚|刚才)?刚面(?:完)?的?[啊呀吧]*", compact))


def _situation_tags(
    text: str,
    interaction_context: str = "",
    enable_clarification: bool = True,
) -> set[str]:
    """识别聊天处境；只看互动动作，不把话题词当成处境。"""
    compact = re.sub(r"\s+", "", text.lower())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last_line = re.sub(r"\s+", "", lines[-1].lower()) if lines else compact
    established_context = len(lines) > 1 and len("".join(lines[:-1])) >= 30
    tags: set[str] = set()
    if _is_brief_exclamation(last_line):
        return {"brief_exclamation"}
    if enable_clarification and _needs_clarification(compact):
        tags.add("needs_clarification")
    presence_call = bool(
        re.search(r"(?:^|[，,。；;！!？?])(?:在吗|人呢|哪呢|冒个泡)(?:$|[，,。；;！!？?])", last_line)
    )
    if presence_call and not (established_context and len(last_line) <= 8):
        tags.add("calling_presence")
    if re.search(r"(?:^|[，,。；;！!？?])(?:[^，,。；;！!？?]{0,8}[，,])?出来(?:玩|吃|喝|见|聚|pk|冒个泡)?(?:$|[，,。；;！!？?])", compact):
        tags.add("calling_presence")
    if any(marker in compact for marker in ("一起", "来不来", "要不要", "走不走", "帮我", "帮忙", "能不能", "可不可以")):
        tags.add("invitation_request")
    state_or_departure = bool(
        re.search(r"我不行(?:了)?[，,。；;！! ]*(?:我)?(?:要|得|先)?(?:去)?(?:睡|休息|躺|撤|走)", compact)
    )
    if not state_or_departure and any(marker in compact for marker in ("我不行", "我太菜", "我真菜", "我废了", "羡慕你们", "羡慕各位", "又亏了", "亏麻了")):
        tags.add("self_deprecation")
    if any(marker in compact for marker in ("牛逼", "厉害", "真强", "太强", "大佬", "优秀", "羡慕", "恭喜", "赚死", "少爷")):
        tags.add("praise")
    if "self_deprecation" in tags and ("羡慕你们" in compact or "羡慕各位" in compact):
        tags.discard("praise")
    if any(marker in compact for marker in ("烦死", "好烦", "累死", "好累", "难受", "疼死", "崩了", "倒霉", "气死", "惨了")):
        tags.add("complaint")
    if (
        any(marker in compact for marker in ("在路上", "出发了", "下班了", "快到了", "刚到家", "已经到家"))
        or re.search(r"(?:我|你|他|她|我们|你们|他们)(?:刚|已经|快|马上|还没)?到(?:了|家|公司|机场|门口)", compact)
    ):
        tags.add("arrival_status")
    if (
        "http://" in compact
        or "https://" in compact
        or any(marker in compact for marker in ("看这个", "这个发你", "发给你", "分享给你", "我分享个"))
    ):
        tags.add("share")
    if any(marker in compact for marker in ("你可真", "就这", "韭菜", "吹牛", "装起来", "复读机", "少装", "别装")):
        tags.add("teasing")
    if any(marker in compact for marker in ("?", "？", "为什么", "怎么", "咋", "什么", "啥", "谁", "哪", "多少", "是不是", "有没有")):
        tags.add("question")
    if re.search(r"(?:牛逼|厉害|真强|优秀|大佬).{0,3}(?:吗|么)[?？]?$", compact):
        tags.discard("praise")
    for skill_name, tag in _SKILL_SITUATIONS.items():
        if skill_name in interaction_context:
            tags.add(tag)
    return tags


def _precise_situation(text: str) -> str:
    """Return one concrete conversational setup when the wording is reliable."""
    compact = re.sub(r"\s+", "", text.lower())
    if re.search(r"(?:那|所以)?你.{0,12}(?:昨天|刚才|前面|之前|上次).{0,18}(?:后来|然后|现在).{0,8}(?:呢|怎么样|咋样|如何)", compact):
        return "thread_followup"
    if re.search(r"(?:人呢|你?在哪(?:里|儿)?|到哪(?:里|儿)?了?|什么位置|发个定位)", compact):
        return "location_followup"
    if (
        re.search(r"(?:^|[，,。；;！!？?])(?:我|我们)?(?:已经|刚)?到(?:了|门口).{0,12}(?:是|对不对|没错吧).{0,10}(?:吗|吧|[?？])", compact)
        or re.search(r"(?:地址|房间|楼栋|门牌).{0,10}(?:对|是|没错).{0,4}(?:吗|吧|[?？])", compact)
        or re.search(r"(?:是|在|到)(?:几号楼|几楼|几室|\d+号楼|\d+楼|\d+室|\d+号门)(?:吗|吧|[?？])", compact)
        or re.search(r"(?:几号楼|几楼|几室|哪个楼|哪栋|哪个房间)(?:吗|呢|吧|[?？])", compact)
    ):
        return "address_confirmation"
    if re.search(r"(?:帮我|帮忙|麻烦你|替我|给我).{0,16}(?:拿|取|带|买|改|看|发|问|弄|处理|接|送|查)", compact):
        return "task_request"
    if re.search(r"(?:能不能|可不可以|可以不可以).{0,6}(?:帮|替|给).{0,16}(?:拿|取|带|买|改|看|发|问|弄|处理|接|送|查)", compact):
        return "task_request"
    if (
        any(marker in compact for marker in ("一起", "来不来", "要不要", "走不走"))
        or re.search(r"出来(?:吃|喝|玩|见|聚)", compact)
    ) and not re.search(r"(?:帮我|帮忙|麻烦你|替我|给我)", compact):
        return "invitation"
    if re.search(r"(?:你可真|你又|你还|就你|少装|别装|真能吹|会吹|装起来)", compact):
        return "playful_teasing"
    if any(marker in compact for marker in ("烦死", "好烦", "累死", "好累", "难受", "疼死", "崩了", "倒霉", "气死", "惨了")):
        return "emotion_vent"
    return ""


def _situation_profile(
    text: str,
    interaction_context: str = "",
    enable_clarification: bool = True,
    enable_information_gap_response_move: bool = False,
) -> dict[str, str]:
    """Describe the social setup, not merely the topic or shared words."""
    compact = re.sub(r"\s+", "", text.lower())
    tags = _situation_tags(text, interaction_context, enable_clarification)
    profile: dict[str, str] = {}
    if "brief_exclamation" in tags:
        profile["expected_response_move"] = "instant_reaction"
    if "praise" in tags or "self_deprecation" in tags:
        if any(marker in compact for marker in ("你们", "各位", "大家")):
            profile["praise_target"] = "listener_group"
        elif (
            re.search(r"(?:^|[，,。；;！!？?])你[^，,。；;！!？?]{0,12}(?:牛逼|厉害|真强|太强|大佬|优秀)", compact)
            or re.search(r"(?:羡慕|恭喜)[^，,。；;！!？?]{0,6}你", compact)
        ):
            profile["praise_target"] = "listener"
        elif re.search(r"(?:他|她|他们|她们|这人|那人)[^，,。；;！!？?]{0,12}(?:牛逼|厉害|真强|太强|大佬|优秀)", compact):
            profile["praise_target"] = "third_party"
        if (
            "self_deprecation" in tags
            or any(marker in compact for marker in ("我不行", "我太菜", "我真菜", "我不配", "我这种穷", "穷逼", "特别痛苦"))
        ):
            profile["speaker_posture"] = "self_lowering"
    if enable_information_gap_response_move and _has_information_gap(text):
        profile["expected_response_move"] = "verify_or_request_detail"
    elif "brief_exclamation" in tags:
        pass
    elif "needs_clarification" in tags:
        profile["expected_response_move"] = "clarify_detail"
    elif "calling_presence" in tags:
        profile["expected_response_move"] = "brief_presence"
    elif "invitation_request" in tags:
        profile["expected_response_move"] = "accept_decline_or_clarify"
    elif "teasing" in tags:
        profile["expected_response_move"] = "playful_counter"
    elif "complaint" in tags:
        profile["expected_response_move"] = "acknowledge_or_share_feeling"
    elif profile.get("speaker_posture") == "self_lowering" and profile.get("praise_target") in {"listener", "listener_group"}:
        profile["expected_response_move"] = "downplay_or_counter_praise"
    elif profile.get("praise_target") in {"listener", "listener_group"}:
        profile["expected_response_move"] = "deflect_or_playfully_accept"
    elif profile.get("praise_target") == "third_party":
        profile["expected_response_move"] = "agree_or_comment"
    elif "self_deprecation" in tags:
        profile["expected_response_move"] = "acknowledge_or_reassure"
    elif "question" in tags:
        profile["expected_response_move"] = "direct_answer"
    elif "share" in tags:
        profile["expected_response_move"] = "acknowledge_share"
    return profile


def _is_clarification_reply(row: dict[str, Any]) -> bool:
    reply = re.sub(r"[\s，,。.!！?？~～]+", "", "".join(str(value) for value in row.get("reply") or []))
    return bool(re.fullmatch(r"(?:比如|例如|举个例子|哪个|哪些|怎么说|具体点呢?|具体什么|哪方面|哪部分)", reply))


def _has_information_gap(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    lines = [re.sub(r"\s+", "", line.lower()) for line in text.splitlines() if line.strip()]
    return bool(
        re.search(r"(?:听|据|传|消息|线人|内部|hr|朋友|同学|亲戚|叔叔|阿姨)[^，,。；;！!？?]{0,16}(?:说|称|透露|告诉|爆料)", compact)
        or re.search(r"(?:人品|性格|能力|水平|技术|表现)[^，,。；;！!？?]{0,8}(?:好|差|强|弱|靠谱|优秀|不行)", compact)
        or re.search(r"(?:同学|朋友|熟人|亲戚)[^，,。；;！!？?]{0,18}(?:来看看|看一下|报价|问问)", compact)
        or bool(lines and re.fullmatch(r"(?:[a-z]*\d+[a-z\d]*|\d+[a-z]+)", lines[-1]))
    )


def _is_information_gap_reply(row: dict[str, Any]) -> bool:
    reply = re.sub(r"\s+", "", "".join(str(value) for value in row.get("reply") or []))
    return bool(
        re.search(r"(?:哪(?:里|儿)?的消息|消息哪来的|谁说的|听谁说的|真的假的)", reply)
        or re.search(r"(?:怎么|咋)(?:看出|判断)|(?:为什么|为啥).{0,8}(?:觉得|认为|这么说)", reply)
        or re.search(r"(?:是不是|是).{0,10}(?:老板|设计师|同学|朋友|同事|领导|员工)", reply)
        or re.search(r"(?:没有|有)(?:完整|全的)|还缺|缺.{0,8}(?:信息|号码|数字|字段)", reply)
    )


def _row_situation_profile(
    row: dict[str, Any],
    enable_clarification: bool = True,
    enable_information_gap_response_move: bool = False,
) -> dict[str, str]:
    inferred = _situation_profile(
        "\n".join(str(value) for value in row.get("context") or []),
        enable_clarification=enable_clarification,
        enable_information_gap_response_move=enable_information_gap_response_move,
    )
    if enable_clarification and _is_clarification_reply(row):
        inferred["expected_response_move"] = "clarify_detail"
    if enable_information_gap_response_move and _is_information_gap_reply(row):
        inferred["expected_response_move"] = "verify_or_request_detail"
    explicit = row.get("situation_profile")
    if isinstance(explicit, dict) and explicit:
        result = {str(key): str(value) for key, value in explicit.items() if value}
        if inferred.get("expected_response_move") == "clarify_detail":
            result["expected_response_move"] = "clarify_detail"
        if inferred.get("expected_response_move") == "verify_or_request_detail":
            result["expected_response_move"] = "verify_or_request_detail"
        return result
    return inferred


def _profile_compatibility(query: dict[str, str], row: dict[str, str]) -> float:
    score = 0.0
    weights = {
        "praise_target": 2.5,
        "speaker_posture": 2.0,
        "expected_response_move": 3.0,
    }
    for key, weight in weights.items():
        query_value = query.get(key)
        row_value = row.get(key)
        if not query_value or not row_value:
            continue
        score += weight if query_value == row_value else -weight
    return score


def _row_situation_tags(
    row: dict[str, Any],
    enable_clarification: bool = True,
) -> set[str]:
    inferred = _situation_tags(
        "\n".join(str(value) for value in row.get("context") or []),
        enable_clarification=enable_clarification,
    )
    if enable_clarification and _is_clarification_reply(row):
        inferred.add("needs_clarification")
    explicit_tags = row.get("situation_tags")
    if isinstance(explicit_tags, list):
        tags = {str(value) for value in explicit_tags if value}
        if tags:
            if "needs_clarification" in inferred:
                tags.add("needs_clarification")
            return tags
    profile = row.get("semantic_profile") or {}
    incoming_act = str(profile.get("incoming_act") or "")
    tags = {
        mapped
        for key, mapped in _PROFILE_SITUATIONS.items()
        if key in incoming_act
    }
    if tags:
        return tags
    return inferred

