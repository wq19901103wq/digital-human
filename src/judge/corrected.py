"""来源 已采用裁判的迁移与执行：大模型判断 + 冻结 LR 保守校正。"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import ConfigError, sha256_file
from .. import cache, tracing
from . import corrected_v1 as runtime

MODE = "corrected_pairwise"
RUNTIME_PATH = Path(runtime.__file__)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_import(source_root: Path) -> tuple[dict, dict, dict[str, bytes]]:
    """先验证来源，再返回自包含资产；不导入训练集或固定集答案。"""
    source_root = source_root.resolve()
    source_path = source_root / "config/current_judge.json"
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if (source.get("status") != "adopted" or not source.get("runnable_as_official")
            or source.get("prediction_type") != "llm_boolean_features_with_logistic_correction"):
        raise ConfigError("来源裁判 尚未采用或不是已支持的冻结校正裁判")
    execution = source["execution"]
    if execution.get("provider") != "codex_cli":
        raise ConfigError("来源裁判 必须显式使用 codex_cli，不可替换成 API 默认模型")
    source_c0 = _json(source_root / "config/current_c0.json")
    c0_id = "c0-" + hashlib.sha256(json.dumps(source_c0, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    if c0_id != source.get("required_current_c0_baseline_id"):
        raise ConfigError("来源裁判 未绑定来源项目当前采用的 C0")
    # 拒绝用旧推理代码解释来源项目中新改过的特征定义。
    manifest = _json(RUNTIME_PATH.with_name("corrected_runtime_sources.json"))
    trees = {}
    for name, spec in manifest.items():
        if spec["path"] not in trees:
            trees[spec["path"]] = ast.parse((source_root / spec["path"]).read_text(encoding="utf-8"))
        nodes = [n for n in trees[spec["path"]].body if getattr(n, "name", None) == name
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        if len(nodes) != 1 or hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() != spec["ast_sha256"]:
            raise ConfigError(f"来源 推理实现已变化，请先更新迁移适配：{name}")

    assets = {"source_judge.json": raw}
    for filename, spec in [("profile.json", source["profile"]),
                           ("reference.json", source["reference"]),
                           ("correction.json", source["correction"]["artifact"])]:
        blob = (source_root / spec["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
            raise ConfigError(f"来源 资产指纹不匹配：{filename}")
        assets[filename] = blob
    correction = json.loads(assets["correction.json"])
    model = correction["final_model"]
    for key in ("model", "input_feature_count", "fit_intercept", "regularization", "C"):
        if model.get(key) != source["correction"][key]:
            raise ConfigError(f"来源 校正模型参数不匹配：{key}")
    threshold = source["correction"]["flip_confidence_threshold"]
    if model.get("correction_threshold") != threshold or not 0.5 <= threshold <= 1:
        raise ConfigError("来源 校正阈值不匹配")
    system = runtime.build_api_judge_messages(
        {}, json.loads(assets["reference.json"]), json.loads(assets["profile.json"])
    )[0]["content"]
    assets["prompt.md"] = ("<judge_instructions>\n" + system
                           + "\n</judge_instructions>\n\n<blind_case>\n{{case}}\n</blind_case>").encode()
    config = {
        "mode": MODE, "llm": dict(execution), "correction_threshold": threshold,
        "runtime_sha256": sha256_file(RUNTIME_PATH),
        "assets": {name: hashlib.sha256(blob).hexdigest() for name, blob in assets.items() if name != "prompt.md"},
        "prompt_sha256": hashlib.sha256(assets["prompt.md"]).hexdigest(),
    }
    baseline = "judge-" + hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    meta = {"origin": "wechat-mac", "source_baseline_id": baseline,
            "source_name": source["name"], "source_config_sha256": hashlib.sha256(raw).hexdigest(),
            "source_config": str(source_path), "source_adopted_at": source.get("adopted_at"),
            "required_source_c0": source.get("required_current_c0_baseline_id"),
            "note": "初始化对齐来源已采用裁判；沿用提示词、参考资料、524 维冻结模型及 0.7 校正阈值"}
    return config, meta, assets


class CodexJudgeClient:
    def __init__(self, config: dict):
        self.config = config
        if config.get("provider") != "codex_cli":
            raise ConfigError("来源 裁判只支持已冻结的 codex_cli 通道")
        version = subprocess.run(["codex", "--version"], text=True, capture_output=True, check=True, timeout=10).stdout.strip().split()[-1]
        if version != config["codex_cli_version"]:
            raise ConfigError(f"Codex CLI 版本不匹配：需要 {config['codex_cli_version']}，当前 {version}")

    def cache_identity(self):
        return {"provider": "codex_cli", "model": self.config["model"],
                "reasoning_effort": self.config["reasoning_effort"],
                "codex_cli_version": self.config["codex_cli_version"],
                "model_revision": self.config.get("model_revision"), "adapter": cache.code_digest(__file__)}

    def run(self, prompt: str, schema: dict | None = None) -> str:
        return cache.memo("llm_request", {"client": self.cache_identity(), "prompt": prompt, "schema": schema},
                          lambda: self._run(prompt, schema), valid=cache.response_valid)

    def _run(self, prompt: str, schema: dict | None = None) -> str:
        from ..iteration.control import request
        with request(), tempfile.TemporaryDirectory(prefix="dh-judge-") as directory:
            output = Path(directory) / "last_message.json"
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                       "--sandbox", "workspace-write", "--cd", directory, "--skip-git-repo-check",
                       "--json", "--output-last-message", str(output), "--model", self.config["model"],
                       "--config", f'model_reasoning_effort="{self.config["reasoning_effort"]}"']
            if schema:
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
                command += ["--output-schema", str(schema_path)]
            with tracing.step("codex", {"prompt": prompt, "schema": schema, "config": self.config}) as call:
                result = subprocess.run(command + ["-"], input=prompt, text=True, capture_output=True,
                                        check=False, timeout=self.config["timeout_seconds"])
                raw = output.read_text(encoding="utf-8") if output.exists() else ""
                call["response"] = {"text": raw, "exit_code": result.returncode}
                if result.returncode:
                    raise RuntimeError(f"Codex Judge 执行失败（exit {result.returncode}）：{result.stderr[-500:]}")
                events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
                call["response"]["usage"] = next((e.get("usage") for e in events if e.get("type") == "turn.completed"), None)
                if any(runtime._event_has_tool_call(event) for event in events):
                    raise RuntimeError("Codex Judge 调用了工具，本题无效")
                return raw


def blind_case(case: dict, reply_a: list[str], reply_b: list[str]) -> tuple[dict, dict]:
    """统一消息转来源中立时间线；角色和时间必须来自导出记录。"""
    messages = case.get("context") or []
    if not messages or case.get("chat_type") not in {"private", "group"}:
        raise ConfigError("来源裁判 需要聊天类型和完整消息上下文")
    if any(not isinstance(m.get("is_self"), bool) or not m.get("timestamp") for m in messages):
        raise ConfigError("旧数据缺少 is_self/timestamp；请先用迁移脚本补齐裁判上下文元数据")
    boundary = max((i for i, m in enumerate(messages) if m["is_self"]), default=-1)
    fragments = []
    for tag, rows in [("conversation_history", messages[:boundary + 1]),
                      ("messages_to_reply", messages[boundary + 1:])]:
        if not rows:
            continue
        rendered = []
        for message in rows:
            seconds = float(message["timestamp"])
            if seconds > 10_000_000_000:
                seconds /= 1000
            stamp = datetime.fromtimestamp(seconds, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
            node = ET.Element("message", role="self" if message["is_self"] else "other", type="text")
            sender = "我" if message["is_self"] else message["sender"]
            node.text = f'{sender}（{stamp}）：{message["text"]}'
            rendered.append(ET.tostring(node, encoding="unicode", short_empty_elements=False))
        fragments.append(f"<{tag}>\n" + "\n".join(rendered) + f"\n</{tag}>")
    blind = {"relationship": case["chat_type"], "context_original": fragments,
             "option_A": reply_a, "option_B": reply_b}
    metadata = runtime.context_metadata(
        {"relationship": case["chat_type"], "chat_id": case.get("source_chat_id")},
        {"chat_name": case.get("chat_name")}, {}, fragments)
    return blind, metadata


class CorrectedJudge:
    mode = MODE

    def __init__(self, config: dict, version_dir: Path, client=None, feature_client=None):
        self.config = config
        from ..iteration.learning_guard import MaterialSeal
        self._material_seal = MaterialSeal(version_dir)
        self.source_check = None
        if sha256_file(RUNTIME_PATH) != config["runtime_sha256"]:
            raise ConfigError("来源裁判 推理代码与冻结版本不一致")
        for name, digest in {**config["assets"], "prompt.md": config["prompt_sha256"]}.items():
            if Path(name).name != name or sha256_file(version_dir / name) != digest:
                raise ConfigError(f"来源裁判 冻结资产不匹配：{name}")
        self.template = (version_dir / "prompt.md").read_text(encoding="utf-8")
        self.model = runtime.load_formal_judge(version_dir / "correction.json")
        self.client = client or CodexJudgeClient(config["llm"])
        feature_config = {**config["llm"], **config.get("feature_llm", {})}
        self.feature_client = feature_client or (CodexJudgeClient(feature_config) if config.get("feature_llm") else self.client)
        self.last_verdict = {}
        self.on_progress = None
        self._check_sources()

    def _check_sources(self):
        self._material_seal.check()
        if self.source_check:
            self.source_check()

    def _stage(self, phase):
        self._check_sources()
        if self.on_progress:
            self.on_progress(phase)

    def is_ai(self, case: dict, candidate_replies: list[str]) -> bool:
        self._check_sources()
        initial, feature = cache.client_identity(self.client), cache.client_identity(self.feature_client)
        identity = {"config": self.config, "initial_client": initial, "feature_client": feature,
                    "case": {k: case.get(k) for k in ("context", "human_reply", "chat_type", "source_chat_id", "chat_name")},
                    "candidate_replies": candidate_replies, "runtime": cache.code_digest(__file__, RUNTIME_PATH)}
        def produce():
            result = self._is_ai(case, candidate_replies, identity if initial and feature else None)
            self._check_sources()
            return {"identified_ai": result, "last_verdict": self.last_verdict}
        result = cache.memo("judge", identity if initial and feature else None, produce)
        self._check_sources()
        self.last_verdict = result["last_verdict"]
        return result["identified_ai"]

    def _is_ai(self, case: dict, candidate_replies: list[str], identity=None) -> bool:
        swap = cache.memo("blind_order", identity, lambda: random.random() < 0.5)
        human = list(case["human_reply"])
        a, b = (candidate_replies, human) if swap else (human, candidate_replies)
        blind, metadata = blind_case(case, a, b)
        tracing.note("blind_mapping", {"candidate_option": "A" if swap else "B", "human_option": "B" if swap else "A",
                                      "blind_case": blind, "context_metadata": metadata})
        prompt = self.template.replace("{{case}}", json.dumps(blind, ensure_ascii=False))
        self._stage("judging")
        def initial_verdict():
            self._check_sources()
            with cache.validation(runtime._parse_api_judge_response):
                value = runtime._parse_api_judge_response(self.client.run(prompt))
            self._check_sources()
            return value
        initial_identity = cache.client_identity(self.client)
        base = cache.memo("judge_initial", {"prompt": prompt, "client": initial_identity,
                          "parser": cache.code_digest(RUNTIME_PATH)} if initial_identity else None, initial_verdict)
        tracing.note("base_verdict", base)
        features_prompt = runtime.feature_extractor_prompt(blind)
        for attempt in range(2):
            self._stage("extracting")
            with cache.validation(runtime.parse_feature_response):
                raw = self.feature_client.run(features_prompt, runtime.feature_response_json_schema())
            self._check_sources()
            try:
                feature_identity = cache.client_identity(self.feature_client)
                features = cache.memo("judge_features", {"raw": raw, "parser": cache.code_digest(RUNTIME_PATH)}
                                      if feature_identity else None, lambda: runtime.parse_feature_response(raw))
                tracing.note("features", {"attempt": attempt + 1, "valid": True, "parsed": features})
                break
            except ValueError as exc:
                tracing.note("features", {"attempt": attempt + 1, "valid": False, "error": str(exc), "raw": raw})
                if attempt == 1:
                    raise
        self._stage("predicting")
        probability = runtime.score_formal_judge_pair(self.model, features, a, b, metadata)
        small = "A" if probability >= 0.5 else "B"
        confidence = max(probability, 1 - probability)
        corrected = small != base["human_option"] and confidence >= self.config["correction_threshold"]
        final = small if corrected else base["human_option"]
        self.last_verdict = {"base": base, "small_model_probability_a": probability,
                             "correction_applied": corrected, "human_option": final,
                             "candidate_option": "A" if swap else "B"}
        tracing.note("correction", {**self.last_verdict, "small_model_human_option": small,
                                   "confidence": confidence, "threshold": self.config["correction_threshold"],
                                   "identified_ai": final != ("A" if swap else "B")})
        return final != ("A" if swap else "B")
