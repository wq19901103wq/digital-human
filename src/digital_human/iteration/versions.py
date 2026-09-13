"""版本与指针（SOP §1/§6）：四种对象，版本即目录，创建后只读，晋升只切指针。

对象：
- data/<d-XXXX>/      数据版本：切分结果、池、人格工作区、manifest（冻结）
- generators/<g-XXXX>/ 生成器版本：行为配置 + persona/场景自包含快照（不可变）
- judges/<j-XXXX>/     Judge 版本：判别配置 + 模型 + meta（训练来源）
- experiments/<e-…>/  实验：见 experiment.py（唯一执行状态源）

指针文件 private/pointers.json 只有四个字段；版本内容不可变，
所以指针不需要一致性校验——指向不存在的版本是唯一错误。
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ..config import ConfigError, ROOT, resolve, sha256_file


def switch_instance(name: str) -> Path:
    """切换数字人实例：每个数字人（如 example-agent / example-operator）一套独立私有树，
    版本、指针、实验完全隔离；机制层与超参全局共享。
    实例名取 env DH_INSTANCE 或各脚本的 --instance 参数。"""
    global PRIVATE, DATA_ROOT, GEN_ROOT, JUDGE_ROOT, POINTERS_PATH
    PRIVATE = ROOT / "instances" / name
    DATA_ROOT = PRIVATE / "data"
    GEN_ROOT = PRIVATE / "generators"
    JUDGE_ROOT = PRIVATE / "judges"
    POINTERS_PATH = PRIVATE / "pointers.json"
    return PRIVATE


import os as _os

switch_instance(_os.environ.get("DH_INSTANCE", "default"))


# ---------- 基础设施 ----------

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_dir(path: Path) -> Path:
    """版本目录 write-once：已存在即拒绝（不可变由创建点保证）。"""
    if path.exists():
        raise ConfigError(f"版本已存在（不可变）: {path}")
    path.mkdir(parents=True)
    return path


def _finalize_readonly(d: Path) -> None:
    """SOP 总原则2/§3「禁止手工编辑」的代码化：版本目录创建完成后置只读。
    要改内容 = 新建版本（ bumps 指针），而不是改旧版本。"""
    import stat
    # 锁 config/manifest/meta/persona/prompt 与顶层目录；scenarios 子树不锁
    # （避免与 copytree 权限传播互相干扰；目录级 r-x 已阻止新增/删除文件）
    entries = [f for f in d.rglob("*") if "scenarios" not in f.parts[len(d.parts):]]
    # 两遍：先确保全部可写（目录加 u+w），再统一锁只读；个别文件锁不住不阻断创建
    for f in entries:
        try:
            f.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass
    for f in entries:
        try:
            f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    try:
        d.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def _next_id(existing: list[str], prefix: str) -> str:
    nums = []
    for name in existing:
        m = re.match(rf"^{re.escape(prefix)}-(\d+)$", name)
        if m:
            nums.append(int(m.group(1)))
    return f"{prefix}-{max(nums, default=0) + 1:04d}"


def _dir_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


# ---------- 指针 ----------

def load_pointers() -> dict[str, str]:
    """五指针：data / production_gen / iteration_gen / production_judge / iteration_judge。
    旧四指针文件自动补 iteration_judge（= judge/production_judge）。"""
    if not POINTERS_PATH.exists():
        raise ConfigError(f"缺少指针文件 {POINTERS_PATH}；请先运行 scripts/bootstrap.py")
    ptr = _read_json(POINTERS_PATH)
    if "judge" in ptr and "production_judge" not in ptr:
        ptr["production_judge"] = ptr["judge"]
    if "production_judge" in ptr and "iteration_judge" not in ptr:
        ptr["iteration_judge"] = ptr["production_judge"]
    return ptr


def save_pointers(pointers: dict[str, str]) -> None:
    _write_json(POINTERS_PATH, pointers)


# ---------- 数据版本 ----------

def create_data_version(
    version_dir_name: str | None,
    manifest: dict[str, Any],
) -> str:
    """bootstrap 把切分产物写入后调用；manifest 由调用方填好。"""
    vid = version_dir_name or _next_id(
        [p.name for p in DATA_ROOT.iterdir()] if DATA_ROOT.exists() else [], "d"
    )
    d = _create_dir(DATA_ROOT / vid)
    manifest = {**manifest, "id": vid, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(d / "manifest.json", manifest)
    return vid  # 注意：数据版本在构建完成后由 finalize_data_version 冻结（场景/池写入在后）


def finalize_data_version(vid: str) -> None:
    _finalize_readonly(data_version_dir(vid))


def data_version_dir(vid: str) -> Path:
    d = DATA_ROOT / vid
    if not (d / "manifest.json").exists():
        raise ConfigError(f"数据版本不存在: {d}")
    return d


def current_data_version() -> Path:
    return data_version_dir(load_pointers()["data"])


# ---------- 生成器版本 ----------

def create_generator_version(
    behavior_cfg: dict[str, Any],
    data_vid: str,
    version_dir_name: str | None = None,
    root: Path | None = None,
) -> str:
    """行为配置 + 从数据版本快照 persona/场景。版本簿记字段不允许出现。
    root 自定义时（如实验目录内的候选版本）不生成 g-XXXX 序号名。"""
    forbidden = {"baseline_id", "adopted_from_run", "adopted_via"}
    leaked = forbidden & set(behavior_cfg)
    if leaked:
        raise ConfigError(f"生成器版本配置不得包含版本簿记字段: {leaked}")
    src = data_version_dir(data_vid)
    root = root or GEN_ROOT
    gid = version_dir_name or _next_id(
        [p.name for p in root.iterdir()] if root.exists() else [], "g"
    )
    d = _create_dir(root / gid)
    shutil.copyfile(src / "persona.md", d / "persona.md")
    shutil.copytree(src / "scenarios", d / "scenarios")
    cfg = {
        **behavior_cfg,
        "data_version": data_vid,
        "persona_sha256": sha256_file(d / "persona.md"),
    }
    _write_json(d / "config.json", cfg)
    _finalize_readonly(d)
    return gid


def generator_dir(gid: str) -> Path:
    d = GEN_ROOT / gid
    if not (d / "config.json").exists():
        raise ConfigError(f"生成器版本不存在: {d}")
    return d


def load_generator(gid: str) -> dict[str, Any]:
    d = generator_dir(gid)
    return {"id": gid, "dir": d, "config": _read_json(d / "config.json")}


def current_generator(role: str) -> dict[str, Any]:
    key = "production_gen" if role == "production" else "iteration_gen"
    return load_generator(load_pointers()[key])


# ---------- Judge 版本 ----------

def create_judge_version(
    behavior_cfg: dict[str, Any],
    meta: dict[str, Any],
    version_dir_name: str | None = None,
    root: Path | None = None,
    source_dir: Path | None = None,
    assets: dict[str, bytes] | None = None,
) -> str:
    """Judge 版本 = 判别配置 + 提示词快照（SOP §4：提示词随版本冻结，
    改全局模板不影响旧版本）。meta 记录来源/校准实验。"""
    root = root or JUDGE_ROOT
    jid = version_dir_name or _next_id(
        [p.name for p in root.iterdir()] if root.exists() else [], "j"
    )
    behavior_cfg = dict(behavior_cfg)
    if assets is None and source_dir is not None:
        assets = {name: (source_dir / name).read_bytes()
                  for name in ["prompt.md", *behavior_cfg.get("assets", {})]}
    if assets is None:
        if behavior_cfg.get("mode", "pairwise_llm") != "pairwise_llm":
            raise ConfigError("导入型 Judge 必须提供完整的来源资产快照")
        assets = {"prompt.md": (ROOT / "prompts" / "judge_pairwise.template.md").read_bytes()}
    for name in assets:
        if Path(name).name != name or name in {"config.json", "meta.json"}:
            raise ConfigError(f"Judge 资产文件名非法：{name}")
    recorded = behavior_cfg.get("prompt_sha256")
    actual = hashlib.sha256(assets["prompt.md"]).hexdigest()
    if recorded and recorded != actual:
        raise ConfigError(
            f"Judge 配置 prompt_sha256 与模板不符（记录 {recorded[:12]}… 实际 {actual[:12]}…）"
        )
    for name, digest in behavior_cfg.get("assets", {}).items():
        if name not in assets or hashlib.sha256(assets[name]).hexdigest() != digest:
            raise ConfigError(f"Judge 资产缺失或指纹不符：{name}")
    behavior_cfg["prompt_sha256"] = actual
    d = _create_dir(root / jid)
    for name, blob in assets.items():
        (d / name).write_bytes(blob)
    _write_json(d / "config.json", behavior_cfg)
    _write_json(d / "meta.json", meta)
    _finalize_readonly(d)
    return jid


def judge_dir(jid: str) -> dict[str, Any]:
    d = JUDGE_ROOT / jid
    if not (d / "config.json").exists():
        raise ConfigError(f"Judge 版本不存在: {d}")
    return {
        "id": jid,
        "dir": d,
        "config": _read_json(d / "config.json"),
        "meta": _read_json(d / "meta.json"),
    }


def current_judge(role: str = "production") -> dict[str, Any]:
    key = "production_judge" if role == "production" else "iteration_judge"
    return judge_dir(load_pointers()[key])


def version_payload(version_dir: Path) -> dict[str, Any]:
    """版本内容的唯一读取入口（收敛项：快照是所有操作的唯一依据）。
    返回 config + 快照资产 hash + meta；任何模块需要版本内容一律从这里取。"""
    payload: dict[str, Any] = {}
    cfg = version_dir / "config.json"
    if cfg.exists():  # 数据版本工作区无 config.json，容忍
        payload["config"] = _read_json(cfg)
    from ..config import sha256_file
    persona = version_dir / "persona.md"
    if persona.exists():
        payload["persona_sha256"] = sha256_file(persona)
    prompt = version_dir / "prompt.md"
    if prompt.exists():
        payload["prompt_sha256"] = sha256_file(prompt)
    sdir = version_dir / "scenarios"
    if sdir.exists():
        payload["scenarios"] = {f.name: sha256_file(f) for f in sorted(sdir.glob("*.md"))}
    meta = version_dir / "meta.json"
    if meta.exists():
        payload["meta"] = _read_json(meta)
    return payload


def dir_fingerprint(path: Path) -> str:
    return _dir_sha256(path)[:16]
