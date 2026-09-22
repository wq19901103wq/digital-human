"""从冻结版本读取可核对的改动；任务说明与实际差异分别展示。"""
from __future__ import annotations

from pathlib import Path

from ..config import sha256_file
from ..iteration.versions import version_payload
from ..judge import normalize_mode

MODES = {'corrected_pairwise': '大模型初判与抽特征 + 小模型预测与校正',
         'pairwise_llm': '单一大模型配对盲测'}
_LABELS = {'mode': '裁判机制', 'llm': '大模型', 'feature_llm': '抽特征大模型', 'private': '私聊', 'group': '群聊',
           'model': '模型', 'timeout_seconds': '超时上限（秒）', 'max_tokens': '输出 token 上限配置',
           'temperature': '生成温度配置', 'provider': '调用通道', 'reasoning_effort': '推理强度',
           'protocol': '接口协议', 'codex_cli_version': 'Codex CLI 版本',
           'max_shots_per_case': '每题风格示例上限', 'shots_char_budget': '风格示例字数上限',
           'retriever': '风格召回', 'enabled': '是否启用', 'correction_threshold': '小模型改判阈值',
           'runtime_sha256': '裁判推理实现指纹', 'api_key_env': '密钥环境变量名', 'base_url_env': '接口环境变量名'}


def _flatten(value: dict, prefix: str = '') -> dict:
    result = {}
    for key, item in value.items():
        path = prefix + key
        if isinstance(item, dict):
            result.update(_flatten(item, path + '.'))
        else:
            result[path] = item
    return result


def _format(key, value):
    if value is None:
        return '未显式设置'
    if key == 'mode':
        return MODES.get(value, value)
    if key == 'correction_threshold':
        return f'{value:.0%}'
    if isinstance(value, bool):
        return '启用' if value else '关闭'
    if key.endswith('sha256'):
        return str(value)[:12]
    return str(value)


def snapshot_delta(instance: Path, folder: str, before: str, after: str) -> dict:
    """配置、实际提示词和裁判资产对比；数据标记不冒充生成策略改动。"""
    paths = [instance / folder / str(ref) for ref in (before, after)]
    payloads = [version_payload(path) for path in paths]
    if any('config' not in p for p in payloads):
        return {'available': False, 'changes': [], 'metadata': [], 'before': before, 'after': after}
    configs = [p['config'] for p in payloads]
    if any(c.get('feature_llm') for c in configs):
        configs = [{**c, 'feature_llm': {**c.get('llm', {}), **c.get('feature_llm', {})}}
                   if normalize_mode(c.get('mode')) == 'corrected_pairwise' else c for c in configs]
    unavailable = {'available': False, 'changes': [], 'metadata': [], 'before': before, 'after': after}
    ignored = {'data_version', 'baseline_id', 'assets', 'persona_sha256', 'prompt_sha256'}
    flat = [_flatten({k: v for k, v in c.items() if k not in ignored}) for c in configs]
    changes = []
    for key in sorted(flat[0].keys() | flat[1].keys()):
        old, new = flat[0].get(key), flat[1].get(key)
        if old != new:
            changes.append({'label': ' · '.join(_LABELS.get(k, k) for k in key.split('.')),
                            'before': _format(key, old), 'after': _format(key, new), 'key': key})
    assets = []
    for path, payload, cfg in zip(paths, payloads, configs):
        if any(key in cfg and key not in payload for key in ('persona_sha256', 'prompt_sha256')):
            return unavailable
        files = {label: payload[key] for key, label in [('persona_sha256', '人格提示词'), ('prompt_sha256', '裁判提示词')]
                 if key in payload}
        files.update({'场景规则 · ' + name: digest for name, digest in (payload.get('scenarios') or {}).items()})
        for filename in cfg.get('assets', {}):
            target = path / filename
            if Path(filename).name != filename or not target.resolve().is_relative_to(path.resolve()):
                return unavailable
            if not target.is_file():
                return unavailable
            label = {'correction.json': '小模型权重与特征定义', 'profile.json': '裁判规则',
                     'reference.json': '裁判参考资料', 'source_judge.json': '来源配置'}.get(filename, filename)
            files[label] = sha256_file(target)
        assets.append(files)
    for key in sorted(assets[0].keys() | assets[1].keys()):
        old, new = assets[0].get(key), assets[1].get(key)
        if old != new:
            changes.append({'label': key, 'before': '未包含' if old is None else '原快照',
                            'after': '已移除' if new is None else '新增快照' if old is None else '内容已更新', 'key': key})
    priority = {'mode': 0, 'llm.model': 1, '小模型权重与特征定义': 2, 'correction_threshold': 3}
    changes.sort(key=lambda row: (priority.get(row['key'], 10), row['label']))
    metadata = []
    for key, label in [('data_version', '生成器快照的数据标记'), ('baseline_id', '来源基线标记')]:
        old, new = configs[0].get(key), configs[1].get(key)
        if old != new:
            metadata.append({'label': label, 'before': _format(key, old), 'after': _format(key, new), 'key': key})
    return {'available': True, 'changes': changes, 'metadata': metadata, 'before': before, 'after': after}
