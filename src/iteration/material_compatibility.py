"""Read-only source coverage diagnostics when evaluation data changes.

Reports bound learning records, not model execution or adoption eligibility.
The normal experiment guards remain authoritative and are never bypassed here.
"""
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import datasets
from .storage import read_json


def _require(ok, message):
    if not ok:
        raise ConfigError(message)


def _read(path, bindings, expected=None):
    path = Path(path).resolve()
    actual = sha256_file(path)
    _require(expected is None or expected == actual, f'学习来源内容变化: {path.name}')
    bindings[str(path)] = actual
    return read_json(path)


def _bound(path, evidence, bindings):
    path = Path(path).resolve()
    _require(str(path) in evidence, f'学习来源未绑定: {path.name}')
    return _read(path, bindings, evidence[str(path)])


def _marker(directory, bindings):
    value = _read(directory / 'learning.json', bindings)
    _require(value.get('verified') is True and value.get('evidence_files'), '缺少已绑定学习来源')
    return value


def _row(record):
    return dict(chat_id=record['source_span']['chat_id'],
                information_end=record['source_span']['end_timestamp'],
                message_ids=record['context_message_ids'] + record['reply_message_ids'])


def judge_sources(directory, bindings):
    marker = _marker(directory, bindings)
    evidence = marker['evidence_files']
    def source(suffix):
        paths = [p for p in evidence if p.endswith('/' + suffix)]
        _require(len(paths) == 1, f'Judge 学习来源不唯一: {suffix}')
        return _bound(paths[0], evidence, bindings)
    sources, audit = source('sources.json'), source('source_audit.json')
    train = [_row(row) for row in sources['train']]
    references = [dict(chat_id=row['chat_id'], information_end=row['information_end'],
                       message_ids=row['message_ids']) for row in audit['reference_spans']]
    _require(max(r['information_end'] for r in train + references) ==
             marker['information_end'] == audit['training_information_end'], 'Judge 学习截止记录不一致')
    return dict(judge_training=train, judge_references=references), audit['data_ref']


def ranker_sources(directory, bindings):
    marker = _marker(directory, bindings)
    name = 'ranker/provenance.json'
    _require(name in marker['asset_files'], '生成器没有绑定的排序模型来源')
    proof = _read(directory / name, bindings, marker['asset_files'][name])
    evidence = proof['evidence_files']
    model = Path(proof['model_directory'])
    parent = _bound(model.parent / 'manifest.json', evidence, bindings)
    manifest = _bound(Path(parent['source']) / 'manifest.json', evidence, bindings)
    inventory = Path(manifest['inventory'])
    labeling = _bound(inventory / 'labeling/manifest.json', evidence, bindings)
    _require(labeling['data'] == proof['data_ref'], '排序模型学习数据记录不一致')
    targets, examples, identities = [], {}, set()
    for selected in labeling['targets']:
        identity = selected['target_id']
        _require(identity not in identities, '排序模型学习上文重复')
        identities.add(identity)
        row = _bound(inventory / 'targets' / (identity + '.json'), evidence, bindings)
        _require(row['target_id'] == identity, '排序模型学习上文绑定不符')
        targets.append(_row(row['target']))
        for entry in row['candidates']:
            example = entry['example']
            value = _row(example)
            _require(example['id'] not in examples or examples[example['id']] == value,
                     '同一示例的学习来源不一致')
            examples[example['id']] = value
    _require(len(targets) == proof['samples']['contexts'] and
             len(examples) == proof['samples']['unique_examples'], '排序模型学习来源数量不一致')
    _require(max(r['information_end'] for r in targets + list(examples.values())) <=
             proof['information_end'] == marker['information_end'], '排序模型学习截止记录不一致')
    return dict(ranker_targets=targets, ranker_examples=list(examples.values())), proof['data_ref']


def summarize(cases, unseen_chats, materials):
    """Keep unseen-chat exposure separate from answer overlap and future data."""
    unseen_chats = set(unseen_chats)
    exposed, answer_cases, future_cases, details = set(), set(), set(), {}
    unseen_cases = {c['case_id'] for c in cases if c['source_span']['chat_id'] in unseen_chats}
    for name, rows in materials.items():
        chats = {r['chat_id'] for r in rows}
        messages = {m for r in rows for m in r['message_ids']}
        end = max((r['information_end'] for r in rows), default=0)
        affected = {c['case_id'] for c in cases if c['case_id'] in unseen_cases and
                    c['source_span']['chat_id'] in chats}
        answers = {c['case_id'] for c in cases if messages.intersection(c['reply_message_ids'])}
        future = {c['case_id'] for c in cases if end >= c['input_cutoff']['timestamp']}
        exposed.update(affected)
        answer_cases.update(answers)
        future_cases.update(future)
        details[name] = dict(records=len(rows), information_end=end,
            records_in_unseen_chats=sum(r['chat_id'] in unseen_chats for r in rows),
            unseen_chats_exposed=len(chats & unseen_chats),
            development_unseen_cases_exposed=len(affected),
            development_answer_id_overlap_cases=len(answers), development_time_conflict_cases=len(future))
    return dict(development_cases=len(cases), unseen_cases=len(unseen_cases),
        unseen_cases_exposed=len(exposed), unseen_cases_unexposed=len(unseen_cases - exposed),
        answer_id_overlap_cases=len(answer_cases), time_conflict_cases=len(future_cases),
        source_conflicts=bool(answer_cases or future_cases), materials=details)


def require_current(data_ref, directory, source_loader):
    """After original-source reconstruction, guard reuse on changed development data.

    Familiar identities are allowed. The existing static-source gate also keeps
    the global learning cutoff before both development and fixed evaluation.
    """
    import json
    from . import versions
    data = versions.data_version_dir(data_ref)
    bindings = {}
    materials, _ = source_loader(directory, bindings)
    path = datasets.case_path(data, 'development')
    cases = [json.loads(line) for line in path.read_text().splitlines() if line]
    _require(bool(cases), '开发集为空')
    result = summarize(cases, (), materials)
    _require(not result['source_conflicts'], '学习材料包含新版评测答案或未来信息')
    _require(all(sha256_file(Path(p)) == h for p, h in bindings.items()), '检查期间来源发生变化')
    return result


def inspect(data_dir, judge_dir, generator_dir):
    """Only read development and bound learning sources; never open fixed answers."""
    import json
    data_dir, judge_dir, generator_dir = map(Path, (data_dir, judge_dir, generator_dir))
    bindings = {}
    purpose = _read(data_dir / 'purposes.json', bindings)
    path = datasets.case_path(data_dir, 'development')
    bindings[str(path.resolve())] = sha256_file(path)
    cases = [json.loads(line) for line in path.read_text().splitlines() if line]
    _require(bool(cases), '开发集为空')
    judge, judge_data = judge_sources(judge_dir, bindings)
    ranker, ranker_data = ranker_sources(generator_dir, bindings)
    result = summarize(cases, purpose['protocol']['unseen_chat_ids'], {**judge, **ranker})
    _require(all(sha256_file(Path(p)) == expected for p, expected in bindings.items()), '检查期间来源发生变化')
    return dict(schema=1, scope='bound_source_records_only_not_model_execution',
        data_ref=data_dir.name, judge_ref=judge_dir.name, generator_ref=generator_dir.name,
        learning_data=dict(judge=judge_data, ranker=ranker_data), summary=result,
        source_conflicts=result['source_conflicts'],
        evaluation_authorized=False, bindings=bindings)
