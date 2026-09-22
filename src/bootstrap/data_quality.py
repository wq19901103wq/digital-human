"""Source-backed dataset audit, shared by publication and scripts/check.py.

Only aggregate metadata is returned, including for sealed acceptance data.
Historical datasets are inspected against the requested timing policy without
rewriting their original protocol or certifying old experiment results.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..config import ConfigError, sha256_file
from . import conversations


def acceptance_binding(directory):
    """Bind reusable acceptance to all audited inputs and the checking code.

    The frozen manifest binds the raw exports checked by import_coverage. Later
    adoption checks these archive/manifest bytes, without reimporting the corpus.
    """
    from ..generator.history_sources import stamp
    from ..iteration import datasets
    directory = Path(directory).resolve()
    purpose = datasets.manifest(directory)
    files = [directory / n for n in
             ('manifest.json', 'messages.jsonl', 'purposes.json', 'fewshot_pool.jsonl', 'report.json')]
    files += [datasets.case_path(directory, role) for role in purpose['roles']]
    root = Path(__file__).parents[1]
    code = [root / n for n in ('bootstrap/data_quality.py', 'bootstrap/import_coverage.py',
        'bootstrap/ingest.py', 'bootstrap/conversations.py', 'generator/history_sources.py',
        'generator/history.py', 'iteration/datasets.py', 'config.py')]
    code.append(root.parent / 'scripts' / 'check.py')
    before = [stamp(p) for p in files + code]
    result = dict(schema=1, directory=str(directory),
        files={str(p.relative_to(directory)): sha256_file(p) for p in files},
        code={str(p.relative_to(root.parent)): sha256_file(p) for p in code})
    if before != [stamp(p) for p in files + code]:
        raise ConfigError('验收输入在读取过程中变化')
    return result


def accepted_report(directory, report):
    """Read a full, frozen-policy check.py report; partial checks cannot adopt."""
    directory, report = Path(directory), Path(report)
    value = json.loads(report.read_text())
    result = value.get('data_quality', {}).get(directory.name, {})
    if (value.get('mode') != 'data' or value.get('status') != 'passed'
            or value.get('evaluation_started') is not False
            or value.get('model_requests_prohibited') is not True
            or result.get('passed') is not True or not result.get('groups')
            or not result.get('frozen_policy') or result.get('policy') != result['frozen_policy']):
        raise ConfigError('数据采用需要按冻结规则通过的完整 check.py data 报告')
    binding = acceptance_binding(directory)
    if result.get('acceptance_binding') != binding:
        raise ConfigError('数据内容或验收代码与报告不符；不能采用旧验收报告')
    return dict(report=str(report.resolve()), report_sha256=sha256_file(report), binding=binding)


def require_passed(result):
    if not result['passed']:
        raise ConfigError('数据质量检查未通过；详见 data_quality 聚合报告')
    return result


def audit(directory, *, segmentation=None):
    from ..generator.history_sources import HistorySources
    directory = Path(directory)
    source = HistorySources(directory)
    rule = conversations.validate_policy(segmentation or source.segmentation or conversations.policy())
    result = dict(data_ref=directory.name, policy=rule, frozen_policy=source.segmentation,
                  source_hashes=source.hashes, groups={}, isolation={})
    def inspect(rows, *, example=False, validated=False):
        counts, bubbles, chat_types, response_bins = Counter(), Counter(), Counter(), Counter()
        seen, affected, total, largest = set(), 0, 0, dict(context=0, response=0, reply=0)
        for row in rows:
            if not validated:
                source.validate(row, example=example)
            identity = row['id' if example else 'case_id']
            if identity in seen:
                raise ConfigError('重复样本 ID')
            seen.add(identity)
            span = row['source_span']
            messages = source.chats[span['chat_id']]
            context = messages[span['start']:span['reply_start']]
            reply = messages[span['reply_start']:span['end']]
            problems = conversations.issues(context, reply, rule)
            counts.update(problems)
            affected += bool(problems)
            total += 1
            bubbles[len(reply)] += 1
            chat_types[reply[0].chat_type] += 1
            response = conversations.gap(context[-1], reply[0])
            for seconds in (120, 300, 600, 1800, 7200, 86400):
                response_bins[f'over_{seconds}s'] += response > seconds
            largest['response'] = max(largest['response'], response)
            for name, fragment in [('context', context), ('reply', reply)]:
                largest[name] = max(largest[name], max((conversations.gap(a, b)
                    for a, b in zip(fragment, fragment[1:])), default=0))
        return dict(total=total, affected=affected, reasons=dict(counts),
                    response_gap_counts=dict(response_bins), max_gaps_seconds=largest,
                    reply_bubbles=dict(sorted(bubbles.items())), chat_types=dict(chat_types))

    with (directory / 'fewshot_pool.jsonl').open() as stream:
        result['groups']['fewshot_pool'] = inspect((json.loads(line) for line in stream if line.strip()), example=True)
    role_ids = {}
    for role in source.purpose['roles']:
        rows = source.role(role)
        result['groups'][role] = inspect(rows, validated=True)
        declared = source.purpose['roles'][role].get('total', len(rows))
        if declared != len(rows):
            raise ConfigError(f'{role} 样本数与清单不符')
        role_ids[role] = {mid for row in rows for mid in row['context_message_ids'] + row['reply_message_ids']}
    # Learning roles may overlap each other; judge/gen development is explicitly
    # shared. Neither is allowed to contain fixed answers or future fragments.
    learning = set().union(*(role_ids.get(r, set()) for r in ('gen_learning', 'gen_optimization', 'judge_training')))
    dev = role_ids.get('development', set()) | role_ids.get('judge_development', set())
    fixed = role_ids.get('fixed_test', set())
    result['isolation'] = dict(learning_development_overlap=len(learning & dev),
        learning_fixed_overlap=len(learning & fixed), development_fixed_overlap=len(dev & fixed))
    source.check()
    result['passed'] = not any(g['affected'] for g in result['groups'].values()) and not any(result['isolation'].values())
    if source.protocol.get('message_ingestion'):
        from .import_coverage import audit as audit_imports
        result['import_coverage'] = audit_imports(directory)
        result['passed'] = result['passed'] and result['import_coverage']['passed']
    result['scope'] = '原始来源、消息身份、时间连续性、用途时间及片段隔离；时间接近不保证语义上一定在回复'
    return result
