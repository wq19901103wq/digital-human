"""Replace only the exhausted fixed sample; preserve learning and development."""
from collections import Counter
from copy import deepcopy
import json
import shutil

from ..config import ConfigError, sha256_file
from ..iteration import datasets, versions
from ..iteration.storage import read_json, write_json
from . import history


def preserved_files(purpose):
    return {'messages.jsonl', 'fewshot_pool.jsonl', 'report.json'} | {
        entry['file'] for role, entry in purpose['roles'].items() if role != 'fixed_test'}


def reserved_batches():
    """Exclude all previously reserved fixed cases, without consulting outcomes."""
    blocked, bindings = set(), {}
    for directory in sorted(versions.DATA_ROOT.glob('d-*')):
        if not (directory / 'fixed_test.jsonl').is_file():
            continue
        path = datasets.case_path(directory, 'fixed_test')
        bindings[directory.name] = sha256_file(path)
        with path.open() as stream:
            blocked.update(str(json.loads(line)['case_id']) for line in stream if line.strip())
    return blocked, bindings


def capacity_report(rows, protocol, exclusions):
    """Describe quota exhaustion without exposing case contents or changing sampling."""
    heldout = set(protocol['unseen_chat_ids'])
    total = protocol['acceptance_total']
    familiar_total = round(total * protocol['familiar_weight'])
    report = {}
    for cohort, size in (('familiar', familiar_total), ('unseen_holdout', total - familiar_total)):
        group_total = round(size * protocol['group_weight'])
        for kind, need in (('group', group_total), ('private', size - group_total)):
            matching = [r for r in rows if r['relationship'] == kind and
                (r['source_span']['chat_id'] in heldout if cohort == 'unseen_holdout' else
                 r['source_span']['chat_id'] not in heldout and r['familiarity'] == 'familiar')]
            entry = {'required': need}
            for name, excluded in exclusions.items():
                chats = Counter(r['source_span']['chat_id'] for r in matching
                                if str(r['id']) not in excluded)
                entry[name] = {'cases': sum(chats.values()), 'chats': len(chats),
                    'capacity': sum(min(n, protocol['evaluation_chat_cap']) for n in chats.values())}
                # Capacity only: report a possible policy change for review,
                # without applying it or selecting cases under different rules.
                entry[f'minimum_cap_{name}'] = next((cap for cap in
                    range(1, max(chats.values(), default=0) + 1)
                    if sum(min(n, cap) for n in chats.values()) >= need), None) if need else 0
            report[f'{cohort}/{kind}'] = entry
    return report


def plan(parent_ref, *, seed, fixed_chat_cap=None):
    parent = versions.data_version_dir(parent_ref)
    purpose = datasets.manifest(parent)
    if not purpose or purpose.get('history_policy') != 'complete_before_input_v1':
        raise ConfigError('固定批次更新需要完整历史用途清单')
    protocol = dict(purpose['protocol'])
    previous_cap = purpose['roles']['fixed_test'].get('chat_cap', protocol['evaluation_chat_cap'])
    cap = previous_cap if fixed_chat_cap is None else fixed_chat_cap
    if type(cap) is not int or cap <= 0:
        raise ConfigError('固定验收每聊天上限必须为正整数')
    # This override applies only to the new fixed batch. The frozen development
    # protocol and every nonfixed role remain byte-identical to the parent.
    protocol['evaluation_chat_cap'] = cap
    excluded, bindings = reserved_batches()
    with datasets.case_path(parent, 'fixed_test').open() as stream:
        parent_ids = {str(json.loads(line)['case_id']) for line in stream if line.strip()}
    heldout = set(protocol['unseen_chat_ids'])
    familiar, unseen, window = [], [], []
    with (parent / 'fewshot_pool.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['source_span']['start_timestamp'] < protocol['acceptance_start']:
                continue
            window.append(row)
            if str(row['id']) in excluded:
                continue
            if row['source_span']['chat_id'] in heldout:
                unseen.append(row)
            elif row['familiarity'] == 'familiar':
                familiar.append(row)
    capacity = capacity_report(window, protocol, {
        'before_exclusions': set(), 'after_parent': parent_ids, 'after_all_reserved': excluded})
    if any(entry['after_all_reserved']['capacity'] < entry['required'] for entry in capacity.values()):
        raise ConfigError('固定批次容量不足（未创建数据或启动评测）：' +
                          json.dumps(capacity, ensure_ascii=False))
    total = protocol['acceptance_total']
    familiar_total = round(total * protocol['familiar_weight'])
    selected = []
    for rows, count in ((familiar, familiar_total), (unseen, total - familiar_total)):
        if count:
            selected.extend(history.choose(rows, count, protocol['group_weight'],
                                           protocol['evaluation_chat_cap'], seed))
    cases = [history.as_case(row, heldout) for row in selected]
    proof = dict(schema=1, parent_ref=parent_ref, seed=seed,
        fixed_sampling=dict(previous_chat_cap=previous_cap, chat_cap=cap),
        parent_manifest_sha256=sha256_file(parent / 'manifest.json'),
        parent_purposes_sha256=sha256_file(parent / 'purposes.json'),
        preserved_files={name: sha256_file(parent / name) for name in sorted(preserved_files(purpose))},
        excluded_batches=bindings)
    return cases, proof


def refresh(parent_ref, *, seed, fixed_chat_cap=None, build=False):
    """Publish an unadopted immutable child. check.py data is the adoption audit."""
    with versions.file_lock(versions.PRIVATE / '.iteration.lock'):
        cases, proof = plan(parent_ref, seed=seed, fixed_chat_cap=fixed_chat_cap)
        summary = dict(parent_ref=parent_ref, seed=seed, total=len(cases),
            fixed_sampling=proof['fixed_sampling'],
            max_cases_per_chat=max(Counter(c['source_span']['chat_id'] for c in cases).values()),
            chat_types=dict(Counter(c['chat_type'] for c in cases)),
            cohorts=dict(Counter(c['familiarity'] for c in cases)),
            excluded_batches=list(proof['excluded_batches']))
        if not build:
            return summary
        parent = versions.data_version_dir(parent_ref)
        manifest = read_json(parent / 'manifest.json')
        vid = versions.create_data_version(None, manifest)
        directory = versions.data_version_dir(vid)
        for name, digest in proof['preserved_files'].items():
            shutil.copyfile(parent / name, directory / name)
            if sha256_file(directory / name) != digest:
                raise ConfigError('复制固定批次期间来源变化')
        purpose = deepcopy(datasets.manifest(parent))
        purpose.update(data_ref=vid, acceptance_refresh=proof)
        fixed = directory / purpose['roles']['fixed_test']['file']
        history.write_rows(fixed, cases)
        entry = purpose['roles']['fixed_test']
        entry.update(sha256=sha256_file(fixed), total=len(cases),
            chat_cap=proof['fixed_sampling']['chat_cap'],
            chat_types=summary['chat_types'], cohorts=summary['cohorts'],
            chats=len({c['source_span']['chat_id'] for c in cases}))
        write_json(directory / 'purposes.json', purpose)
        manifest = read_json(directory / 'manifest.json')
        manifest['purposes_sha256'] = sha256_file(directory / 'purposes.json')
        manifest['testsets']['fixed_test'] = {**entry,
            'group': entry['chat_types'].get('group', 0),
            'private': entry['chat_types'].get('private', 0), 'file_sha256': entry['sha256']}
        write_json(directory / 'manifest.json', manifest)
        datasets.fixed_refresh_compatible(parent_ref, vid, required=True)
        versions.finalize_data_version(vid)
        return dict(summary, data_ref=vid, adopted=False, requires='check.py data and prepare-data')
