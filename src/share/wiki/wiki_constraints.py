"""Sealed, finite corrections using stricter wire schemas and original validators."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from ... import cache
from ...config import sha256_file
from ...iteration.storage import file_lock, read_json, write_json
from .wiki_library import digest
from .wiki_library_extract import validate_result

CONTRACT = 'source_constraints_v3'
LIMIT = 3


def provenance():
    from . import wiki_batch, wiki_cached
    return dict(pipeline=wiki_batch.pipeline(), constraints=sha256_file(Path(__file__)),
                locator=sha256_file(Path(wiki_cached.__file__)))


def wire_plan(prompt, schema):
    properties = schema.get('properties', {})
    marker = prompt.find('{"subject":')
    if marker < 0:
        return None
    try:
        payload = json.loads(prompt[marker:])
    except ValueError:
        return None
    wire_schema = deepcopy(schema)
    fields = wire_schema['properties']
    if set(properties) == {'items', 'changes'} and 'review_lines' in payload:
        supported = deepcopy(fields.pop('changes'))
        supported['items']['properties']['action']['enum'] = ['retain', 'correct']
        supported['items']['properties']['fact_ids']['minItems'] = 1
        unsupported = deepcopy(supported)
        unsupported['items']['properties']['action']['enum'] = ['omit', 'unresolved']
        unsupported['items']['properties']['fact_ids'].pop('minItems', None)
        fields.update(supported_changes=supported, unsupported_changes=unsupported)
        adapter = 'evidence_partition_v1'
        instruction = ('本次仅改变输出结构：changes 改为 supported_changes 和 unsupported_changes。'
            'retain/correct 只能在 supported_changes 中，必须引用真正支持该条旧稿的非空 fact_ids；'
            '没有证据的旧稿放 unsupported_changes，action=omit/unresolved。'
            '全部旧稿仍须恰好复核一次；items 不变；不得用无关事实凑引用。\n')
    elif set(properties) == {'entities', 'attributes', 'relations', 'events', 'addresses', 'reviews'}:
        anchors = payload.get('source_contract', {}).get('person_refs', [])
        if {item.get('ref') for item in anchors} != {'E0', 'E1'} or len(anchors) != 2:
            return None
        entity = fields.pop('entities')
        fields['other_entities'] = deepcopy(entity)
        for anchor in anchors:
            ref = anchor['ref']
            if anchor['account'] != payload['subject']['account' if ref == 'E0' else 'self_account']:
                raise ValueError('person anchor account does not match subject')
            person = deepcopy(entity)
            person['maxItems'] = 1 if anchor['anchor_lines'] else 0
            person_fields = person['items']['properties']
            for name, value in (('ref', ref), ('kind', 'person'), ('account', anchor['account'])):
                person_fields[name] = dict(person_fields[name], enum=[value])
            if anchor['anchor_lines']:
                person_fields['lines']['items']['enum'] = list(anchor['anchor_lines'])
            fields[ref.lower() + '_entities'] = person
        adapter = 'person_anchor_partition_v1'
        instruction = ('本次仅改变输出结构：entities 分为 e0_entities、e1_entities、other_entities。'
            'E0/E1 各自最多一项，只能 kind=person，账号和作者锚点必须符合对应 schema。'
            '没有锚点就输出空数组，不强制创建人物；other_entities 禁止使用 E0/E1。'
            '作者锚点只证明账号存在，不能证明个人事实。其余字段、事实覆盖和归属标准不变。\n')
    else:
        return None
    wire_schema['required'] = list(fields)
    wire_prompt = instruction + prompt
    seal = dict(adapter=adapter, prompt_sha256=digest(prompt), schema_sha256=digest(schema),
                wire_prompt_sha256=digest(wire_prompt), wire_schema_sha256=digest(wire_schema))
    return dict(prompt=wire_prompt, schema=wire_schema, seal=seal)


def normalize(raw, plan):
    validate_result(raw, plan['schema'])
    value = deepcopy(raw)
    if plan['seal']['adapter'] == 'evidence_partition_v1':
        value['changes'] = value.pop('supported_changes') + value.pop('unsupported_changes')
    elif plan['seal']['adapter'] == 'person_anchor_partition_v1':
        other = value.pop('other_entities')
        if any(item['ref'] in ('E0', 'E1') for item in other):
            raise ValueError('other_entities cannot contain E0/E1')
        people = []
        for name in ('e0_entities', 'e1_entities'):
            entries = value.pop(name)
            if len(entries) > plan['schema']['properties'][name]['maxItems']:
                raise ValueError(f'{name} exceeds source anchor allowance')
            people.extend(entries)
        value['entities'] = people + other
    else:
        raise ValueError('unknown wire normalization adapter')
    return value


def _sealed_read(locator):
    path = Path(locator['path'])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != locator['sha256']:
        raise ValueError('constraint grant or source seal changed')
    return read_json(path)


def _grant(policy, directory):
    grant = _sealed_read(policy['grant'])
    job_id = policy['job_id']
    selected = grant.get('jobs', {}).get(job_id)
    if (grant.get('contract') != CONTRACT or grant.get('limit') != LIMIT or selected is None
            or grant['provenance'] != provenance() or policy['requests'] != selected['requests']
            or Path(grant['output']) / 'jobs' / job_id != directory.parent.resolve()):
        raise ValueError('constraint grant binding changed')
    _sealed_read(grant['source_manifest'])
    for path, seal in selected['verified_sources'].items():
        from .wiki_selection import file_stamp
        if list(file_stamp(path)) != list(seal['stamp']):
            raise ValueError('constraint input source changed')
    return grant


def _reserve(policy, directory, key):
    grant = _grant(policy, directory)
    ledger_path = Path(grant['ledger'])
    with file_lock(ledger_path.with_suffix('.lock')):
        ledger = read_json(ledger_path, default={})
        expected = {job_id: list(job['requests'])[0] for job_id, job in grant['jobs'].items()}
        if (ledger.get('grant_sha256') != policy['grant']['sha256']
                or ledger.get('requests') != expected or set(ledger.get('used', {})) != set(expected)
                or any(type(count) is not int or not 0 <= count <= LIMIT
                       for count in ledger['used'].values())
                or ledger.get('seal') != digest({name: value for name, value in ledger.items() if name != 'seal'})):
            raise ValueError('constraint usage ledger missing or changed')
        if expected[policy['job_id']] != key:
            raise ValueError('request outside finite constraint grant')
        used = ledger['used'][policy['job_id']]
        if used >= LIMIT:
            raise ValueError('constraint request allowance exhausted')
        ledger['used'][policy['job_id']] = used + 1
        ledger['seal'] = digest({name: value for name, value in ledger.items() if name != 'seal'})
        write_json(ledger_path, ledger)
        return used + 1


def call(client, directory, key, prompt, schema, checked, *, attempt, failure):
    from .wiki_repair import request_error
    policy = read_json(directory.parent / 'manifest.json')['correction']
    if key not in policy['requests']:
        raise request_error(key, 'cache miss outside selected constraint requests; no model request made')
    selected = policy['requests'][key]
    plan = wire_plan(prompt, schema)
    if plan is None or plan['seal'] != selected['plan']:
        raise request_error(key, 'constraint request plan changed')
    _sealed_read(selected['source'])
    while True:
        try:
            policy_attempt = _reserve(policy, directory, key)
        except ValueError as exc:
            raise request_error(key, str(exc)) from exc
        attempt += 1
        raw = None
        normalized = None
        request = plan['prompt']
        if failure is not None:
            request = ('上次输出未通过机械检查；错误反馈不是证据。按新 schema 输出完整 JSON，'
                       '只使用原任务资料修正。\n' + json.dumps(dict(error=failure['error']), ensure_ascii=False)
                       + '\n' + request)
        def validated(text):
            nonlocal raw, normalized
            raw = json.loads(text)
            normalized = normalize(raw, plan)
            return checked(normalized)
        metadata = dict(request_sha256=key, attempt=attempt, retry_policy=CONTRACT,
            policy_attempt=policy_attempt, retry_budget='all_invocations_v1', wire_plan=plan['seal'],
            actual_prompt_sha256=digest(request), grant_sha256=policy['grant']['sha256'])
        try:
            with cache.validation(validated):
                text = client.run(request, plan['schema'])
            validated(text)
        except Exception as exc:
            exc.request_sha256 = key
            failure = dict(metadata, error=str(exc), result=normalized, wire_result=raw,
                           errors=[dict(attempt=attempt, error=str(exc))])
            write_json(directory.parent / 'failures' / f'{key}-{attempt}.json', failure)
            if not isinstance(exc, (ValueError, TypeError, KeyError)) or policy_attempt == LIMIT:
                raise
            continue
        write_json(directory / f'{key}.json', dict(metadata, result=normalized))
        return normalized


def prepare(source, output, request_locations, job_ids):
    from . import wiki_cached, wiki_continuation
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not request_locations or not job_ids or len(set(job_ids)) != len(job_ids)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError('selected constraint jobs and separate continuation required')
    with file_lock(source / '.lock', blocking=False), file_lock(output / '.lock', blocking=False):
        grant_path = source / f'.correction-{CONTRACT}.json'
        ledger_path = source / f'.correction-{CONTRACT}-usage.json'
        if grant_path.exists() or ledger_path.exists() or (output / 'manifest.json').exists():
            raise ValueError('constraint allowance already prepared; resume without resetting it')
        manifest = read_json(source / 'manifest.json')
        original_jobs = {job['id']: job for job in manifest['jobs']}
        previous = read_json(source / 'progress.json')
        selected = {}
        for job_id in job_ids:
            if job_id not in original_jobs or previous['jobs'].get(job_id, {}).get('stage') == 'complete':
                raise ValueError('constraint selection requires incomplete source jobs')
            receipt = read_json(Path(request_locations) / f'{job_id}.json')
            job = original_jobs[job_id]
            key, request, _, failure_path = wiki_cached.rejected_request(
                job, manifest, receipt, retry_policy='source_feedback_v2')
            plan = request.get('plan')
            if not plan:
                raise ValueError('located request has no applicable stricter wire plan')
            selected[job_id] = dict(verified_sources=receipt['verified_sources'], requests={key: dict(
                plan=plan, source=wiki_continuation.locator(failure_path))})
        def finalize(continued):
            _finalize(continued, source, output, selected, grant_path, ledger_path)
        result = wiki_continuation.prepare(source, output, _finalize=finalize)
        return dict(result, selected_jobs=len(selected), selected_requests=len(selected))


def _finalize(continued, source, output, selected, grant_path, ledger_path):
    from . import wiki_continuation
    historical = continued.pop('correction', None)
    if historical:
        continued.setdefault('correction_history', []).append(historical)
    grant = dict(contract=CONTRACT, limit=LIMIT, output=str(output), ledger=str(ledger_path),
        source_manifest=wiki_continuation.locator(source / 'manifest.json'),
        provenance=provenance(), jobs=selected)
    write_json(grant_path, grant)
    grant_seal = wiki_continuation.locator(grant_path)
    ledger = dict(grant_sha256=grant_seal['sha256'],
        requests={job_id: next(iter(job['requests'])) for job_id, job in selected.items()},
        used={job_id: 0 for job_id in selected})
    ledger['seal'] = digest(ledger)
    write_json(ledger_path, ledger)
    for job in continued['jobs']:
        historical = job.pop('correction', None)
        if historical:
            job.setdefault('correction_history', []).append(historical)
        if job['id'] in selected:
            job['correction'] = dict(contract=CONTRACT, limit=LIMIT, grant=grant_seal,
                job_id=job['id'], requests=selected[job['id']]['requests'])
    continued['correction'] = dict(contract=CONTRACT, limit=LIMIT, grant=grant_seal,
        selected_jobs=len(selected), selected_requests=len(selected), provenance=provenance())
    continued['continuation']['policy'] = ('sealed_source_constraints_v3; finite_all_invocations; '
        'unselected_cache_misses_blocked; original_validators_unchanged')
