"""Background feature diagnostics: scope, blind inputs, evidence and reuse."""
from __future__ import annotations

import copy
import io
import json

import pytest

from src import cache
from src.judge import background_features as features
from src.judge import feature_inspection as inspection
from src.judge import context_attribution as separate


@pytest.fixture
def case():
    return {'case_id': 'synthetic-case', 'source_span': {'chat_id': 'group:account:room'},
            'input_cutoff': {'timestamp': 100}}


@pytest.fixture
def knowledge(case):
    fact = {'id': 'work', 'predicate': 'employment_affiliation', 'subject_id': 'person:self',
            'value': '示例公司', 'scope_case_id': case['case_id'], 'observed_at': [],
            'status': '案例范围内确认', 'known_at': '2026-09-23',
            'description': 'DO_NOT_SEND_REVIEW_CONCLUSION', 'evidence_refs': ['original']}
    return {'identities': [
        {'id': 'person:self', 'name': '本人', 'status': 'source_account_bound',
         'description': 'DO_NOT_SEND_DESCRIPTION'},
        {'id': 'person:friend', 'name': '朋友', 'status': 'user_confirmed',
         'aliases': ['某哥'], 'decision': {'scope': {'chat_id': 'group:account:room'}}},
        {'id': 'person:pending', 'name': '待定身份', 'status': 'pending', 'aliases': ['另一人']},
    ], 'fact_claims': [fact], 'evidence': {'original': {'quote': 'DO_NOT_SEND_RAW_TEXT'}}}


@pytest.fixture
def blind():
    return {'relationship': 'group', 'context_original': ['问题正文'],
            'option_A': ['并不会'], 'option_B': ['问朋友'],
            'human_option': 'A', 'candidate_option': 'B', 'label': 'SECRET_LABEL'}


def response():
    options = {option: {'attribution_checks': [],
                     'features': {name: {'state': 'unknown', 'evidence_refs': [],
                                        'reason': '材料不足'} for name in features.FEATURES},
                     'routing': {name: '未知' for name in (
                         'question_topic', 'question_target', 'delegated_to', 'background_owner')}}
            for option in ('option_A', 'option_B')}
    return {'context_attribution': [], **options}


def attribution(role, speaker, statement, ref):
    return {'speaker_role': role, 'speaker': speaker, 'subject': speaker,
            'target': '未知', 'proposition': statement, 'attitude': 'affirmed',
            'statement': statement, 'evidence_refs': [ref]}


def test_projection_removes_raw_evidence_and_review_conclusions(case, knowledge):
    projected = features.diagnostic_projection(knowledge, case)
    assert projected['case_correction_refs'] == ['fact:1']
    assert not projected['historical_admission']
    assert 'DO_NOT_SEND' not in json.dumps(projected)
    assert projected['source_refs']['fact:1'] == {'claim_id': 'work', 'evidence_refs': ['original']}
    assert projected['payload']['facts'][0]['value'] == '示例公司'


@pytest.mark.parametrize('change', [
    {'scope_case_id': 'different'}, {'scope_chat_id': 'different'},
    {'observed_at': [90, 101]}, {'status': '撤下无依据的职业归属'},
    {'predicate': 'unsupported_occupation'}, {'scope_case_id': None},
])
def test_projection_rejects_out_of_scope_or_unsupported_claims(case, knowledge, change):
    knowledge['fact_claims'][0].update(change)
    assert not features.diagnostic_projection(knowledge, case)['payload']['facts']


def test_confirmed_aliases_are_chat_scoped_and_pending_identity_not_merged(case, knowledge):
    base = knowledge['fact_claims'][0]
    knowledge['fact_claims'] += [dict(base, id='friend', subject_id='person:friend'),
                               dict(base, id='pending', subject_id='person:pending')]
    identities = features.diagnostic_projection(knowledge, case)['payload']['identities']
    assert identities[-1]['aliases'] == ['某哥']
    assert len(identities) == 2
    knowledge['identities'][1]['decision']['scope']['chat_id'] = 'other-group'
    assert 'aliases' not in features.diagnostic_projection(knowledge, case)['payload']['identities'][-1]


def test_blind_payload_and_only_categories_reach_scoring(blind):
    payload = features.build_input(blind, {'identities': [], 'facts': []})
    prompt = features.prompt_for(payload)
    assert 'SECRET_LABEL' not in prompt and 'human_option' not in prompt
    parsed = features.parse_response(json.dumps(response()), payload)
    vector = features.categorical_features(parsed)
    assert set(vector['option_A']) == set(features.FEATURES)
    assert set(vector['option_A'].values()) == {'unknown'}


def test_message_roles_and_quoted_first_person_do_not_get_merged(blind):
    blind['context_original'] = [
        '<conversation_history><message role="other">同事：我会查</message>'
        '<message role="self">我：好 [引用 同事：我会查]</message></conversation_history>']
    payload = features.build_input(blind, {'facts': []})
    assert [row['role'] for row in payload['context']] == ['other', 'self']
    value = response()
    value['context_attribution'] = [attribution('other', '同事', '承诺查询', 'context:0:0')]
    parsed = features.parse_response(json.dumps(value), payload)
    assert 'context_attribution' not in features.categorical_features(parsed)
    value['context_attribution'][0]['speaker_role'] = 'self'
    with pytest.raises(ValueError, match='同角色'):
        features.parse_response(json.dumps(value), payload)
    value['context_attribution'][0]['evidence_refs'] = ['option_A']
    with pytest.raises(ValueError, match='候选回复'):
        features.parse_response(json.dumps(value), payload)


def test_schema_separates_senders_and_requires_judgment_evidence(blind):
    blind['context_original'] = [
        '<conversation_history><message role="other">同事：我会查</message>'
        '<message role="self">我：好 [引用 同事：我会查]</message></conversation_history>']
    payload = features.build_input(blind, {'facts': []})
    schema = features.response_schema(payload)['properties']
    alternatives = schema['context_attribution']['items']['anyOf']
    refs_by_role = {a['properties']['speaker_role']['enum'][0]:
                   a['properties']['evidence_refs']['items']['enum'] for a in alternatives}
    assert refs_by_role == {'self': ['context:0:1'], 'other': ['context:0:0']}
    observation = schema['option_A']['properties']['features']['properties']['entity_reference']
    judged, unjudged = observation['anyOf']
    assert judged['properties']['evidence_refs']['minItems'] == 1
    assert unjudged['properties']['state']['enum'] == ['not_applicable', 'unknown']
    value = response()
    value['context_attribution'] = [attribution(
        'self', '我', '本人引用同事承诺查询的消息并确认收到', 'context:0:1')]
    assert features.parse_response(json.dumps(value), payload) == value
    value['context_attribution'][0]['evidence_refs'].append('context:0:0')
    with pytest.raises(ValueError, match='同角色'):
        features.parse_response(json.dumps(value), payload)


def test_attribution_cannot_merge_two_other_speakers(blind):
    blind['context_original'] = [
        '<messages_to_reply><message role="other">甲：我去查</message>'
        '<message role="other">乙：我去查</message></messages_to_reply>']
    payload = features.build_input(blind, {'facts': []})
    value = response()
    item = attribution('other', '甲', '承诺查询', 'context:0:0')
    value['context_attribution'] = [item]
    assert features.parse_response(json.dumps(value), payload) == value
    item['evidence_refs'].append('context:0:1')
    with pytest.raises(ValueError, match='只引用一条'):
        features.parse_response(json.dumps(value), payload)


@pytest.mark.parametrize('bad_refs', [[], ['option_A'], ['fact:1'], ['context:missing']])
def test_ownership_shift_requires_context_not_reply_or_wiki(blind, bad_refs):
    payload = features.build_input(blind, {'facts': [{'ref': 'fact:1'}]})
    value = response()
    value['option_A']['attribution_checks'] = [{
        'claim': '接续他人的经历', 'claim_mode': 'self_continuation',
        'context_owner': 'other', 'context_refs': bad_refs,
        'alignment': 'owner_shift', 'reason': '引用必须定位上文'}]
    with pytest.raises(ValueError, match='归属'):
        features.parse_response(json.dumps(value), payload)


def test_subject_and_stance_are_distinct_from_source_sender(blind):
    blind['context_original'] = [
        '<messages_to_reply><message role="self">我：我不同意同事的说法</message>'
        '</messages_to_reply>']
    payload = features.build_input(blind, {'facts': []})
    value = response()
    value['context_attribution'] = [{**attribution('self', '本人', '本人反对同事观点', 'context:0:0'),
        'subject': '同事', 'target': '同事观点', 'proposition': '同事的说法', 'attitude': 'rejected'}]
    value['option_A']['attribution_checks'] = [{
        'claim': '引用同事观点', 'claim_mode': 'quotation', 'context_owner': 'other',
        'context_refs': ['context:0:0'], 'alignment': 'consistent', 'reason': '本人曾引用同事观点'}]
    parsed = features.parse_response(json.dumps(value), payload)
    assert parsed == value
    assert '同事' not in json.dumps(features.categorical_features(parsed), ensure_ascii=False)


@pytest.mark.parametrize('bad_refs', [['fact:missing'], ['option_B'], []])
def test_judgment_requires_available_own_option_evidence(blind, bad_refs):
    payload = features.build_input(blind, {'facts': []})
    value = response()
    value['option_A']['features']['delegation_topic_fit'].update(
        state='contradicted', evidence_refs=bad_refs)
    with pytest.raises(ValueError):
        features.parse_response(json.dumps(value), payload)


def test_saved_case_uses_latest_success_not_failed_retry(tmp_path, monkeypatch, case, blind):
    experiment = tmp_path / 'experiments' / 'source'
    (experiment / 'traces').mkdir(parents=True)
    for ref in ('a' * 32, 'b' * 32):
        trace = {'case': case, 'trace_ref': ref, 'versions': {}, 'operations': [
            {'kind': 'judge', 'branch': 'candidate', 'round': 0, 'status': 'ok',
             'events': [{'kind': 'blind_mapping', 'data': {'blind_case': blind,
                        'human_option': 'A', 'candidate_option': 'B'}}]}]}
        (experiment / 'traces' / f'{ref}.json').write_text(json.dumps(trace))
    rows = [dict(case_id=case['case_id'], status=status, trace_ref=ref)
            for status, ref in [('ok', 'a' * 32), ('ok', 'b' * 32), ('failed', 'c' * 32)]]
    # Emulate a legacy read stream; current writers correctly forbid replacing successes.
    original_open = type(experiment).open

    def open_fixture(path, *args, **kwargs):
        if path == experiment / 'cases.jsonl':
            return io.StringIO('\n'.join(json.dumps(row) for row in rows))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(experiment), 'open', open_fixture)
    assert inspection.saved_case(experiment, case['case_id'], 'candidate', 0)['source_trace'] == 'b' * 32


def test_diagnostic_resume_reuses_requests_and_share_change_invalidates(
        tmp_path, monkeypatch, case, knowledge, blind):
    monkeypatch.setattr(inspection.versions, 'PRIVATE', tmp_path / 'instance')
    source = tmp_path / 'share'
    (source / 'content').mkdir(parents=True)
    wiki = source / 'content' / 'knowledge.json'
    wiki.write_text(json.dumps(knowledge))
    monkeypatch.setattr(inspection.shares, 'load', lambda ref: {
        'dir': source, 'manifest': {'sha256': cache.digest(json.loads(wiki.read_text()))}})
    frozen_config = {'llm': {'model': 'synthetic-test', 'reasoning_effort': 'high'},
                     'feature_llm': {'reasoning_effort': 'low'}}
    original_config = copy.deepcopy(frozen_config)
    monkeypatch.setattr(inspection.versions, 'judge_dir', lambda ref: {'config': frozen_config})
    monkeypatch.setattr(inspection, 'saved_case', lambda *args: {
        'case': case, 'mapping': {'blind_case': blind, 'human_option': 'A', 'candidate_option': 'B'},
        'old_features': None, 'versions': {}, 'source_trace': 'a' * 32})

    class Client:
        calls = 0

        def cache_identity(self):
            return {'model': 'synthetic-test'}

        def run(self, prompt, schema):
            def produce():
                self.calls += 1
                return json.dumps(response())
            return cache.memo('llm_request', {'prompt': prompt, 'schema': schema},
                              produce, valid=cache.response_valid)

    client = Client()
    output = tmp_path / 'instance' / 'analysis' / 'diagnostic'
    args = (tmp_path / 'existing', case['case_id'], 's-0001', 'j-0001', output)
    report = inspection.inspect(*args, client=client, conditions=['context_only', 'with_background'])
    assert report['status'] == 'ok' and not report['formal_accuracy_evidence']
    assert client.calls == 2
    inspection.inspect(*args, client=client, conditions=['context_only', 'with_background'])
    assert client.calls == 2
    changed = copy.deepcopy(knowledge)
    changed['fact_claims'][0]['value'] = '另一家公司'
    wiki.write_text(json.dumps(changed))
    inspection.inspect(*args, client=client, conditions=['context_only', 'with_background'])
    assert client.calls == 3
    captured = []
    monkeypatch.setattr(inspection, 'CodexJudgeClient', lambda config: captured.append(config) or client)
    overridden = inspection.inspect(*args, conditions=['context_only'], reasoning_effort='medium')
    assert captured == [{'model': 'synthetic-test', 'reasoning_effort': 'medium'}]
    assert overridden['diagnostic_overrides'] == {'reasoning_effort': 'medium'}
    assert frozen_config == original_config
    assert not (output / 'cases.jsonl').exists()

    class SeparateClient(Client):
        context_calls = 0
        reply_calls = 0

        def run(self, prompt, schema):
            context_only = set(schema['properties']) == {'context_attribution'}

            def produce():
                if context_only:
                    self.context_calls += 1
                    assert '并不会' not in prompt and '问朋友' not in prompt
                    assert '示例公司' not in prompt and 'SECRET_LABEL' not in prompt
                    return json.dumps({'context_attribution': []})
                self.reply_calls += 1
                if self.reply_calls == 1:
                    raise RuntimeError('synthetic interrupted reply stage')
                return json.dumps({key: value for key, value in response().items()
                                   if key != 'context_attribution'})
            return cache.memo('llm_request', {'prompt': prompt, 'schema': schema},
                              produce, valid=cache.response_valid)

    separate_client = SeparateClient()
    staged_args = dict(client=separate_client, conditions=['context_only'],
                       extraction_mode='separate_context')
    with pytest.raises(RuntimeError, match='interrupted'):
        inspection.inspect(*args, **staged_args)
    partial = json.loads((output / 'report.json').read_text())
    assert partial['status'] == 'incomplete'
    assert partial['context_extractions']['context_only']['context_attribution'] == []
    completed = inspection.inspect(*args, **staged_args)
    assert completed['status'] == 'ok'
    assert completed['extraction_protocol'] == separate.VERSION
    assert (separate_client.context_calls, separate_client.reply_calls) == (1, 2)
    # Changed replies reuse the isolated context call, not the reply check.
    blind['option_A'] = ['全新回复']
    inspection.inspect(*args, **staged_args)
    assert (separate_client.context_calls, separate_client.reply_calls) == (1, 3)


def test_separate_context_input_excludes_replies_background_and_labels(blind):
    payload = features.build_input(blind, {'facts': [{'ref': 'fact:1', 'value': 'SECRET_WIKI'}]})
    payload['label'] = 'SECRET_LABEL'
    isolated = separate.build_input(payload)
    prompt = separate.prompt_for(isolated)
    assert set(isolated) == {'schema', 'relationship', 'context'}
    assert '并不会' not in prompt and '问朋友' not in prompt and 'SECRET' not in prompt
    value = {'context_attribution': [attribution('unknown', '未知', '问题', 'context:0')]}
    assert separate.parse_response(json.dumps(value), isolated) == value
    value['context_attribution'][0]['evidence_refs'] = ['option_A']
    with pytest.raises(ValueError, match='候选回复'):
        separate.parse_response(json.dumps(value), isolated)


def test_reply_stage_cannot_rewrite_context_or_add_foreign_evidence(blind):
    payload = features.build_input(blind, {'facts': []})
    attribution_value = {'context_attribution': [attribution('unknown', '未知', '问题', 'context:0')]}
    value = response()
    with pytest.raises(ValueError, match='不得改写'):
        separate.parse_reply_response(json.dumps(value), payload, attribution_value)
    del value['context_attribution']
    parsed = separate.parse_reply_response(json.dumps(value), payload, attribution_value)
    assert parsed['context_attribution'] == attribution_value['context_attribution']
    value['option_A']['features']['self_history_consistency'].update(
        state='unsupported', evidence_refs=['context:missing'])
    with pytest.raises(ValueError, match='引用非法'):
        separate.parse_reply_response(json.dumps(value), payload, attribution_value)
