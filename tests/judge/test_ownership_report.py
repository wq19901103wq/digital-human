import json

import pytest

from src.judge.ownership_report import build_report


def observation(human, base, final, a=False, b=False, round_index=0):
    return dict(kind='judge', branch='candidate', round=round_index, status='ok',
                result={'identified_ai': ('A' if final >= .5 else 'B') == human}, events=[
                    {'kind': 'blind_mapping', 'data': {'human_option': human}},
                    {'kind': 'ownership_features', 'data': {'decision': {
                        'policy': 'test', 'base_probability_a': base, 'probability_a': final,
                        'triggered': a != b, 'flags': {
                            'option_A': [{}] if a else [], 'option_B': [{}] if b else []}}}}])


def save_case(path, cid, ref, operations, baseline, candidate):
    (path / 'traces' / (ref + '.json')).write_text(json.dumps(dict(
        case_id=cid, status='ok', operations=operations)))
    return dict(case_id=cid, status='ok', baseline_correct=baseline,
                candidate_correct=candidate, trace_ref=ref)


@pytest.mark.parametrize('family', ['ownership_features', 'contribution_features'])
def test_report_separates_rounds_human_flags_and_actual_harm(tmp_path, family):
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'spec.json').write_text('{"kind":"judge_eval"}')
    rows = [dict(case_id='help', status='failed')]
    rows.append(save_case(tmp_path, 'help', 'a' * 32,
        [observation('B', .8, 0, a=True), observation('A', .8, .8, round_index=1)], False, True))
    rows.append(save_case(tmp_path, 'harm', 'b' * 32,
        [observation('A', .5, 0, a=True)], True, False))
    rows.append(save_case(tmp_path, 'dual', 'c' * 32,
        [observation('B', .2, .2, a=True, b=True)], True, True))
    rows.append(dict(case_id='missing', status='ok', baseline_correct=True,
                     candidate_correct=True, trace_ref='d' * 32))
    rows.append(dict(case_id='still_failed', status='failed'))
    (tmp_path / 'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    for path in (tmp_path / 'traces').glob('*.json'):
        path.write_text(path.read_text().replace('ownership_features', family))
    report = build_report(tmp_path, family)
    assert report['retries'] == 1
    assert report['successful_cases'] == 4
    assert report['main']['observed'] == 3 and report['main']['missing'] == 1
    counts = report['main']['counts']
    assert counts['human_flagged'] == counts['generated_flagged'] == 2
    assert counts['both_flagged'] == counts['helped'] == counts['harmed'] == 1
    assert report['main']['net_corrected'] == 0
    assert report['supplemental']['observed'] == 1
    assert report['paired_first_round']['wins'] == report['paired_first_round']['losses'] == 1
    assert len(report['evidence_gaps']) == 1


def test_missing_or_inconsistent_evidence_is_not_counted_as_no_flags(tmp_path):
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'spec.json').write_text('{"kind":"judge_eval"}')
    op = observation('A', .8, .8)
    op['result']['identified_ai'] = False
    row = save_case(tmp_path, 'bad', 'e' * 32, [op], True, True)
    (tmp_path / 'cases.jsonl').write_text(json.dumps(row))
    report = build_report(tmp_path)
    assert report['main']['observed'] == 0
    assert report['main']['rates']['human_flagged'] is None
    assert report['evidence_gaps'][0]['case_id'] == 'bad'


def test_feature_breakdown_deduplicates_flags_and_does_not_double_count_overall_help(tmp_path):
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'spec.json').write_text('{"kind":"judge_eval"}')
    op = observation('B', .8, 0, a=True)
    event = op['events'][1]
    event['kind'] = 'contribution_features'
    event['data']['decision']['flags']['option_A'] = [
        {'feature': 'place_event_binding'}, {'feature': 'place_event_binding'},
        {'feature': 'reply_target_alignment'}]
    row = save_case(tmp_path, 'help', 'f' * 32, [op], False, True)
    (tmp_path / 'cases.jsonl').write_text(json.dumps(row))
    report = build_report(tmp_path, 'contribution_features')
    assert report['main']['counts']['helped'] == 1
    for feature in ('place_event_binding', 'reply_target_alignment'):
        assert report['main']['feature_breakdown'][feature] == dict(
            human_flagged=0, generated_flagged=1, helped_when_present=1, harmed_when_present=0)
