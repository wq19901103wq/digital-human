"""Library coverage, ambiguous names, exact source binding and crash recovery."""
import json
from pathlib import Path

import pytest

from src.bootstrap import wiki_batch as batch
from src.iteration.storage import read_json, write_json
from test_wiki_repair import row, source


def inputs(tmp_path):
    library = tmp_path / 'wiki'
    for folder in ('users', 'groups', 'topics'):
        (library / folder).mkdir(parents=True)
    for folder, name in [('users', '甲'), ('users', '同名'), ('users', 'peer2'),
                         ('users', '无来源'), ('groups', '群'), ('topics', '话题')]:
        (library / folder / f'{name}.md').write_text(f'# {name}\n- 一条旧摘要。\n')
    rows = [dict(row('private:self:peer1', 'peer1', 1000), sender='甲', chat_name='甲'),
            dict(row('private:self:peer2', 'peer2', 1001), sender='同名', chat_name='同名'),
            dict(row('private:self:peer3', 'peer3', 1002), sender='同名', chat_name='同名'),
            dict(row('private:other:peer4', 'peer4', 1003), sender='甲', chat_name='甲')]
    messages = source(tmp_path, rows)
    output = tmp_path / 'batch'
    batch.prepare(library, messages, output, 'self', {})
    return library, messages, output


def test_inventory_names_are_leads_and_ambiguous_accounts_never_merge(tmp_path):
    _, _, output = inputs(tmp_path)
    manifest = read_json(output / 'manifest.json')
    assert manifest['summary']['files'] == 6 and len(manifest['jobs']) == 2
    jobs = {j['account']: j for j in manifest['jobs']}
    assert jobs['peer1']['binding'] == 'unverified_name_candidate'
    assert jobs['peer2']['binding'] == 'account_filename'
    assert jobs['peer1']['binding_evidence'][0]['sender_id'] == 'peer1'
    ambiguous = next(g for g in manifest['gaps'] if g['reason'] == 'ambiguous_name')
    assert ambiguous['candidates'] == ['peer2', 'peer3']
    assert manifest['summary']['gaps']['requires_group_or_topic_pipeline'] == 2
    assert 'raw history' not in json.dumps(manifest)


def test_batch_retries_failures_and_resume_skips_completed_requests(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(account)
        if len(calls) == 1:
            return dict(stage='incomplete', errors=['temporary'])
        write_json(directory / 'content/knowledge.json', dict(schema='wiki_structured_v1',
            subject=dict(account=account, self_account=self_account, entity_id='target'),
            entities=[dict(id='target', account=account, kind='person', label='测试',
                           identity_status='exact_account', evidence_refs=[])],
            attributes=[], relations=[], events=[], addresses=[], evidence={}))
        return dict(stage='complete')
    monkeypatch.setattr(batch.wiki_repair, 'repair', repair)
    state = batch.run(output, max_jobs=1)
    assert state['stage'] == 'incomplete' and state['counts']['complete'] == 1
    assert len(calls) == 2 and calls[0] == calls[1]
    state = batch.run(output)
    assert state['stage'] == 'complete' and state['counts']['complete'] == 2
    assert len(calls) == 3
    batch.run(output)
    assert len(calls) == 3
    completed = next(iter(state['jobs'].values()))
    Path(completed['knowledge']).write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        batch.run(output)


def test_batch_refuses_changed_raw_source(tmp_path):
    _, messages, output = inputs(tmp_path)
    messages.write_text(messages.read_text() + '\n')
    with pytest.raises(ValueError, match='raw source changed'):
        batch.run(output)


def test_status_distinguishes_live_job_from_interrupted_saved_work(tmp_path, monkeypatch):
    _, _, output = inputs(tmp_path)
    assert batch.status(output)['counts']['pending'] == 2
    jobs = read_json(output / 'manifest.json')['jobs']
    write_json(output / 'progress.json', dict(stage='running', active_job=jobs[0]['id'],
        jobs={j['id']: dict(stage='running') for j in jobs}))
    monkeypatch.setattr(batch, 'locked', lambda _: True)
    live = batch.status(output)
    assert live['running'] and live['counts']['running'] == 1
    assert live['counts']['interrupted'] == 1
    assert live['active']['id'] == jobs[0]['id']
    monkeypatch.setattr(batch, 'locked', lambda _: False)
    stopped = batch.status(output)
    assert stopped['stage'] == 'interrupted' and stopped['active'] is None
    assert stopped['counts']['running'] == 0 and stopped['counts']['interrupted'] == 2
