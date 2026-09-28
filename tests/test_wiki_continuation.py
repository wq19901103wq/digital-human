"""Continuation preserves source inventory/artifacts and reuses only safe work."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.bootstrap import wiki_batch, wiki_continuation as continuation, wiki_delivery
from src.bootstrap import wiki_prompt, wiki_structured, wiki_supplement
from src.iteration.storage import LockBusy, file_lock, read_json, write_json
from test_wiki_delivery import batch_fixture


def prepared(tmp_path):
    batch, knowledge = batch_fixture(tmp_path)
    wiki = tmp_path / 'old.md'
    wiki.write_text('# synthetic old wiki\n')
    manifest = read_json(batch / 'manifest.json')
    manifest.update(messages=continuation.locator(tmp_path / 'messages.jsonl'),
        self_account='self', config={}, max_chars=120000, pipeline=wiki_batch.pipeline())
    for name in ('wiki_structured.py', 'wiki_repair.py'):
        current = manifest['pipeline'][name]
        manifest['pipeline'][name] = sorted(old for old, new in
            continuation.VALIDATION_REVISIONS[name] if new == current)[0]
    manifest['jobs'][0].update(sources=[continuation.locator(wiki)], wiki=continuation.locator(wiki))
    write_json(batch / 'manifest.json', manifest)
    return batch, knowledge


@pytest.mark.parametrize('issue', ['forward', 'entity_anchor', 'self_report'])
def test_continuation_reuses_checked_pages_and_only_retries_missing_or_unsafe(tmp_path, monkeypatch, issue):
    batch, source = prepared(tmp_path)
    manifest, state = read_json(batch / 'manifest.json'), read_json(batch / 'progress.json')
    data = read_json(source)
    if issue == 'forward':
        data['evidence']['1']['message_kind'] = 'forward'
    elif issue == 'entity_anchor':
        data['entities'][0]['evidence_refs'] = ['1']
    else:
        data['events'][0].update(mode='self_report', evidence_refs=['1'])
        data['events'][0]['participants'] = [dict(entity_id=data['subject']['entity_id'], role='发言者')]
    for name in ('bad', 'interrupted', 'pending'):
        job = dict(manifest['jobs'][0], id=name, output=str(batch / 'jobs' / name))
        manifest['jobs'].append(job)
        if name == 'pending':
            continue
        write_json(Path(job['output']) / 'cache/exact-request.json', dict(result={'cached': name}))
        if name == 'interrupted':
            state['jobs'][name] = dict(stage='running')
            continue
        wiki_structured.export(Path(job['output']) / 'content', data)
        knowledge = Path(job['output']) / 'content/knowledge.json'
        exported = wiki_prompt.export(knowledge, batch / 'prompt' / name)
        state['jobs'][name] = dict(stage='complete', knowledge=str(knowledge),
            sha256=continuation.locator(knowledge)['sha256'],
            records=exported['records'], packets=exported['packets'])
    write_json(batch / 'manifest.json', manifest)
    write_json(batch / 'progress.json', state)
    before = {p: p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    output = tmp_path / 'continued'
    result = continuation.prepare(batch, output)
    assert result == dict(total=4, reused=1, source_review=1, resumed=1, pending=1, cached_requests=2)
    new = read_json(output / 'manifest.json')
    assert new['gaps'] == manifest['gaps'] and new['jobs'][0]['id'] == 'bad'
    for job in new['jobs']:
        old = next(j for j in manifest['jobs'] if j['id'] == job['id'])
        assert all(job[k] == old[k] for k in ('account', 'binding', 'sources', 'wiki'))
    assert (output / 'jobs/bad/cache/exact-request.json').read_bytes() == (
        batch / 'jobs/bad/cache/exact-request.json').read_bytes()
    assert all(p.read_bytes() == content for p, content in before.items())
    assert wiki_delivery.inspect([output])['summary']['counts'] == {'pending': 3, 'complete': 1}

    calls = []
    def repair(wiki, messages, directory, account, self_account, config, **kwargs):
        calls.append(directory.name)
        # The batch passes the untouched raw inputs, never delivery review text.
        assert Path(wiki) == tmp_path / 'old.md' and Path(messages) == tmp_path / 'messages.jsonl'
        wiki_structured.export(directory / 'content', read_json(source))
        return dict(stage='complete')
    monkeypatch.setattr(wiki_batch.wiki_repair, 'repair', repair)
    assert wiki_batch.run(output)['stage'] == 'complete'
    assert calls == ['bad', 'interrupted', 'pending']
    wiki_batch.run(output)
    assert len(calls) == 3
    assert wiki_delivery.require_valid([output], require_complete=True)['stage'] == 'complete'
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('mutation', ['pipeline', 'raw', 'wiki', 'artifact'])
def test_continuation_rejects_nonvalidation_changes_or_corruption(tmp_path, mutation):
    batch, source = prepared(tmp_path)
    if mutation == 'pipeline':
        manifest = read_json(batch / 'manifest.json')
        manifest['pipeline']['wiki_repair.py'] = 'unregistered-change'
        write_json(batch / 'manifest.json', manifest)
    else:
        path = {'raw': tmp_path / 'messages.jsonl', 'wiki': tmp_path / 'old.md', 'artifact': source}[mutation]
        path.write_text('changed')
    with pytest.raises(ValueError):
        continuation.prepare(batch, tmp_path / 'continued')
    assert not (tmp_path / 'continued/manifest.json').exists()


def test_live_source_cannot_be_continued(tmp_path):
    batch, _ = prepared(tmp_path)
    with file_lock(batch / '.lock', blocking=False):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(continuation.prepare, batch, tmp_path / 'continued')
            with pytest.raises(LockBusy):
                future.result()


def test_supplement_continuation_binds_new_parent_and_preserves_gap_accounting(tmp_path, monkeypatch):
    batch, _ = prepared(tmp_path)
    new_parent = tmp_path / 'new-parent'
    continuation.prepare(batch, new_parent)
    supplement = tmp_path / 'supplement'
    supplement.mkdir()
    old = read_json(batch / 'manifest.json')
    index = tmp_path / 'index.sqlite3'
    index.write_bytes(b'synthetic index')
    manifest = deepcopy(old)
    manifest.update(schema='wiki_supplement_v1', pipeline=wiki_supplement.pipeline(),
        parent=continuation.locator(batch / 'manifest.json'), index=continuation.locator(index))
    manifest['jobs'][0]['id'] = 'gap'
    manifest['jobs'][0]['reuse'] = continuation.locator(batch / 'jobs/page/content/knowledge.json')
    manifest['gaps'] = []
    manifest['summary']['gaps'] = {}
    wiki_prompt.export(manifest['jobs'][0]['reuse']['path'], supplement / 'prompt/gap')
    state = read_json(batch / 'progress.json')
    state['jobs'] = {'gap': state['jobs']['page']}
    write_json(supplement / 'manifest.json', manifest)
    write_json(supplement / 'progress.json', state)
    new_supplement = tmp_path / 'new-supplement'
    with pytest.raises(ValueError, match='does not descend'):
        continuation.prepare(supplement, new_supplement, parent=batch)
    assert continuation.prepare(supplement, new_supplement, parent=new_parent)['reused'] == 1
    value = wiki_delivery.inspect([new_parent, new_supplement])
    assert value['summary']['counts'] == {'complete': 2} and value['gaps'] == []
    class Index:
        def __init__(self, *a):
            pass
        def close(self):
            pass
    monkeypatch.setattr(wiki_supplement.wiki_source_index, 'SourceIndex', Index)
    def no_generation(*a, **k):
        raise AssertionError('completed scope must not be generated again')
    monkeypatch.setattr(wiki_supplement.wiki_scope, 'generate', no_generation)
    monkeypatch.setattr(wiki_supplement.wiki_repair, 'repair', no_generation)
    assert wiki_supplement.run(new_supplement)['stage'] == 'complete'
