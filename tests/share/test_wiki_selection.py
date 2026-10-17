import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sqlite3

import pytest

from src.share.wiki import wiki_batch as batch
from src.share.wiki import wiki_mentions as mentions
from src.share.wiki import wiki_repair as repair
from src.share.wiki import wiki_selection as selection
from test_wiki_repair import Client, row, source


def index_for(path, terms=(), **kwargs):
    return selection.SelectionIndex(dict(path=str(path), sha256=selection.file_digest(path)),
        'self', sorted(set(mentions.normalize(term) for term in terms)), path.parent / 'index',
        **kwargs).prepare()


def compare(index, account='person', wiki='', title='阿林', radius=8, gap=600):
    args = (index.source, account, 'self', radius, gap)
    expected = repair.select_history(*args)
    assert index.select_history(*args) == expected
    args = (index.source, expected[0], account, 'self', wiki, title, radius, gap)
    assert index.select_mentions(*args) == mentions.select_mentions(*args)


@pytest.mark.parametrize('radius,gap', [(0, 0), (1, 600), (8, 600), (20, 10000)])
def test_selections_preserve_rows_hashes_sort_boundaries_and_reports(tmp_path, radius, gap):
    records = [row('private:self:person', 'self', 1700000000200, 'later'),
        row('private:self:person', 'person', 1700000000, 'first'),
        row('group:self:room', 'other', 1700000600001, '阿林 Alice'),
        row('group:self:room', 'person', 1700000000),
        row('group:self:room', 'other', 1700000600000, 'edge'),
        row('group:self:room', 'person', 1700000000000, 'same time'),
        row('group:another:room', 'person', 1700000000, '阿林'),
        row('private:self:other', 'other', 1700000000, '@ＡＬＩＣＥ 阿林'),
        row('private:self:other', 'self', 1700000001, 'context'),
        row('group:self:unrelated', 'other', 1700000000, 'xalicey'),
        row('group:self:homonym', 'homonym', 1700000000),
        row('system:self:feed', 'homonym', 1700000000),
        row('malformed', 'person', 1700000000, '阿林')]
    records[1]['sender'] = '阿林'
    records[10]['sender'] = '阿林'
    records[11]['sender'] = 'Alice'
    records[12]['chat_type'] = 'private'
    path = source(tmp_path, records)
    index = index_for(path, ['阿林', 'Alice', 'person'])
    compare(index, wiki='- 别名：Alice', radius=radius, gap=gap)
    with sqlite3.connect(index.path) as db:
        assert {entry[1] for entry in db.execute('PRAGMA table_info(messages)')} == {
            'line', 'offset', 'size', 'chat_id', 'chat_type', 'peer', 'sender_id', 'stamp', 'seq_all', 'seq_group'}
        assert db.execute('SELECT count(*) FROM messages').fetchone()[0] == 10


def test_display_name_term_order_follows_base_then_raw_source_not_alphabet(tmp_path):
    records = [row('group:self:far', 'person', 1000),
        row('private:self:person', 'person', 1000),
        row('group:self:mentions', 'other', 1000, 'Zulu Alpha Base 阿林'),
        row('group:self:later', 'person', 1000)]
    for record, name in zip(records, ['Zulu', 'Base', 'other', 'Alpha']):
        record['sender'] = name
    path = source(tmp_path, records)
    index = index_for(path, ['阿林', 'person'])
    base, _ = repair.select_history(path, 'person', 'self')
    base = [entry for entry in base if entry['chat_type'] == 'private']
    args = (path, base, 'person', 'self', '', '阿林')
    expected = mentions.select_mentions(*args)
    assert expected[1]['anchors'][0]['terms'] == ['阿林', 'base', 'zulu', 'alpha']
    assert index.select_mentions(*args) == expected


@pytest.mark.parametrize('text', ['ＡＬＩＣＥ 阿林老师 阿林', '_alice alice_ xalice alice2',
    '@Alice ALICE foo alice! 阿林老师', 'éAlice林 aliceé 阿林林', 'aaa abcd ab abc a-b (阿林)'])
def test_matcher_equals_individual_legacy_regex_search(text):
    terms = ['alice', '阿林', '阿林老师', '林老师', 'aa', 'aaa', 'ab', 'abc', 'abcd', 'a-b', '(阿林)']
    message = mentions.normalize(text)
    expected = {term for term in terms if re.search(
        (r'(?<![a-z0-9_])' if term[0].isascii() and term[0].isalnum() else '') + re.escape(term) +
        (r'(?![a-z0-9_])' if term[-1].isascii() and term[-1].isalnum() else ''), message)}
    assert selection.MentionMatcher(terms).find(message) == expected


def test_mixed_chat_types_have_same_legacy_history_window_semantics(tmp_path):
    records = [row('group:self:person', 'person', 1000), row('group:self:person', 'other', 1001),
        row('group:self:room', 'other', 1000), row('group:self:room', 'person', 1001),
        row('group:self:room', 'other', 1002)]
    records[0]['chat_type'] = 'private'
    records[2]['chat_type'] = 'private'
    path = source(tmp_path, records)
    compare(index_for(path, ['阿林', 'person']), radius=1)


@pytest.mark.parametrize('records', [[], [row('group:other:room', 'person', 1000)],
    [row('private:self:person', 'self', 1000)]])
def test_missing_subject_history_fails_like_reference(tmp_path, records):
    path = source(tmp_path, records)
    index = index_for(path)
    for function in (repair.select_history, index.select_history):
        with pytest.raises(ValueError, match='no raw history'):
            function(path, 'person', 'self')


def test_cache_reuse_and_concurrent_readers_never_rescan_jsonl(tmp_path, monkeypatch):
    path = source(tmp_path, [row('private:self:person', 'person', 1000),
                            row('group:self:room', 'other', 1000, '阿林')])
    index = index_for(path, ['阿林', 'person'])
    expected = index.select_history(path, 'person', 'self')
    def no_build(self):
        raise AssertionError('warm index rebuilt')
    monkeypatch.setattr(selection.SelectionIndex, 'build', no_build)
    reused = index_for(path, ['阿林', 'person'])
    assert reused.stats['reused']
    original_open = Path.open
    class SeekOnly:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def seek(self, *args):
            return self.stream.seek(*args)
        def read(self, size):
            assert size > 0
            return self.stream.read(size)
        def __iter__(self):
            raise AssertionError('per-object full source scan')
    def seek_only_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return SeekOnly(stream) if self.resolve() == path.resolve() else stream
    monkeypatch.setattr(Path, 'open', seek_only_open)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: reused.select_history(path, 'person', 'self'), range(12))) == [expected] * 12
        list(pool.map(lambda _: reused.select_mentions(path, expected[0], 'person', 'self', '', '阿林'), range(12)))


def test_cache_bytes_and_source_changes_fail_closed(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    index = index_for(path)
    with index.path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='index changed'):
        index.select_history(path, 'person', 'self')
    with pytest.raises(ValueError, match='cache binding or bytes changed'):
        index_for(path)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='raw source changed'):
        index.select_history(path, 'person', 'self')
    with pytest.raises(ValueError, match='raw source changed'):
        selection.SelectionIndex(dict(path=str(path), sha256=index.source_sha256),
            'self', [], path.parent / 'index').prepare()


def test_verified_source_stamp_cannot_hide_hashing_race(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    stamp = selection.file_stamp(path)
    source_hash = selection.file_digest(path)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='raw source changed after verification'):
        selection.SelectionIndex(dict(path=str(path), sha256=source_hash), 'self', [],
            path.parent / 'index', verified_source_stamp=stamp)


def test_warm_index_change_after_hash_cannot_be_adopted_as_trusted_stamp(tmp_path, monkeypatch):
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    index = index_for(path)
    original_digest = selection.file_digest
    def change_after_digest(file):
        value = original_digest(file)
        if Path(file) == index.path:
            with index.path.open('ab') as stream:
                stream.write(b'changed-after-hash')
        return value
    monkeypatch.setattr(selection, 'file_digest', change_after_digest)
    with pytest.raises(ValueError, match='index changed'):
        index_for(path)


def test_interrupted_uncertified_build_is_rebuilt_not_reused(tmp_path, monkeypatch):
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    original_write = selection.write_json
    def interrupted_write(*args, **kwargs):
        raise RuntimeError('interrupted before certification')
    monkeypatch.setattr(selection, 'write_json', interrupted_write)
    with pytest.raises(RuntimeError, match='before certification'):
        index_for(path)
    databases = list((tmp_path / 'index').glob('*.sqlite3'))
    assert len(databases) == 1 and not databases[0].with_suffix('.json').exists()
    with databases[0].open('ab') as stream:
        stream.write(b'uncertified bytes must not be reused')
    monkeypatch.setattr(selection, 'write_json', original_write)
    index = index_for(path)
    assert not index.stats['reused']
    assert index.path == databases[0]
    assert index.path.with_suffix('.json').exists()
    compare(index, title='person')


def test_unbound_terms_fail_closed(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    index = index_for(path)
    base, _ = index.select_history(path, 'person', 'self')
    with pytest.raises(ValueError, match='not bound'):
        index.select_mentions(path, base, 'person', 'self', '', 'unbound alias')


def test_scoped_override_restores_and_preserves_exact_generation_cache(tmp_path):
    path = source(tmp_path, [row('private:self:person', 'person', 1000, 'raw account history')])
    wiki = tmp_path / 'person.md'
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    output = tmp_path / 'output'
    client = Client()
    repair.repair(wiki, path, output, 'person', 'self', {}, client=client, workers=2)
    prompts = list(client.prompts)
    before = batch.pipeline()
    history, recall = repair.select_history, mentions.select_mentions
    index = index_for(path, ['person'])
    with selection.accelerated(index):
        assert batch.pipeline() == before
        result = repair.repair(wiki, path, output, 'person', 'self', {}, client=client, workers=2)
    assert result['stage'] == 'complete'
    assert client.prompts == prompts
    assert repair.select_history is history and mentions.select_mentions is recall
    with pytest.raises(RuntimeError, match='exit'):
        with selection.accelerated(index):
            raise RuntimeError('exit')
    assert repair.select_history is history and mentions.select_mentions is recall


def test_offline_check_does_not_overwrite_live_runtime_receipt(tmp_path):
    from scripts.check import main
    path = source(tmp_path, [row('private:self:person', 'person', 1000)])
    wiki = tmp_path / 'legacy' / 'users' / 'person.md'
    wiki.parent.mkdir(parents=True)
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    output = tmp_path / 'batch'
    batch.prepare(wiki.parent.parent, path, output, 'self', {})
    receipt = output / 'source_selection_runtime.json'
    receipt.write_text('live-runtime-evidence')
    report = tmp_path / 'check.json'
    assert main(['--output', str(report), 'wiki-selection', '--batch', str(output)]) == 0
    assert receipt.read_text() == 'live-runtime-evidence'
    value = json.loads(report.read_text())
    assert value['status'] == 'passed' and value['model_requests_prohibited']
    assert len(value['checks'][0]['detail']['jobs']) == 1


def test_batch_cache_is_shared_outside_sealed_source_directory(tmp_path):
    sealed = tmp_path / 'sealed'
    sealed.mkdir()
    path = source(sealed, [row('private:self:person', 'person', 1000)])
    wiki = tmp_path / 'legacy' / 'users' / 'person.md'
    wiki.parent.mkdir(parents=True)
    wiki.write_text('# Person\n## 基本信息\n- legacy claim\n')
    output = tmp_path / 'background' / 'batch'
    batch.prepare(wiki.parent.parent, path, output, 'self', {})
    manifest = json.loads((output / 'manifest.json').read_text())
    sealed.chmod(0o555)
    try:
        index = selection.for_batch(manifest, output)
        assert index.path.parent == output.parent / '.cache/wiki-selection'
        assert not (sealed / '.cache').exists()
        receipt = (output / 'source_selection_runtime.json').read_bytes()
        checked = selection.check_batch(output)
        assert checked['equivalent'] and checked['preparation']['reused']
        assert (output / 'source_selection_runtime.json').read_bytes() == receipt
        sibling = selection.for_batch(manifest, output.parent / 'other-batch')
        assert sibling.stats['reused'] and sibling.path == index.path
    finally:
        sealed.chmod(0o755)
