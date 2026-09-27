"""Shared knowledge snapshots and independent consumer pins; no model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.config import ConfigError
from src.iteration import shares, versions


@pytest.fixture
def registry(tmp_path, monkeypatch):
    private = tmp_path / 'instance'
    for name, path in {
        'PRIVATE': private, 'DATA_ROOT': private / 'data',
        'GEN_ROOT': private / 'generators', 'JUDGE_ROOT': private / 'judges',
        'POINTERS_PATH': private / 'pointers.json',
    }.items():
        monkeypatch.setattr(versions, name, path)
    source = tmp_path / 'gen-source'
    (source / 'scenarios').mkdir(parents=True)
    (source / 'persona.md').write_text('synthetic persona')
    (source / 'scenarios' / 'friend.md').write_text('synthetic scenario')
    (source / 'learning.json').write_text('{"source": "synthetic"}')
    gen = versions.create_generator_version({'llm': {'model': 'synthetic'}},
        'd-0001', source_dir=source)
    model = b'synthetic model'
    judge = versions.create_judge_version(
        {'mode': 'pairwise_llm', 'assets': {'model.bin': hashlib.sha256(model).hexdigest()}},
        {'source': 'synthetic'}, assets={'prompt.md': b'synthetic prompt',
            'model.bin': model, 'learning.json': b'{"source": "synthetic"}',
            'training-source.json': b'{"evidence": "synthetic"}'})
    versions.save_pointers({'data': 'd-0001', 'production_gen': gen, 'iteration_gen': gen,
                            'production_judge': judge, 'iteration_judge': judge})
    wiki = tmp_path / 'wiki'
    (wiki / 'people').mkdir(parents=True)
    (wiki / 'people' / 'one.md').write_text('source-backed synthetic fact')
    return wiki


def test_freeze_preserves_old_content_and_deduplicates(registry):
    first = shares.create(registry)
    assert first == 's-0001'
    assert shares.create(registry) == first
    info = shares.load(first)
    frozen = info['dir'] / 'content' / 'people' / 'one.md'
    assert frozen.stat().st_mode & 0o222 == 0
    (registry / 'people' / 'one.md').write_text('corrected synthetic fact')
    second = shares.create(registry)
    assert second == 's-0002'
    assert frozen.read_text() == 'source-backed synthetic fact'
    assert shares.load(second)['manifest']['sha256'] != info['manifest']['sha256']
    assert shares.create(info['dir'] / 'content') == first


def test_gen_and_judge_pin_independently_without_changing_pointers(registry):
    pointers = versions.POINTERS_PATH.read_bytes()
    original_gen = versions.version_payload(versions.generator_dir('g-0001'))
    original_judge = versions.version_payload(versions.judge_dir('j-0001')['dir'])
    first = shares.create(registry)
    gen_first = shares.bind_model('gen', 'g-0001', first)
    judge_first = shares.bind_model('judge', 'j-0001', first)
    (registry / 'people' / 'one.md').write_text('corrected fact')
    second = shares.create(registry)
    gen_second = shares.bind_model('gen', gen_first, second)
    assert gen_first != gen_second != 'g-0001'
    assert shares.bind_model('gen', gen_second, second) == gen_second
    assert shares.bind_model('judge', judge_first, first) == judge_first
    assert versions.load_generator(gen_second)['config']['share_ref'] == second
    assert versions.judge_dir(judge_first)['config']['share_ref'] == first
    assert versions.POINTERS_PATH.read_bytes() == pointers
    assert versions.version_payload(versions.generator_dir('g-0001')) == original_gen
    assert versions.version_payload(versions.judge_dir('j-0001')['dir']) == original_judge
    for name in ('prompt.md', 'model.bin', 'learning.json', 'training-source.json', 'meta.json'):
        assert (versions.JUDGE_ROOT / judge_first / name).read_bytes() == (
            versions.JUDGE_ROOT / 'j-0001' / name).read_bytes()
    for name in ('persona.md', 'scenarios/friend.md', 'learning.json'):
        assert (versions.GEN_ROOT / gen_second / name).read_bytes() == (
            versions.GEN_ROOT / 'g-0001' / name).read_bytes()
    payload = versions.version_payload(versions.generator_dir(gen_second))
    assert payload['config']['share_sha256'] == shares.binding(second)['share_sha256']
    rows = {r['id']: r for r in shares.inventory()}
    assert rows[first]['consumers'] == {'gen': [gen_first], 'judge': [judge_first]}
    assert rows[second]['consumers'] == {'gen': [gen_second], 'judge': []}


@pytest.mark.parametrize('corruption', ['content', 'manifest', 'missing', 'extra'])
def test_snapshot_tampering_rejected_at_consumer_load(registry, corruption):
    ref = shares.create(registry)
    gen = shares.bind_model('gen', 'g-0001', ref)
    judge = shares.bind_model('judge', 'j-0001', ref)
    directory = shares.load(ref)['dir']
    fact = directory / 'content' / 'people' / 'one.md'
    if corruption == 'content':
        fact.chmod(0o644)
        fact.write_text('unrecorded change')
    elif corruption == 'manifest':
        path = directory / 'manifest.json'
        value = json.loads(path.read_text())
        value['sha256'] = 'invalid'
        path.chmod(0o644)
        path.write_text(json.dumps(value))
    elif corruption == 'missing':
        fact.parent.chmod(0o755)
        fact.unlink()
    else:
        fact.parent.chmod(0o755)
        (fact.parent / 'extra.md').write_text('unrecorded fact')
    for read in (lambda: versions.load_generator(gen), lambda: versions.judge_dir(judge),
                 lambda: versions.version_payload(versions.GEN_ROOT / gen)):
        with pytest.raises(ConfigError, match='指纹|不能为空'):
            read()


@pytest.mark.parametrize('config', [
    {'share_ref': 's-0001'}, {'share_sha256': 'bad'},
    {'share_ref': 's-0001', 'share_sha256': 'bad'},
    {'share_ref': 's-9999', 'share_sha256': 'bad'},
    {'share_ref': '../s-0001', 'share_sha256': 'bad'},
])
def test_invalid_pins_cannot_create_versions(registry, config):
    shares.create(registry)
    with pytest.raises(ConfigError, match='Share'):
        versions.create_generator_version(config, 'd-0001', source_dir=versions.GEN_ROOT / 'g-0001')
    with pytest.raises(ConfigError, match='Share'):
        versions.create_judge_version(config, {})
    assert len(list(versions.GEN_ROOT.iterdir())) == 1
    assert len(list(versions.JUDGE_ROOT.iterdir())) == 1


def test_dashboard_reads_share_from_version_instance(registry, monkeypatch, tmp_path):
    from src.dashboard.changes import snapshot_delta

    instance = versions.PRIVATE
    ref = shares.create(registry)
    gen = shares.bind_model('gen', 'g-0001', ref)
    version_dir = versions.GEN_ROOT / gen
    expected = versions.version_payload(version_dir)

    # A second instance can have the same Share ID with different contents.
    other = tmp_path / 'other-instance'
    monkeypatch.setattr(versions, 'PRIVATE', other)
    (registry / 'people' / 'one.md').write_text('another person, another instance')
    assert shares.create(registry) == ref
    assert shares.load(ref)['manifest']['sha256'] != expected['config']['share_sha256']
    assert versions.version_payload(version_dir) == expected
    delta = snapshot_delta(instance, 'generators', 'g-0001', gen)
    assert delta['available']
    assert any(row['key'] == 'share_ref' for row in delta['changes'])
    assert versions.PRIVATE == other

    # Explicit instance resolution must retain the original integrity checks.
    fact = shares.load(ref, instance=instance)['dir'] / 'content' / 'people' / 'one.md'
    fact.chmod(0o644)
    fact.write_text('unrecorded change')
    with pytest.raises(ConfigError, match='指纹'):
        versions.version_payload(version_dir)


def test_review_only_pins_fail_before_runtime_or_api_initialization(registry, monkeypatch):
    from src.generator.generator import ReplyGenerator
    from src.judge.corrected import CorrectedJudge
    from src.judge.judge import Judge, build_judge
    import src.judge.judge as judge_module

    def no_client(*args, **kwargs):
        pytest.fail('must reject review-only knowledge before initializing an API client')

    monkeypatch.setattr(judge_module, 'ChatClient', no_client)
    pin = shares.binding(shares.create(registry))
    for build in (lambda: ReplyGenerator({}, pin, None), lambda: Judge(pin, None),
                  lambda: CorrectedJudge(pin, Path('/unused')),
                  lambda: build_judge({}, {'config': pin})):
        with pytest.raises(ConfigError, match='仅可审阅'):
            build()
    assert shares.validate_binding({'llm': {}}) is None
    shares.require_runtime({'llm': {}})
    assert shares.inventory()[0]['runtime_usable'] is False


def test_source_and_registry_must_not_overlap_or_follow_links(registry):
    with pytest.raises(ConfigError, match='递归复制'):
        shares.create(versions.PRIVATE)
    target = registry / 'people' / 'one.md'
    (registry / 'linked.md').symlink_to(target)
    with pytest.raises(ConfigError, match='符号链接'):
        shares.create(registry)
    (registry / 'linked.md').unlink()
    shares.root().symlink_to(registry, target_is_directory=True)
    with pytest.raises(ConfigError, match='Share'):
        shares.create(registry)


def test_cli_freeze_show_list_and_bind(registry, monkeypatch, capsys):
    from scripts.share import main
    monkeypatch.setattr(versions, 'switch_instance', lambda name: versions.PRIVATE)
    main(['--instance', 'synthetic', 'freeze', '--source', str(registry)])
    assert json.loads(capsys.readouterr().out)['id'] == 's-0001'
    main(['--instance', 'synthetic', 'bind', '--kind', 'gen', '--model', 'g-0001', '--share', 's-0001'])
    result = json.loads(capsys.readouterr().out)
    assert result['version'] == 'g-0002' and result['pointers_changed'] is False
    main(['--instance', 'synthetic', 'show', 's-0001'])
    assert json.loads(capsys.readouterr().out)['runtime_usable'] is False
    main(['--instance', 'synthetic', 'list'])
    assert json.loads(capsys.readouterr().out)[0]['consumers']['gen'] == ['g-0002']
