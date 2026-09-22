import pytest

from src.config import ConfigError
from src.generator.few_shot import PersonaFewShotRetriever
from src.generator.ranker_expansion import prepare, read_inventory
from src.generator.ranker_preparation import seed_checkpoints
from src.iteration.storage import read_json, write_json
from tests.test_ranker_expansion import setup_batch


def cached_batch(tmp_path):
    fixture, teacher, generator, old = setup_batch(tmp_path)
    source = tmp_path / 'larger'
    prepare(fixture.directory, teacher, generator, source, observations=100,
            reuse_inventory=old, progress=lambda *a, **k: None)
    # Only checkpoint files are needed; final inventory may not yet exist.
    (source / 'inventory.json').unlink()
    (source / 'report.json').unlink()
    return fixture, teacher, generator, old, source


def test_resize_reuses_partial_preparation_without_retrieval(tmp_path, monkeypatch):
    fixture, teacher, generator, old, source = cached_batch(tmp_path)
    output = tmp_path / 'smaller'
    source_bytes = (source / 'expansion_plan.json').read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('cached retrieval must not repeat')
    monkeypatch.setattr(PersonaFewShotRetriever, 'retrieve', forbidden)
    provenance = seed_checkpoints(source, output, observations=2, reuse_inventory=old)
    assert provenance['copied_targets'] == 2
    assert seed_checkpoints(source, output, observations=2, reuse_inventory=old) == provenance
    result = prepare(fixture.directory, teacher, generator, output, observations=2,
                     reuse_inventory=old, progress=lambda *a, **k: None)
    assert result['candidate_observations'] == 2 and result['shortfall'] == 0
    _, rows = read_inventory(output)
    assert len({(r['target_id'], e['example']['id']) for r in rows for e in r['candidates']}) == 2
    assert (source / 'expansion_plan.json').read_bytes() == source_bytes
    with pytest.raises(ConfigError, match='frozen artifact changed'):
        seed_checkpoints(source, output, observations=3, reuse_inventory=old)


@pytest.mark.parametrize('change', ['checkpoint', 'input', 'reuse_source'])
def test_resize_rejects_changed_sources(tmp_path, change):
    _, _, _, old, source = cached_batch(tmp_path)
    if change == 'checkpoint':
        path = next((source / 'preparation').glob('*.json'))
        record = read_json(path)
        record['candidates'] = []
        write_json(path, record)
    elif change == 'input':
        path = old / 'inventory.json'
        path.write_text(path.read_text() + '\n')
    else:
        old = tmp_path / 'different'
    with pytest.raises(ConfigError, match='变化'):
        seed_checkpoints(source, tmp_path / 'smaller', observations=2, reuse_inventory=old)
