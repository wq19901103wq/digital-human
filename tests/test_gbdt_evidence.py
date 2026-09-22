"""End-to-end clean training provenance and forged GBDT rejection checks."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.offline_demo import run, SyntheticChat
from src.cache import digest
from src.config import ConfigError, sha256_file
from src.iteration import gbdt_evidence as audit, learning_guard, pack_transport, runtime, training_evidence, versions
from src.judge import corrected_v1 as rt
from src.judge.gbdt import option_vectors

pytest.importorskip('xgboost')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp('gbdt-source') / 'demo'
    result = run(output)
    return output, result


@pytest.fixture
def candidate(trained, tmp_path, monkeypatch):
    output, result = trained
    private = output / 'instances/demo'
    for key, value in dict(PRIVATE=private, DATA_ROOT=private/'data', GEN_ROOT=private/'generators',
                            JUDGE_ROOT=private/'judges', POINTERS_PATH=private/'pointers.json').items():
        monkeypatch.setattr(versions, key, value)
    monkeypatch.setenv('DH_SETTINGS_FILE', str(output / 'settings.yaml'))
    monkeypatch.setattr(training_evidence, 'build_clients', lambda settings, cfg: SyntheticChat(cfg))
    source = private / 'judge_training/synthetic-training'
    source_spec = audit.read(source / 'spec.json')
    arm = 'candidate'
    sweep = private / 'judge_training' / tmp_path.name
    recipe = dict(id='small-gbdt', kind='gbdt', objective='rank:pairwise', n_estimators=2,
        max_depth=1, learning_rate=.1, reg_lambda=1., reg_alpha=0., min_child_weight=0.,
        subsample=1., colsample_bytree=1., tree_method='hist', n_jobs=1, random_state=0)
    historical_code = tmp_path / 'historical_code.py'
    historical_code.write_text(f'value = {tmp_path.name!r}\n')
    spec = dict(kind='paired_classifier_sweep', dataset='development', adoption_allowed=False,
        source=str(source), source_sha256=sha256_file(source/'spec.json'), arms=[arm],
        recipes=[recipe], packages={'xgboost': audit.importlib.metadata.version('xgboost')},
        inputs={**source_spec['inputs'], str(source/'spec.json'): sha256_file(source/'spec.json'),
                str(historical_code): sha256_file(historical_code)},
        **{key: source_spec[key] for key in ('data_ref','generator_ref','feature_configs','training_total')})
    save(sweep / 'spec.json', spec)
    rows = audit.read(source / arm / 'training.json')['rows']
    entries = audit.read(source / arm / 'features.json')['entries']
    template = rt.load_formal_judge(source / 'model_template.json')
    pairs = [option_vectors(template, entries[r['case_id']]['features'], r['blind'], r['metadata']) for r in rows]
    a, b = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    y = np.array([r['human_option'] == 'A' for r in rows], dtype=float)
    model_path = sweep / arm / 'models/small-gbdt.json'
    receipt_path = sweep / arm / 'receipts/small-gbdt.json'
    manifest_path = sweep / arm / 'training_source.json'
    paths = [source / arm / name for name in ('spec.json','training.json','features.json','correction.json','candidate.json')]
    manifest = dict(study_sha256=sha256_file(sweep/'spec.json'), arm=arm,
        inputs={str(p.resolve()): sha256_file(p) for p in paths},
        training_matrix_sha256=digest(dict(case_ids=[r['case_id'] for r in rows], a=a.tolist(), b=b.tolist(), y=y.tolist())),
        dimensions=a.shape[1], training_total=len(rows), evaluation_used_for_fit=False)
    save(manifest_path, manifest)
    save(model_path, audit._refit(recipe, a, b, y))
    save(receipt_path, dict(identity=digest(dict(training=manifest, recipe=recipe)), model_sha256=sha256_file(model_path)))
    original = private / 'judges' / result['judge_ref']
    directory = tmp_path / 'candidate'
    directory.mkdir()
    for p in original.iterdir():
        if p.is_file():
            shutil.copyfile(p, directory / p.name)
    shutil.copyfile(model_path, directory / 'scorer.json')
    save(directory/'scorer_source.json', dict(schema=1, kind='paired_gbdt_v1', study=sweep.name, arm=arm, recipe_id=recipe['id']))
    proof_paths = [sweep/'spec.json', manifest_path, model_path, receipt_path, source/'model_template.json',
                   *paths, *runtime.verify_inputs(spec['inputs'])]
    record = audit.read(original / 'learning.json')
    record['asset_files'].update({n: sha256_file(directory/n) for n in ('scorer.json','scorer_source.json')})
    record['evidence_files'].update({str(p.resolve()): sha256_file(p) for p in proof_paths})
    save(directory/'learning.json', record)
    cfg = audit.read(original/'config.json')
    cfg.update(decision_policy='gbdt_only', scorer_file='scorer.json')
    cfg['assets'].update({n: sha256_file(directory/n) for n in ('scorer.json','scorer_source.json','learning.json')})
    save(directory/'config.json', cfg)
    return dict(directory=directory, original=original, study=source, source_spec=source_spec,
        arm=arm, model_path=model_path, receipt_path=receipt_path, manifest_path=manifest_path,
        a=a, b=b, y=y, recipe=recipe, historical_code=historical_code)


def check(c):
    return learning_guard.require_materials(c['source_spec']['data_ref'], [c['directory']])


def resign(c):
    """A forged self-attestation must never substitute for reconstruction."""
    d = c['directory']
    record = audit.read(d/'learning.json')
    record['asset_files'] = {n: sha256_file(d/n) for n in record['asset_files']}
    record['evidence_files'] = {p: sha256_file(Path(p)) for p in record['evidence_files']}
    save(d/'learning.json', record)
    cfg = audit.read(d/'config.json')
    cfg['assets'] = {n: sha256_file(d/n) for n in cfg['assets']}
    save(d/'config.json', cfg)


def test_clean_gbdt_reconstructs_source_without_requests(candidate, monkeypatch):
    from src.judge import corrected
    monkeypatch.setattr(corrected.CodexJudgeClient, 'run', lambda *a, **kw: pytest.fail('no model requests allowed'))
    before = versions.POINTERS_PATH.read_bytes()
    assert check(candidate)['promotion_eligible'] is True
    assert versions.POINTERS_PATH.read_bytes() == before


@pytest.mark.parametrize('damage', [None, 'archive', 'missing_binding', 'data', 'weights'])
def test_archived_code_compatibility_preserves_guard(candidate, damage):
    c, d = candidate, candidate['directory']
    code = c['historical_code']
    expected = sha256_file(code)
    runtime.archive_inputs({str(code): expected})
    code.unlink()
    before = {p: p.read_bytes() for p in [d/'learning.json', d/'config.json', versions.POINTERS_PATH]}
    record = audit.read(d/'learning.json')
    if damage == 'archive':
        (versions.PRIVATE/'evidence_blobs'/expected).write_text('changed')
    elif damage == 'missing_binding':
        record['evidence_files'].pop(str(code))
    elif damage == 'data':
        c['manifest_path'].write_text('{}')
    elif damage == 'weights':
        (d/'scorer.json').write_text('{}')
    original = audit.verify
    with pack_transport.archived_gbdt_paths(audit, runtime), pack_transport.archived_gbdt_paths(audit, runtime):
        args = (d, record, c['original'], c['study'], c['source_spec'], c['arm'])
        if damage:
            with pytest.raises(ConfigError):
                audit.verify(*args)
        else:
            assert check(c)['promotion_eligible'] is True
    assert audit.verify is original
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize('operation', ['verify', 'bind'])
def test_adoption_learning_uses_verified_archives(candidate, monkeypatch, operation):
    from src.iteration import gates
    c = candidate
    code = c['historical_code']
    expected = sha256_file(code)
    runtime.archive_inputs({str(code): expected})
    code.unlink()
    monkeypatch.setattr(learning_guard, operation, lambda spec: check(c))
    original = audit.verify
    action = getattr(gates, operation + '_learning')
    action({})
    assert audit.verify is original
    (versions.PRIVATE / 'evidence_blobs' / expected).write_text('corrupt')
    with pytest.raises(ConfigError):
        action({})
    assert audit.verify is original


@pytest.mark.parametrize('damage', [False, True])
def test_baseline_migration_uses_verified_archives(candidate, monkeypatch, damage):
    from src.iteration import baseline
    c = candidate
    code = c['historical_code']
    expected = sha256_file(code)
    runtime.archive_inputs({str(code): expected})
    code.unlink()
    original_lookup = versions.judge_dir
    monkeypatch.setattr(versions, 'judge_dir', lambda ref: (
        {'dir': c['directory']} if ref == 'archived-test' else original_lookup(ref)))
    before = versions.POINTERS_PATH.read_bytes()
    frozen = {name: (c['directory']/name).read_bytes() for name in ('learning.json', 'config.json')}
    receipt = baseline.prepare(c['source_spec']['data_ref'], c['source_spec']['generator_ref'],
                               'archived-test', reason='test archived migration')
    assert versions.POINTERS_PATH.read_bytes() == before
    if damage:
        (versions.PRIVATE/'evidence_blobs'/expected).write_text('corrupt')
        with pytest.raises(ConfigError):
            baseline.apply(receipt.stem)
        assert versions.POINTERS_PATH.read_bytes() == before
    else:
        assert baseline.apply(receipt.stem)['production_judge'] == 'archived-test'
    assert all((c['directory']/name).read_bytes() == value for name, value in frozen.items())


@pytest.mark.parametrize('part', ['feature_model', 'reference', 'origin', 'cutoff', 'arm', 'missing_evidence'])
def test_forged_learning_or_feature_source_rejected(candidate, part):
    c, d = candidate, candidate['directory']
    if part == 'feature_model':
        cfg = audit.read(d/'config.json')
        cfg['feature_llm']['model'] = 'different-model'
        save(d/'config.json', cfg)
    elif part in ('reference', 'origin'):
        save(d/('reference.json' if part == 'reference' else 'source_judge.json'), {})
    elif part == 'arm':
        value = audit.read(d/'scorer_source.json')
        value['arm'] = 'different-arm'
        save(d/'scorer_source.json', value)
    else:
        value = audit.read(d/'learning.json')
        if part == 'cutoff':
            value['information_end'] = 0
        else:
            value['evidence_files'].pop(str(c['receipt_path'].resolve()))
        save(d/'learning.json', value)
    resign(c)
    with pytest.raises(ConfigError):
        check(c)


@pytest.mark.parametrize('part', ['matrix', 'evaluation_flag', 'receipt', 'weights'])
def test_forged_fit_even_with_resigned_hashes_rejected(candidate, part):
    c = candidate
    if part in ('matrix', 'evaluation_flag'):
        value = audit.read(c['manifest_path'])
        value['training_matrix_sha256' if part == 'matrix' else 'evaluation_used_for_fit'] = 'forged' if part == 'matrix' else True
        save(c['manifest_path'], value)
    elif part == 'receipt':
        value = audit.read(c['receipt_path'])
        value['identity'] = '0'*64
        save(c['receipt_path'], value)
    else:
        altered = audit._refit({**c['recipe'], 'n_estimators': 3}, c['a'], c['b'], c['y'])
        altered['recipe'] = c['recipe']
        save(c['model_path'], altered)
        shutil.copyfile(c['model_path'], c['directory']/'scorer.json')
        value = audit.read(c['receipt_path'])
        value['model_sha256'] = sha256_file(c['model_path'])
        save(c['receipt_path'], value)
    resign(c)
    with pytest.raises(ConfigError, match='GBDT'):
        check(c)
