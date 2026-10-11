"""Reusable generation and feature stages with source-checked checkpoints."""
from __future__ import annotations
from threading import local
from src.infrastructure import cache, tracing
from src.infrastructure.config import ConfigError, load_settings
from ..generator.generator import GenerationConstraints, ReplyGenerator
from ..llm import build_clients
from ..judge import lr_retrain as lr, corrected_v1 as rt
from ..judge.corrected import CodexJudgeClient
from . import versions
from .parallel import completed_map
from src.infrastructure.storage import write_json, read_json as read
from .task_state import update as status
from .training_evidence import checkpoint

def check_generations(value, cases, identity):
    if value['identity'] != identity:
        raise ConfigError('生成检查点输入已变化')
    by_id = {c['case_id']: c for c in cases}
    for key, entry in value['entries'].items():
        if key not in by_id or entry['input_sha256'] != cache.digest(by_id[key]):
            raise ConfigError('训练题内容或来源已变化')
        if entry['status'] == 'ok':
            if ReplyGenerator._validate({'replies': entry['replies']}, True) is None:
                raise ConfigError('保存的训练 AI 回复无效')
            if entry['output_sha256'] != cache.digest(entry['replies']):
                raise ConfigError('保存的训练 AI 回复被修改')

def generate(directory, spec, snapshot, workers, limit=0):
    cases = snapshot['cases']
    from .materials import validation
    seal = validation.verify_generation(spec, cases)
    identity = cache.digest(spec)
    path = directory / 'generations.json'
    value = read(path) if path.exists() else {'identity': identity, 'entries': {}}
    check_generations(value, cases, identity)
    selected = cases[:limit] if limit else cases
    pending = [c for c in selected if value['entries'].get(c['case_id'], {}).get('status') != 'ok']
    tls, settings = local(), load_settings()
    gen = versions.load_generator(spec['generator_ref'])

    def one(case):
        seal.check()
        trace = tracing.CaseTrace(directory, case, spec)
        entry = {'input_sha256': cache.digest(case), 'trace_ref': trace.ref}
        try:
            if not hasattr(tls, 'generator'):
                tls.generator = ReplyGenerator(settings, gen['config'], build_clients(settings, gen['config']['llm']),
                    prompt_root=gen['dir'], pool_path=versions.data_version_dir(spec['data_ref']) / 'fewshot_pool.jsonl')
            excluded = snapshot['retrieval_excluded_ids_by_chat'][case['source_message_id'].rpartition(':')[0]]
            constraints = GenerationConstraints(frozenset(excluded), seal.check)
            with trace.operation('generation', 'training', 0, {'excluded_ids_sha256': cache.digest(excluded)}) as op:
                result = tls.generator.generate(case, constraints=constraints)
                op['result'] = result
            entry.update(status='ok', replies=result['replies'], output_sha256=cache.digest(result['replies']))
        except ConfigError:
            trace.finish('interrupted')
            raise
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:1200])
        except BaseException:
            trace.finish('interrupted')
            raise
        seal.check()
        trace.finish(entry['status'])
        return case['case_id'], entry

    _run_pending(directory, 'generating', cases, pending, value, path, one, seal, workers)
    return value


def train_features(directory, spec, rows, workers, limit):
    from .materials import validation
    from .materials.seals import RunSeal
    # Seal the old checkpoint/traces before reading them; no reuse or request
    # occurs until their complete source proof has been reconstructed below.
    previous = RunSeal([directory])
    try:
        value = checkpoint(directory, spec, rows)
    except (KeyError, ValueError, TypeError) as exc:
        raise ConfigError('特征检查点缺少有效的冻结来源信息') from exc
    seal = validation.verify_training(directory, spec, rows, value['entries'])
    previous.check()
    tls = local()
    def one(row):
        seal.check()
        trace = tracing.CaseTrace(directory, {'case_id': row['case_id'], 'blind': row['blind']}, spec)
        entry = {'input_sha256': cache.digest(row), 'trace_ref': trace.ref}
        try:
            if not hasattr(tls, 'client'):
                tls.client = CodexJudgeClient(spec['feature_config'])
            with trace.operation('feature_extraction', 'training', 0, {'prompt_sha256': cache.digest(rt.feature_extractor_prompt(row['blind']))}):
                features = lr.extract(tls.client, row['blind'], seal.check)
            entry.update(status='ok', features=features, features_sha256=cache.digest(features))
        except ConfigError:
            trace.finish('interrupted')
            raise
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:1200])
        except BaseException:
            trace.finish('interrupted')
            raise
        seal.check()
        trace.finish(entry['status'])
        return row['case_id'], entry
    pending = [r for r in rows if value['entries'].get(r['case_id'], {}).get('status') != 'ok']
    if limit:
        pending = pending[:limit]
    return _run_pending(directory, 'extracting', rows, pending, value,
                        directory / 'features.json', one, seal, workers)


def _run_pending(directory, phase, rows, pending, checkpoint_value, path, execute, seal, workers):
    """Retry failed cases and retain every attempt's trace in one checkpoint loop."""
    entries = checkpoint_value['entries']

    def progress():
        successful = sum(entry['status'] == 'ok' for entry in entries.values())
        status(directory, phase, total=len(rows), successful=successful,
               failed=len(entries) - successful, workers=workers)
        return successful

    progress()
    for attempt in range(3):
        for key, entry in completed_map(execute, pending, workers):
            seal.check()
            previous = entries.get(key, {})
            entry['attempts'] = previous.get('attempts', 0) + 1
            entry['previous_traces'] = [*previous.get('previous_traces', []),
                                       *([previous['trace_ref']] if previous else [])]
            entries[key] = entry
            write_json(path, checkpoint_value)
            successful = progress()
            print(f'{phase} {successful}/{len(rows)}; case={key} status={entry["status"]}', flush=True)
        pending = [row for row in pending if entries[row['case_id']]['status'] != 'ok']
        if not pending:
            break
        if attempt < 2:
            print(f'retrying {len(pending)} failed cases; pass={attempt + 2}', flush=True)
    seal.check()
    return progress() == len(rows)
