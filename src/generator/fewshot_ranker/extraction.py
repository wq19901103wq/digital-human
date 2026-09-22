"""Independent semantic extraction, content addressed and safe to resume."""
import json
import os
from pathlib import Path
from threading import Event
import time

from ...config import ConfigError
from ...iteration import control
from ...iteration.parallel import completed_map
from ...iteration.storage import file_lock, read_json, write_json
from ..history_sources import digest, require
from . import schema
from .features import context_view, reply_view


def task(kind, view, identity):
    request = dict(kind=kind, prompt=schema.prompt(kind, view), schema=schema.output_schema(kind), client=identity)
    return digest(request), request


def prepare(groups, identity):
    tasks, refs = {}, {}
    def add(kind, view):
        key, request = task(kind, view, identity)
        tasks[key] = request
        return key
    for group in groups:
        target_key = add('context', context_view(group['target']))
        for entry in group['candidates']:
            example = entry['example']
            refs[(group['target_id'], example['id'])] = (target_key,
                add('context', context_view(example, example=True)), add('reply', reply_view(example)))
    return tasks, refs


def cached(directory, key, request):
    path = Path(directory) / (key + '.json')
    if not path.exists():
        return None
    value = read_json(path)
    require(value.get('key') == key and key == digest(request) and value.get('payload_sha256') ==
            digest({k: v for k, v in value.items() if k != 'payload_sha256'}), 'Feature cache binding changed')
    return schema.validate(request['kind'], value['features'])


def extract_one(directory, key, request, client):
    directory = Path(directory)
    with file_lock(directory / 'locks' / (key + '.lock')):
        saved = cached(directory, key, request)
        if saved is not None:
            return saved
        require(client.cache_identity() == request['client'], 'Feature client differs from frozen request')
        raw = client.run(request['prompt'], request['schema'])
        value = dict(key=key, features=schema.validate(request['kind'], json.loads(raw)), raw=raw,
                     completed_at=time.time())
        write_json(directory / (key + '.json'), {**value, 'payload_sha256': digest(value)})
        return value['features']


def run(tasks, directory, output, client, *, workers=16, passes=3):
    directory, output = Path(directory), Path(output)
    values = {key: value for key, request in tasks.items() if (value := cached(directory, key, request)) is not None}
    initial, started, errors, stop = len(values), time.time(), {}, Event()
    def progress(status, pass_number):
        elapsed = time.time() - started
        speed = (len(values) - initial) / elapsed if elapsed else 0
        state = dict(status=status, pid=os.getpid(), total=len(tasks), completed=len(values),
            reused=initial, remaining=len(tasks)-len(values), failed=len(errors), pass_number=pass_number,
            workers=workers, started_at=started, updated_at=time.time(), per_minute=round(speed*60, 2),
            eta_seconds=(len(tasks)-len(values))/speed if speed else None,
            recent_errors=list(errors.values())[-3:])
        write_json(output / 'feature_progress.json', state)
        print(json.dumps({k:v for k,v in state.items() if k != 'recent_errors'}, ensure_ascii=False), flush=True)
        return state
    def worker(key):
        if stop.is_set():
            return key, None, None
        control.check()
        try:
            return key, extract_one(directory, key, tasks[key], client), None
        except control.StopRequested:
            raise
        except Exception as exc:
            error = dict(key=key, type=type(exc).__name__, message=str(exc), updated_at=time.time())
            write_json(output / 'feature_failures' / (key + '.json'), error)
            return key, None, error
    last_saved, consecutive = started, 0
    for pass_number in range(1, passes+1):
        progress('extracting', pass_number)
        pending = [key for key in tasks if key not in values]
        for key, value, error in completed_map(worker, pending, workers):
            if value is not None:
                values[key] = value
                errors.pop(key, None)
                consecutive = 0
            elif error:
                errors[key] = error
                consecutive += 1
                if consecutive >= max(workers*2, 8):
                    stop.set()
            if time.time()-last_saved >= 15 or len(values) == len(tasks) or stop.is_set():
                progress('needs_attention' if stop.is_set() else 'extracting', pass_number)
                last_saved = time.time()
        if stop.is_set() or len(values) == len(tasks):
            break
    status = 'complete' if len(values) == len(tasks) else 'needs_attention'
    progress(status, pass_number)
    if status != 'complete':
        raise ConfigError(f'Feature extraction incomplete: {len(values)}/{len(tasks)}; successful results cached')
    return values
