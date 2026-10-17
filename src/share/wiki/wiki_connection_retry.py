"""Opt-in connection retries around unchanged, budgeted Wiki requests."""
import json
from pathlib import Path
from threading import RLock
import time
import uuid

from ...iteration.storage import read_json, write_json
from ...judge.codex_transport import CodexServiceBlocked, CodexTransportError
from . import wiki_repair
from .wiki_library import digest


CONNECTION_SIGNALS = frozenset({'stream_disconnected', 'request_timeout'})


def retryable_diagnostic(diagnostic):
    terminal = diagnostic.get('terminal_signals', [])
    signals = diagnostic.get('signals', [])
    if (not all(isinstance(signal, str) for signal in terminal + signals)
            or diagnostic.get('service_errors')
            or not set(signals) <= CONNECTION_SIGNALS | {'model_catalog_error', 'mcp_connection_error'}):
        return False
    if terminal:
        return set(terminal) <= CONNECTION_SIGNALS
    return (diagnostic.get('failure_kind') == 'subprocess_timeout'
            and diagnostic.get('exit_code_source') == 'client_timeout'
            and 'request_timeout' in signals)


class ConnectionRetries:
    def __init__(self, output, seconds):
        self.output = Path(output)
        self.seconds = seconds
        self.guard = RLock()
        self.state = dict(schema='wiki_connection_retry_v1', run_id=uuid.uuid4().hex,
                          interval_seconds=seconds, total_retries=0, waiting={})
        write_json(self.output / 'connection-retry.json', self.state)

    def wrap(self, client, job):
        return ConnectionRetryClient(client, self, job)

    def record(self, job, request_sha256, phase, retries, diagnostic=None):
        with self.guard:
            event = dict(run_id=self.state['run_id'], job_id=job['id'],
                         request_sha256=request_sha256, phase=phase, retries=retries,
                         updated_at=time.time())
            if diagnostic is not None:
                event['diagnostic'] = diagnostic
            key = f"{job['id']}:{request_sha256}"
            if phase == 'waiting':
                event['next_retry_at'] = event['updated_at'] + self.seconds
                self.state['waiting'][key] = event
                self.state['total_retries'] += 1
            else:
                self.state['waiting'].pop(key, None)
            self.state['last_event'] = event
            with (self.output / 'connection-retry.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            write_json(self.output / 'connection-retry.json', self.state)


class ConnectionRetryClient:
    def __init__(self, client, retries, job):
        self.client = client
        self.retries = retries
        self.job = job

    def cache_identity(self):
        return self.client.cache_identity()

    def run(self, prompt, schema=None):
        request_sha256 = digest([self.cache_identity(), prompt, schema])
        retries = 0
        while True:
            try:
                result = self.client.run(prompt, schema)
            except CodexTransportError as exc:
                diagnostic = wiki_repair._transport_diagnostic(str(exc))
                if (isinstance(exc, CodexServiceBlocked) or diagnostic is None
                        or not retryable_diagnostic(diagnostic)):
                    if retries:
                        self.retries.record(self.job, request_sha256, 'stopped', retries, diagnostic)
                    raise
                retries += 1
                self.retries.record(self.job, request_sha256, 'waiting', retries, diagnostic)
                time.sleep(self.retries.seconds)
                self.retries.record(self.job, request_sha256, 'retrying', retries)
            else:
                if retries:
                    self.retries.record(self.job, request_sha256, 'recovered', retries)
                return result


def status(output, run_id, *, running):
    state = read_json(Path(output) / 'connection-retry.json', default={})
    if not run_id or state.get('run_id') != run_id:
        return None
    return dict(run_id=run_id, interval_seconds=state['interval_seconds'],
                total_retries=state['total_retries'],
                waiting=list(state['waiting'].values()) if running else [],
                last_event=state.get('last_event'))
