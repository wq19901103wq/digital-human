"""逐题实录与实际请求一致，失败/补测可追踪，固定集不可旁路读取。"""
import io
import json
import os
import subprocess
from src.iteration.storage import write_json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from src import llm, tracing
from src.dashboard import trace_view
from src.generator.generator import ReplyGenerator
from src.dashboard import report
from src.iteration import experiment
from src.iteration.progress import RunProgress, track_run
from src.judge import corrected
import test_live_dashboard
from test_report import _run
import test_corrected_judge
from test_corrected_judge import features

live_server = test_live_dashboard.live_server
bundle = test_corrected_judge.bundle
case = test_corrected_judge.case


@pytest.fixture
def setup_case(tmp_path, monkeypatch, case):
    monkeypatch.setenv('TRACE_URL', 'https://example.test/api/coding')
    monkeypatch.setenv('TRACE_KEY', 'secret-trace-key')
    settings = {'llm': {'base_url_env': 'TRACE_URL', 'api_key_env': 'TRACE_KEY', 'max_retries': 2},
                'evaluation': {'few_shots_per_case': 2, 'few_shots_char_budget': 1000}}
    exp, _ = _run(tmp_path, state={'status': 'running'})
    progress = RunProgress(exp)
    progress.start_case(case)
    return exp, progress, settings


def _response(text):
    return io.BytesIO(json.dumps({'content': [{'type': 'text', 'text': text},
        {'type': 'thinking', 'thinking': 'PRIVATE_REASONING'}], 'stop_reason': 'end_turn',
        'usage': {'output_tokens': 21}}).encode())


def test_actual_prompt_before_send_retry_and_format_retry(setup_case, case, monkeypatch):
    exp, progress, settings = setup_case
    calls = []
    def transport(request, timeout):
        doc = tracing.read(exp, progress.trace.ref)
        event = doc['operations'][-1]['events'][-1]
        assert event['status'] == 'running' and 'response' not in event
        assert event['request']['body'] == json.loads(request.data)  # 包括协议附加的 system JSON 规则
        calls.append(event)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, 'limited', {}, io.BytesIO(b'secret-trace-key'))
        return _response('not JSON' if len(calls) == 2 else '{"replies":["八点"]}')
    monkeypatch.setattr(llm.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(llm.time, 'sleep', lambda *_: None)
    generator = ReplyGenerator(settings, {'retriever': {'enabled': False}}, llm.ChatClient(settings, {'model': 'test'}),
                               prompt_root=exp.parent.parent / 'generators/base')
    result = progress.generate(generator, case, 'candidate', False, 2)
    assert result['replies'] == ['八点']
    doc = tracing.read(exp, progress.trace.ref)
    op = doc['operations'][0]
    assert op['round'] == 2 and op['branch'] == 'candidate'
    assert op['input']['forced_reply'] is False
    assert [e['status'] for e in op['events'] if e['kind'] == 'llm'] == ['failed', 'ok', 'ok']
    assert [e['data']['valid'] for e in op['events'] if e['kind'] == 'validation'] == [False, True]
    wire = calls[-1]['request']['body']
    assert '<unread>\n几点？\n</unread>' in wire['messages'][0]['content']
    assert 'JSON' in wire['system'] and 'force_reply' not in wire['messages'][0]['content']
    saved = progress.trace.path.read_text()
    assert 'PRIVATE_REASONING' not in saved and 'secret-trace-key' not in saved
    page = trace_view.payload(exp, progress.trace.ref)['html']
    assert '实际提示词 · system' in page and '模型返回正文 · 完整文本' in page
    assert '&lt;unread&gt;' in page and '<unread>' not in page


def test_openai_trace_matches_request(setup_case, case, monkeypatch):
    exp, progress, settings = setup_case
    monkeypatch.setenv('TRACE_URL', 'https://example.test/v1')
    def create(**kwargs):
        event = tracing.read(exp, progress.trace.ref)['operations'][-1]['events'][-1]
        assert kwargs == event['request']['body']
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='ok'))])
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    with progress.trace.operation('judge', 'baseline', 0, {}):
        assert llm.ChatClient(settings, {'model': 'test'}).chat([{'role': 'user', 'content': '原始提示词'}], True) == 'ok'


def test_corrected_real_adapter_records_feature_retry_probability_and_mapping(setup_case, bundle, case, monkeypatch):
    exp, progress, _ = setup_case
    cfg, directory = bundle
    execution = {'provider': 'codex_cli', 'model': 'test', 'reasoning_effort': 'low', 'timeout_seconds': 360, 'codex_cli_version': '0.144.1'}
    cfg['llm'] = execution
    calls = []
    def run(command, **kwargs):
        if command == ['codex', '--version']:
            return SimpleNamespace(stdout='codex-cli 0.144.1')
        calls.append(kwargs['input'])
        event = tracing.read(exp, progress.trace.ref)['operations'][-1]['events'][-1]
        assert event['request']['prompt'] == kwargs['input'] and event['status'] == 'running'
        schema = event['request']['schema']
        if schema:
            assert json.loads(Path(command[command.index('--output-schema') + 1]).read_text()) == schema
        text = json.dumps({'human_option': 'B', 'confidence': .9, 'reason': '测试'})
        if len(calls) == 2:
            text = 'invalid features'
        elif len(calls) == 3:
            text = json.dumps({'option_A': features(), 'option_B': features()})
        Path(command[command.index('--output-last-message') + 1]).write_text(text)
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed","usage":{"output_tokens":12}}\n', stderr='')
    monkeypatch.setattr(corrected.subprocess, 'run', run)
    monkeypatch.setattr(corrected.random, 'random', lambda: .9)
    monkeypatch.setattr(corrected.runtime, 'score_formal_judge_pair', lambda *_: .8)
    assert progress.judge(corrected.CorrectedJudge(cfg, directory), case, ['九点'], 'baseline')
    doc = tracing.read(exp, progress.trace.ref)
    events = doc['operations'][0]['events']
    assert [e['data']['valid'] for e in events if e['kind'] == 'features'] == [False, True]
    verdict = events[-1]['data']
    assert verdict['candidate_option'] == 'B' and verdict['correction_applied'] is True
    assert verdict['small_model_probability_a'] == .8
    page = trace_view.payload(exp, progress.trace.ref)['html']
    assert '80.00% / 70.00%' in page and '盲测 A/B 对应关系' in page


def test_failure_and_resume_keep_previous_attempt(setup_case, case):
    exp, progress, _ = setup_case
    old_ref = progress.trace.ref
    class Failure:
        def generate(self, *_, **__):
            with tracing.step('llm', {'body': {'messages': [{'role': 'user', 'content': '已发送'}]}}):
                raise TimeoutError('timeout')
    with pytest.raises(TimeoutError):
        progress.generate(Failure(), case, 'baseline')
    row = {'case_id': case['case_id'], 'status': 'failed'}
    progress.finish_case(row)
    assert row['trace_ref'] == old_ref
    resumed = RunProgress(exp)
    resumed.start_case(case)
    assert resumed.trace.ref != old_ref
    assert resumed.trace.value['previous_trace_ref'] == old_ref
    previous = tracing.read(exp, old_ref)
    assert previous['operations'][0]['events'][0]['error']['type'] == 'TimeoutError'
    assert '已发送' in trace_view.payload(exp, old_ref)['html']


def test_interruption_persists_inflight_prompt_and_clears_sink(setup_case, case, monkeypatch):
    exp, progress, _ = setup_case
    test_live_dashboard._select_execution_instance(exp, monkeypatch)
    (exp.parent.parent / 'data/data/dev_pool.jsonl').write_text(json.dumps(case) + '\n')
    @track_run
    def execute(directory, *, _progress):
        _progress.start_case(case)
        class Interrupt:
            def generate(self, *_, **__):
                with tracing.step('llm', {'body': {'messages': []}}):
                    raise KeyboardInterrupt()
        _progress.generate(Interrupt(), case, 'baseline')
    with pytest.raises(KeyboardInterrupt):
        execute(exp)
    ref = experiment.state_of(exp)['progress']['trace_ref']
    doc = tracing.read(exp, ref)
    assert doc['status'] == 'interrupted'
    assert doc['operations'][0]['events'][0]['status'] == 'failed'
    assert tracing._sink.get() is None


def test_pending_case_visible_and_prompts_are_lazy_loaded(setup_case, case, live_server):
    exp, progress, _ = setup_case
    with progress.trace.operation('generation', 'baseline', 0, {}):
        with tracing.step('llm', {'body': {'messages': [{'role': 'user', 'content': 'LAZY_PROMPT<script>bad()</script>'}]}}):
            page = urlopen(live_server + '/instances/demo/experiments/test-run/index.html').read().decode()
            assert 'LAZY_PROMPT' not in page and f'data-case-id="{case["case_id"]}"' in page
            endpoint = f'{live_server}/api/trace/demo/test-run/{progress.trace.ref}'
            payload = json.load(urlopen(endpoint))
            assert 'LAZY_PROMPT&lt;script&gt;bad()&lt;/script&gt;' in payload['html']
            assert payload['status'] == 'running'
            assert 'html' not in json.load(urlopen(endpoint + '?revision=' + payload['revision']))
    assert 'cases' in report.live_payload(exp.parent.parent, exp.name)['regions']


@pytest.mark.parametrize('suffix', ['', '?download=1', '%3Fdownload=1'])
def test_raw_traces_blocked_even_through_symlink(setup_case, live_server, suffix):
    exp, progress, _ = setup_case
    alias = exp / 'alias.json'
    alias.symlink_to(progress.trace.path)
    for path in [f'/instances/demo/experiments/test-run/traces/{progress.trace.ref}.json', '/instances/demo/experiments/test-run/alias.json']:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urlopen(live_server + path + suffix)
        assert caught.value.code == 403


def test_custom_fixed_run_trace_api_denied(setup_case, live_server):
    exp, progress, _ = setup_case
    spec = experiment.spec_of(exp)
    spec['dataset'] = 'fixed_test'
    write_json(exp / 'spec.json', spec)
    for path in [f'/api/trace/demo/test-run/{progress.trace.ref}', '/instances/demo/experiments/test-run/cases.jsonl']:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urlopen(live_server + path)
        assert caught.value.code == 403
    page = urlopen(live_server + '/instances/demo/experiments/test-run/index.html').read().decode()
    assert 'data-trace-url=' not in page
    assert 'source_message_id' not in page


def test_historical_data_boundary_warning_does_not_invent_actual_prompt(tmp_path):
    _, run = _run(tmp_path, records=[{'case_id': 'x', 'context': [{'sender': '王芊', 'text': '自己的回复', 'is_self': True}]}])
    page = report._cases_html(run)
    assert '本人 · 被模仿对象' in page and '数据边界异常' in page
    assert '历史未记录' in page and '不能当作本题实际输入' in page


@pytest.fixture
def cached_trace(setup_case, case):
    exp, progress, _ = setup_case
    source = exp.with_name('source-run')
    source.mkdir()
    spec = {**experiment.spec_of(exp), 'id': source.name}
    write_json(source / 'spec.json', spec)
    original = tracing.CaseTrace(source, case, spec)
    with original.operation('generation', 'baseline', 0, {}):
        tracing.note('validation', {'raw': 'UNRELATED_STEP'})
    with original.operation('generation', 'candidate', 0, {}) as op:
        with tracing.step('llm', {'body': {'model': 'original-model', 'system': 'ORIGINAL_SYSTEM',
                'messages': [{'role': 'user', 'content': 'ORIGINAL_PROMPT<script>bad()</script>'}]}}) as event:
            event['response'] = {'text': 'ORIGINAL_RESPONSE'}
        op['result'] = {'replies': ['ORIGINAL_RESPONSE']}
    original.finish('ok')
    origin = {'experiment_id': source.name, 'experiment_path': '/old/location/experiments/source-run',
              'trace_ref': original.ref, 'operation_index': 1, 'branch': 'candidate', 'round': 0}
    with progress.trace.operation('generation', 'baseline', 0, {}):
        tracing.note('cache_hit', {'layer': 'generation', 'key': 'saved-key', 'origin': origin})
    return exp, progress.trace, source, original, origin


def test_cached_prompt_source_is_lazy_scoped_and_read_only(cached_trace, live_server):
    exp, current, source, original, _ = cached_trace
    before = original.path.read_bytes()
    endpoint = f'/api/trace/demo/{exp.name}/{current.ref}'
    page = json.load(urlopen(live_server + endpoint))['html']
    source_url = f'/api/trace/demo/{source.name}/{original.ref}?operation=1'
    assert source_url in page and '展开缓存来源的实际提示词与调用过程' in page
    assert 'ORIGINAL_PROMPT' not in page
    loaded = json.load(urlopen(live_server + source_url))
    assert '2. 候选 · 初测 · 生成回复' in loaded['html']
    assert 'ORIGINAL_SYSTEM' in loaded['html'] and 'ORIGINAL_RESPONSE' in loaded['html']
    assert 'ORIGINAL_PROMPT&lt;script&gt;bad()&lt;/script&gt;' in loaded['html']
    assert 'UNRELATED_STEP' not in loaded['html']
    assert '没有重新发送这些请求' in loaded['html']
    assert 'html' not in json.load(urlopen(live_server + source_url + '&revision=' + loaded['revision']))
    assert before == original.path.read_bytes()


@pytest.mark.parametrize('problem', ['missing', 'fixed', 'outside', 'different-case', 'bad-id', 'bad-step', 'self'])
def test_unavailable_cache_origin_does_not_invent_prompts(cached_trace, problem, tmp_path):
    exp, current, source, original, origin = cached_trace
    if problem == 'missing':
        original.path.unlink()
    elif problem == 'fixed':
        write_json(source / 'spec.json', {**experiment.spec_of(source), 'dataset': 'fixed_test'})
    elif problem == 'outside':
        moved = tmp_path / 'outside-instance'
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    elif problem == 'different-case':
        original.value['case_id'] = 'another-case'
        original.save()
    elif problem == 'bad-id':
        origin['experiment_id'] = '../source-run'
    elif problem == 'bad-step':
        origin['operation_index'] = 100
    else:
        origin.update(experiment_id=exp.name, trace_ref=current.ref, operation_index=0)
    current.save()
    page = trace_view.payload(exp, current.ref)['html']
    assert '缓存来源实录不可用' in page and '不能用当前版本模板还原' in page
    assert 'data-trace-url=' not in page and 'ORIGINAL_PROMPT' not in page


@pytest.mark.parametrize('operation,status', [('bad', 400), ('-1', 400), ('100', 404)])
def test_invalid_trace_operation_rejected(cached_trace, live_server, operation, status):
    _, _, source, original, _ = cached_trace
    with pytest.raises(urllib.error.HTTPError) as caught:
        urlopen(f'{live_server}/api/trace/demo/{source.name}/{original.ref}?operation={operation}')
    assert caught.value.code == status


def _check_trace_browser(url, expected_prompt):
    """复用浏览器验收：真实点击、缓存来源、轮询后的展开状态；经 check.py 调用。"""
    result = subprocess.run(['node', '-e', r'''
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true, timeout: 15000});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.argv[1], {waitUntil: 'domcontentloaded'});
    assert.deepEqual(errors, [], '页面初始化不得有 JavaScript 错误');
    const item = page.locator('.case:not([hidden])').first();
    await item.locator(':scope > summary').click();
    const trace = item.locator('[data-trace-url]').first();
    await trace.locator(':scope > summary').click();
    await trace.locator(':scope > .detail-body > [data-trace-body] > p').first().waitFor();
    assert.match(await trace.getAttribute('data-revision'), /\d/);
    await trace.locator('[data-detail-key^="operation-"]').first().locator(':scope > summary').click();
    const event = trace.locator('[data-detail-key^="event-"]').filter({has: page.locator('[data-trace-url*="?operation="]')}).first();
    await event.locator(':scope > summary').click();
    const source = event.locator('[data-trace-url]').first();
    await source.locator(':scope > summary').click();
    const body = source.locator(':scope > .detail-body > [data-trace-body]');
    await body.locator('[data-detail-key^="operation-"]').waitFor({state: 'attached'});
    assert.ok((await body.textContent()).includes(process.argv[2]));
    assert.ok((await source.getAttribute('data-revision')).includes(':'));
    await body.locator('[data-detail-key^="operation-"] > summary').click();
    const request = body.locator('[data-detail-key^="event-"]').filter({has: page.getByText('完整请求参数', {exact: true})}).first();
    await request.locator(':scope > summary').click();
    for (const summary of await request.locator('summary').filter({hasText: /^实际提示词|^模型返回正文/}).all()) {
      await summary.click();
    }
    assert.ok(await request.locator('pre:visible').first().isVisible(), '提示词正文必须实际可见');
    await page.waitForTimeout(3500);
    assert.equal(await source.getAttribute('open'), '');
    assert.ok((await body.textContent()).includes(process.argv[2]));
    assert.ok(await request.locator('pre:visible').first().isVisible());
    assert.deepEqual(errors, [], '加载和实时刷新不得有 JavaScript 错误');
    console.log('浏览器通过：列表初始化、实录展开、来源提示词加载、实时刷新保留展开状态');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
''', url, expected_prompt], capture_output=True, text=True, timeout=50)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


@pytest.mark.skipif(os.getenv('DH_BROWSER_TESTS') != '1', reason='可选 Chrome + Node Playwright 浏览器回归')
def test_browser_trace_loading_and_live_refresh(cached_trace, live_server):
    exp, _, _, _, _ = cached_trace
    _check_trace_browser(f'{live_server}/instances/demo/experiments/{exp.name}/index.html', 'ORIGINAL_PROMPT')


@pytest.mark.skipif(not os.getenv('DH_DASHBOARD_TEST_URL'), reason='可选现有开发实验的只读浏览器验收')
def test_browser_existing_experiment_trace():
    _check_trace_browser(os.environ['DH_DASHBOARD_TEST_URL'], '实际提示词')
