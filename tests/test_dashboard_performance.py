"""Bound dashboard I/O by distinct inputs and preserve freshness and isolation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from threading import Barrier
from urllib.request import urlopen

import pytest

from src.dashboard import changes, diagnostics, report
from src.dashboard.read_scope import once, scope
from test_live_dashboard import live_server as _live_server
from test_report import _run, _write

live_server = _live_server


def _history(tmp_path, count=8):
    exp, run = _run(tmp_path)
    for i in range(count - 1):
        spec = {**run['spec'], 'id': f'old-{i}', 'created': f'2026-01-{i+1:02}',
                'single_change': f'历史任务 {i}'}
        _write(exp.parent / spec['id'] / 'spec.json', spec)
        _write(exp.parent / spec['id'] / 'state.json', run['state'])
    return exp


def test_history_reads_each_spec_once_and_refreshes_changed_records(tmp_path, monkeypatch):
    exp = _history(tmp_path)
    calls = Counter()
    original = report.experiment.spec_of

    def read(directory):
        calls[directory] += 1
        return original(directory)

    monkeypatch.setattr(report.experiment, 'spec_of', read)
    first = report.dashboard_html(exp.parent)
    assert len(calls) == 8 and set(calls.values()) == {1}
    assert first.count('<tr data-item') == 8
    spec = original(exp)
    _write(exp / 'spec.json', {**spec, 'single_change': '新任务说明'})
    second = report.live_payload(exp.parent.parent)
    assert '新任务说明' in second['regions']['history']
    assert set(calls.values()) == {2}


def test_http_home_renders_once_without_writing_static_reports(tmp_path, live_server, monkeypatch):
    exp = _history(tmp_path)
    calls = []
    original = report.dashboard_html

    def render(directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(report, 'dashboard_html', render)
    content = urlopen(live_server + '/dashboard/demo/index.html').read().decode()
    assert len(calls) == 1 and content.count('<tr data-item') == 8
    assert not (tmp_path / 'dashboard').exists()
    assert not (exp / 'index.html').exists()


def test_shared_packs_and_assets_read_once_but_changes_are_not_hidden(tmp_path, monkeypatch):
    exp, run = _run(tmp_path, kind='judge_eval')
    instance = exp.parent.parent
    pack = instance / 'judge_eval/calibration-pack/pack.json'
    _write(pack, {'c0_gen_version': 'base', 'rows': []})
    for ref in ('base', 'candidate'):
        _write(instance / 'judges' / ref / 'config.json', {'assets': {'weights.json': 'declared'}})
        _write(instance / 'judges' / ref / 'weights.json', {'weights': [1]})
    old = {**run['spec'], 'id': 'old', 'created': '2026-01-01'}
    _write(exp.parent / 'old/spec.json', old)
    reads = Counter()
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def text(path, *args, **kwargs):
        reads[path] += 1
        return original_text(path, *args, **kwargs)

    def blob(path):
        reads[path] += 1
        return original_bytes(path)

    monkeypatch.setattr(Path, 'read_text', text)
    monkeypatch.setattr(Path, 'read_bytes', blob)
    report.dashboard_html(exp.parent)
    assert reads[pack] == 1
    asset = instance / 'judges/candidate/weights.json'
    assert reads[asset] == 1
    _write(asset, {'weights': [2]})
    result = scope(changes.snapshot_delta)(instance, 'judges', 'base', 'candidate')
    assert result['changes'][0]['label'] == 'weights.json'
    assert reads[asset] == 2


def test_read_scopes_are_concurrent_and_exceptions_do_not_leak_cached_values():
    barrier = Barrier(2)

    @once
    def value():
        return object()

    @scope
    def render():
        first = value()
        barrier.wait(timeout=5)
        assert value() is first
        return first

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda _: render(), range(2)))
    assert a is not b

    @scope
    def broken():
        assert value() is value()
        raise ValueError('interrupted render')

    with pytest.raises(ValueError):
        broken()
    assert value() is not value()


def test_diagnostics_profile_real_paths_without_writing_results(tmp_path):
    exp, _ = _run(tmp_path)
    before = {p: p.read_bytes() for p in exp.parent.parent.rglob('*') if p.is_file()}
    result = diagnostics.inspect(exp.parent.parent)
    assert set(result) == {'home', 'live'}
    assert all(v['bytes'] > 0 and v['profile'] for v in result.values())
    assert before == {p: p.read_bytes() for p in exp.parent.parent.rglob('*') if p.is_file()}
    json.dumps(result)


def _check_home_browser(url):
    """Check navigation, actual rendering, and retained controls across polling."""
    result = subprocess.run(['node', '-e', r'''
const assert = require('node:assert/strict');
let chromium;
try { ({chromium} = require('playwright')); }
catch { ({chromium} = require('playwright-core')); }
(async () => {
  const executable = process.env.DH_BROWSER_EXECUTABLE_PATH;
  const browser = await chromium.launch({...(executable ? {executablePath:executable} : {channel:'chrome'}), headless:true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const response = await page.goto(process.argv[1], {waitUntil:'domcontentloaded', timeout:30000});
    assert.equal(response.status(), 200);
    await page.waitForFunction(() => document.querySelector('[data-live-status]').textContent.startsWith('实时更新'));
    const rows = page.locator('#history tr[data-item]:visible');
    assert.ok(await rows.count() > 0);
    assert.ok(await rows.count() <= 10);
    const title = (await rows.first().locator('.task-link').textContent()).trim();
    const search = page.locator('#history input[type="search"]');
    await search.fill(title);
    const details = rows.first().locator('details').first();
    await details.locator(':scope > summary').click();
    const lastSync = await page.locator('[data-live-status]').textContent();
    await page.waitForResponse(r => r.url().includes('/api/live/') && r.status() === 200);
    await page.waitForFunction(previous => {
      const status = document.querySelector('[data-live-status]').textContent;
      return status.startsWith('实时更新') && status !== previous;
    }, lastSync);
    assert.equal(await search.inputValue(), title);
    assert.equal(await details.getAttribute('open'), '');
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log(JSON.stringify(await page.evaluate(() => {
      const nav = performance.getEntriesByType('navigation')[0];
      return {first_byte_ms:nav.responseStart, dom_ready_ms:nav.domContentLoadedEventEnd,
              rows:document.querySelectorAll('#history tr[data-item]').length};
    })));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
''', url], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(os.getenv('DH_BROWSER_TESTS') != '1', reason='可选 Chrome + Node Playwright 浏览器回归')
def test_browser_home_retains_search_and_details(tmp_path, live_server):
    _history(tmp_path)
    result = _check_home_browser(live_server + '/dashboard/demo/index.html')
    assert result['rows'] == 8


@pytest.mark.skipif(not os.getenv('DH_DASHBOARD_HOME_URL'), reason='未指定真实后台首页')
def test_browser_existing_home():
    result = _check_home_browser(os.environ['DH_DASHBOARD_HOME_URL'])
    print('dashboard browser:', json.dumps(result))
    assert result['rows'] > 0
