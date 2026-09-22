"""版本详情 HTTP 路由、真实文件内容与固定数据边界。"""
import html
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from src.dashboard import version_view
from src.dashboard import report
from test_live_dashboard import live_server as _live_server_fixture
from test_report import _run, _write

live_server = _live_server_fixture


def _get(server, path):
    return urlopen(server + path).read().decode()


@pytest.mark.parametrize('kind,ref,filename,content', [
    ('data', 'data', 'persona.md', '冻结的人格材料'),
    ('generators', 'base', 'scenarios/group.md', '冻结的群聊规则 <script>bad()</script>'),
    ('judges', 'candidate', 'correction.json', '{"coefficients": [0.12, -0.34], "training": "frozen"}'),
])
def test_version_routes_show_exact_frozen_file_without_writing(tmp_path, live_server, kind, ref, filename, content):
    exp, _ = _run(tmp_path)
    path = exp.parent.parent / kind / ref / filename
    _write(path, content)
    before = path.read_bytes()
    page = _get(live_server, f'/dashboard/demo/versions/{kind}/{ref}/index.html?file={filename}')
    assert content in html.unescape(page)
    assert '<script>bad()</script>' not in page
    assert 'SHA-256' in page and path.read_bytes() == before


def test_history_links_bind_judge_pack_generator_and_both_compared_versions(tmp_path):
    exp, run = _run(tmp_path, kind='judge_eval')
    _write(exp.parent.parent / 'judge_eval/calibration-pack/pack.json', {'c0_gen_version': 'frozen-gen'})
    page = report._history_versions(run)
    for path in ('data/data', 'generators/frozen-gen', 'judges/base', 'judges/candidate'):
        assert f'/versions/{path}/index.html' in page
    assert '/generators/base/' not in page
    assert '/generators/candidate/' not in page


def test_data_pages_preserve_records_order_and_never_render_fixed_answers(tmp_path, live_server):
    exp, _ = _run(tmp_path)
    directory = exp.parent.parent / 'data/data'
    rows = [{'case_id': f'case-{i}', 'context': [{'sender': '甲', 'text': f'输入-{i}'}],
             'human_reply': [f'回复-{i}']} for i in range(23)]
    _write(directory / 'dev_pool.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    _write(directory / 'fixed_test.jsonl', '{"human_reply": ["FIXED_SECRET"]}')
    base = '/dashboard/demo/versions/data/data/index.html'
    page1 = _get(live_server, base)
    page2 = _get(live_server, base + '?view=development&page=2')
    assert page1.count('class="version-record"') == 20
    assert page2.count('class="version-record"') == 3
    assert '输入-20' not in page1 and '输入-20' in page2
    assert 'FIXED_SECRET' not in page1 + page2
    assert '下一页</a>' in page1 and '下一页</a>' not in page2


def test_fewshot_page_shows_actual_example_not_development_answer(tmp_path, live_server):
    exp, _ = _run(tmp_path)
    directory = exp.parent.parent / 'data/data'
    _write(directory / 'fewshot_pool.jsonl', json.dumps({'id': 'pool-row', 'context': ['甲: 上下文'], 'reply': '风格原句'}, ensure_ascii=False))
    page = _get(live_server, '/dashboard/demo/versions/data/data/index.html?view=fewshot')
    assert 'pool-row' in page and '风格原句' in page and '上下文' in page


def test_full_history_pool_cannot_expose_sealed_answers(tmp_path, live_server):
    exp, _ = _run(tmp_path)
    directory = exp.parent.parent / 'data/data'
    _write(directory / 'purposes.json', {'roles': {}, 'protocol': {'public_history_before': 100}})
    _write(directory / 'fewshot_pool.jsonl', '\n'.join(json.dumps(r) for r in [
        {'id': 'old', 'context': [], 'reply': ['PUBLIC_PAST'], 'source_span': {'end_timestamp': 10}},
        {'id': 'later', 'context': [], 'reply': ['SEALED_FUTURE'], 'source_span': {'end_timestamp': 200}}]))
    page = _get(live_server, '/dashboard/demo/versions/data/data/index.html?view=fewshot')
    assert 'PUBLIC_PAST' in page and 'SEALED_FUTURE' not in page
    with pytest.raises(HTTPError) as caught:
        _get(live_server, '/instances/demo/data/data/fewshot_pool.jsonl')
    assert caught.value.code == 403


@pytest.mark.parametrize('suffix,status', [
    ('?file=fixed_test.jsonl', 403), ('?view=fixed_test', 403),
    ('?file=../../pointers.json', 403), ('?page=0', 400), ('?page=bad', 400),
])
def test_data_routes_reject_unavailable_content_and_bad_queries(tmp_path, live_server, suffix, status):
    _run(tmp_path)
    with pytest.raises(HTTPError) as caught:
        _get(live_server, '/dashboard/demo/versions/data/data/index.html' + suffix)
    assert caught.value.code == status


@pytest.mark.parametrize('alias', ['persona.md', 'dev_pool.jsonl'])
def test_fixed_answers_cannot_be_read_via_symlink_alias(tmp_path, live_server, alias):
    exp, _ = _run(tmp_path)
    directory = exp.parent.parent / 'data/data'
    _write(directory / 'fixed_test.jsonl', '{"human_reply": ["FIXED_SECRET"]}')
    (directory / alias).symlink_to(directory / 'fixed_test.jsonl')
    suffix = '?file=persona.md' if alias == 'persona.md' else ''
    with pytest.raises(HTTPError) as caught:
        _get(live_server, '/dashboard/demo/versions/data/data/index.html' + suffix)
    assert caught.value.code == 403


def test_unknown_versions_return_404(tmp_path, live_server):
    _run(tmp_path)
    for path in ('generators/missing', 'unknown/base'):
        with pytest.raises(HTTPError) as caught:
            _get(live_server, f'/dashboard/demo/versions/{path}/index.html')
        assert caught.value.code == 404


def test_archive_distinguishes_smoke_snapshots_and_links_unused_versions(tmp_path, live_server):
    exp, _ = _run(tmp_path, smoke=True)
    _write(exp.parent.parent / 'generators/unused/config.json', {'data_version': 'data'})
    page = _get(live_server, '/dashboard/demo/versions/index.html')
    assert '3 个生成器快照' in page
    assert '连接检查 1 次、完整开发评测 0 次、固定验收 0 次' in page
    assert '/versions/generators/unused/index.html' in page
    detail = _get(live_server, '/dashboard/demo/versions/generators/unused/index.html')
    assert '未找到引用此版本的实验记录' in detail


def test_generator_page_separates_metadata_from_behavior_change(tmp_path):
    exp, _ = _run(tmp_path, smoke=True)
    instance = exp.parent.parent
    for ref, data in [('g-0001', 'old-data'), ('g-0002', 'new-data')]:
        _write(instance / 'generators' / ref / 'config.json', {'llm': {'model': 'same'}, 'data_version': data})
        _write(instance / 'generators' / ref / 'persona.md', '同一人格')
    page = version_view.detail_html(instance, 'generators', 'g-0002', {})
    assert '行为配置、人格与场景内容相同的其他快照' in page
    assert '/versions/generators/g-0001/index.html' in page
    assert '生成器快照的数据标记' in page
    assert '相邻编号不代表父子关系' in page
