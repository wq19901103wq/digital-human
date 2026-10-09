"""Share bindings and read-only knowledge browsing use the frozen model configuration."""
import html
from urllib.error import HTTPError

import pytest

from src.dashboard import report
from test_live_dashboard import live_server as _live_server_fixture
from test_report import _run, _write
from test_version_view import _get

live_server = _live_server_fixture


def _share(instance, ref='s-0001'):
    directory = instance / 'share' / ref
    _write(directory / 'manifest.json', {'files': {'knowledge.json': 'digest', 'README.md': 'digest'},
                                       'sha256': 'share-digest', 'runtime_usable': False})
    _write(directory / 'content/knowledge.json', {
        'pages': [{'id': f'p-{i}', 'title': f'人物-{i}'} for i in range(23)],
        'fact_claims': [{'id': 'fact-1', 'title': '人物归属', 'page_ids': ['p-0'],
                         'description': '关系 <script>bad()</script>', 'evidence_refs': ['raw-1']}],
        'evidence': {'raw-1': {'id': 'legacy-id', 'message_id': 'source-message', 'line': 42}}})
    _write(directory / 'content/README.md', '分享资料 <script>bad()</script>')
    return directory


def test_model_bindings_are_independent_and_share_addition_refreshes_baselines(tmp_path, monkeypatch):
    from src.iteration import shares, versions

    exp, _ = _run(tmp_path)
    instance = exp.parent.parent
    monkeypatch.setattr(versions, 'PRIVATE', instance)
    source = tmp_path / 'share-source'
    pins = {}
    for index in range(1, 4):
        _write(source / 'README.md', f'Synthetic knowledge {index}')
        ref = shares.create(source)
        pins[ref] = shares.binding(ref)
    _write(instance / 'pointers.json', {'production_gen': 'base', 'iteration_gen': 'candidate',
                                      'production_judge': 'base', 'iteration_judge': 'candidate'})
    for folder, ref, share in [('generators', 'base', 's-0001'), ('generators', 'candidate', 's-0002'),
                               ('judges', 'candidate', 's-0003')]:
        _write(instance / folder / ref / 'config.json', pins[share])
    from src.dashboard.share_view import current_bindings
    assert current_bindings(instance, 's-0001') == ['生产 gen base']
    assert current_bindings(instance, 's-0002') == ['开发 gen candidate']
    assert current_bindings(instance, 's-0003') == ['开发 judge candidate']
    payload = report.live_payload(instance)
    page = payload['regions']['configuration']
    for ref in ['s-0001', 's-0002', 's-0003']:
        assert f'/share/{ref}/index.html' in page
    assert page.count('Share：<span class="help">未绑定</span>') == 1
    assert 'configuration' not in report.live_payload(instance, config_revision=payload['config_revision'])['regions']
    _share(instance, 's-0004')
    assert '/share/s-0004/index.html' in report.live_payload(
        instance, config_revision=payload['config_revision'])['regions']['configuration']


def test_share_routes_show_records_sources_search_and_paginate_without_mutation(tmp_path, live_server):
    exp, _ = _run(tmp_path)
    directory = _share(exp.parent.parent)
    before = (directory / 'content/knowledge.json').read_bytes()
    base = '/dashboard/demo/versions/share/s-0001/index.html'
    page = _get(live_server, base)
    assert '审阅材料' in page and '当前基线绑定：无' in page
    assert 'href="/instances/demo/share/s-0001/content/knowledge.json" download' in page
    assert _get(live_server, '/instances/demo/share/s-0001/content/knowledge.json').encode() == before
    assert '人物-19' in page and '人物-20' not in page
    assert '人物归属' in page and '&lt;script&gt;bad()&lt;/script&gt;' in page
    second = _get(live_server, base + '?page=2')
    assert second.count('class="version-record"') == 3 and '人物-22' in second
    result = _get(live_server, base + '?view=evidence&q=raw-1')
    assert 'source-message' in result and '共 1 条' in result
    preview = _get(live_server, base + '?file=content/README.md')
    assert '分享资料 <script>bad()</script>' in html.unescape(preview)
    assert '<script>bad()</script>' not in preview
    assert (directory / 'content/knowledge.json').read_bytes() == before
    assert '/versions/share/s-0001/index.html' in _get(live_server, '/dashboard/demo/versions/index.html')


def test_structured_share_resolves_people_and_searches_event_participants(tmp_path, live_server):
    exp, _ = _run(tmp_path)
    directory = _share(exp.parent.parent)
    _write(directory / 'content/knowledge.json', {
        'schema': 'wiki_structured_v1', 'subject': {'self_account': 'self'},
        'entities': [{'id': 'e0', 'label': '小林', 'account': 'person'},
                     {'id': 'e1', 'label': '小周', 'account': 'self'}],
        'attributes': [{'subject_id': 'e0', 'field': 'name', 'value': '林某'}],
        'addresses': [{'speaker_id': 'e1', 'target_id': 'e0', 'term': '阿林', 'usage': 'direct'}],
        'events': [{'event_type': '出行', 'description': '计划去海边', 'status': 'planned',
                    'participants': [{'entity_id': 'e0', 'role': '组织者'}],
                    'details': [{'field': '地点', 'value': '海边 <某处>'}]}]})
    base = '/dashboard/demo/versions/share/s-0001/index.html'
    page = _get(live_server, base)
    assert '人物属性 · 1' in page and '<strong>主体：</strong>小林' in page and '林某' in page
    addresses = _get(live_server, base + '?view=addresses')
    assert '<strong>说话人：</strong>小周（本人）' in addresses
    assert '<strong>称呼对象：</strong>小林' in addresses and '直接称呼' in addresses
    events = _get(live_server, base + '?view=events&q=%E5%B0%8F%E6%9E%97')
    assert '共 1 条' in events and '<strong>组织者：</strong>小林' in events
    assert '海边 &lt;某处&gt;' in events and '计划中' in events


@pytest.mark.parametrize('suffix,status', [('?file=content/secret.md', 403),
    ('?file=../../pointers.json', 403), ('?page=0', 400), ('?view=unknown', 400)])
def test_share_rejects_unlisted_files_and_invalid_queries(tmp_path, live_server, suffix, status):
    exp, _ = _run(tmp_path)
    directory = _share(exp.parent.parent)
    _write(directory / 'content/secret.md', 'UNLISTED')
    with pytest.raises(HTTPError) as caught:
        _get(live_server, '/dashboard/demo/versions/share/s-0001/index.html' + suffix)
    assert caught.value.code == status


@pytest.mark.parametrize('escape', ['file_symlink', 'directory_symlink', 'traversal'])
def test_share_manifest_cannot_escape_snapshot(tmp_path, live_server, escape):
    exp, _ = _run(tmp_path)
    directory = _share(exp.parent.parent)
    if escape == 'file_symlink':
        (directory / 'content/alias.json').symlink_to(exp.parent.parent / 'pointers.json')
        filename = 'alias.json'
    elif escape == 'directory_symlink':
        (directory / 'content/outside').symlink_to(exp.parent.parent, target_is_directory=True)
        filename = 'outside/pointers.json'
    else:
        filename = '../../pointers.json'
    _write(directory / 'manifest.json', {'files': {filename: 'digest'}})
    with pytest.raises(HTTPError) as caught:
        _get(live_server, '/dashboard/demo/versions/share/s-0001/index.html')
    assert caught.value.code == 403
