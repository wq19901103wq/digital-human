"""按需展开一题的实际调用；固定集不经此接口展示。"""
from __future__ import annotations

from .. import tracing
from ..iteration.report import _badge, _details, _e, _pre


def slot(url: str, title: str = "实际提示词与调用过程") -> str:
    return f'<details class="trace-loader" data-trace-url="{_e(url)}"><summary>{_e(title)}</summary><div class="detail-body"><p data-trace-status aria-live="polite">点击展开后加载本题实录</p><div data-trace-body></div></div></details>'


def _request(request: dict) -> str:
    body = ''
    if 'prompt' in request:
        body += _details('实际提示词 · 完整文本', _pre(request['prompt']))
        if request.get('schema'):
            body += _details('要求模型返回的 JSON 结构', _pre(request['schema']))
    wire = request.get('body') or {}
    if wire.get('system'):
        body += _details('实际提示词 · system', _pre(wire['system']))
    for i, message in enumerate(wire.get('messages', []), 1):
        body += _details(f'实际提示词 · {message.get("role", "message")} · {i}', _pre(message.get('content')))
    return body + _details('完整请求参数', _pre(request))


def _event(event: dict, index: int, call_number: int = 0) -> str:
    kind = event['kind']
    data = event.get('data') or {}
    labels = {'retrieval': '风格召回', 'validation': '回复格式校验', 'blind_mapping': '盲测 A/B 对应关系',
              'base_verdict': '大模型初判解析结果', 'features': '抽取特征与解析结果', 'correction': '小模型预测与最终判定'}
    title = labels.get(kind, '模型调用')
    body = ''
    if kind in ('llm', 'codex'):
        request = event['request']
        model = (request.get('body') or request.get('config') or {}).get('model', '')
        purpose = '抽取特征' if request.get('schema') else ('裁判初判' if kind == 'codex' else '模型请求')
        title = f'调用 {call_number} · {purpose} · {model}'
        if request.get('attempt'):
            title += f' · 接口尝试 {request["attempt"]}'
        body += _request(request)
        response = event.get('response')
        if response is not None:
            body += _details('模型返回正文 · 完整文本', _pre(response.get('text', '')))
            body += _details('结束原因、用量与响应信息', _pre({k: v for k, v in response.items() if k != 'text'}))
        else:
            body += '<p class="help">尚无响应正文；请求超时或进程中断时可能没有返回。</p>'
    elif kind == 'correction':
        base = data.get('base') or {}
        body += '<dl>' + ''.join(f'<dt>{_e(k)}</dt><dd>{_e(v)}</dd>' for k, v in [
            ('大模型认为真人在', base.get('human_option')),
            ('小模型预测 A 是真人的概率', f'{data["small_model_probability_a"]:.2%}'),
            ('小模型置信度 / 改判阈值', f'{data["confidence"]:.2%} / {data["threshold"]:.2%}'),
            ('是否改判', '是' if data.get('correction_applied') else '否'),
            ('最终认为真人在', data.get('human_option')),
            ('实际 AI 回复在', data.get('candidate_option')),
            ('是否识别出 AI', '是' if data.get('identified_ai') else '否')]) + '</dl>'
        body += _details('完整判定数据', _pre(data))
    elif kind == 'blind_mapping':
        body += f'<p>真人回复在 {_e(data.get("human_option"))}；待识别的 AI 回复在 {_e(data.get("candidate_option"))}。此对应关系仅用于事后核对，不发送给裁判。</p>' + _pre(data)
    else:
        body += _pre(data)
    if 'error' in event:
        body += '<div class="error-box">' + _pre(event['error']) + '</div>'
    state = {'ok': '已完成', 'failed': '失败', 'running': '等待返回'}.get(event['status'], event['status'])
    if data.get('valid') is False:
        state = '校验未通过'
    elif kind == 'retrieval' and data.get('error'):
        state = '失败，已降级为无示例'
    if kind in ('validation', 'features') and data.get('attempt'):
        title += f' · 第 {data["attempt"]} 次'
    elapsed = f' · {event["elapsed_ms"] / 1000:.2f} 秒' if 'elapsed_ms' in event else ''
    return f'<details data-detail-key="event-{index}"><summary>{_e(title)} · {_e(state + elapsed)}</summary><div class="detail-body">{body}</div></details>'


def payload(exp_dir, ref: str, revision: str = '') -> dict:
    # read() 同时验证 ID 与 realpath containment。
    doc = tracing.read(exp_dir, ref)
    stamp = str(doc['updated_at'])
    response = {'revision': stamp, 'status': doc['status']}
    if stamp == revision:
        return response
    status = {'ok': '本题完成', 'failed': '本题失败', 'running': '记录中', 'interrupted': '执行中断'}.get(doc['status'], doc['status'])
    body = f'<p>{_badge(status)} · 输入在发送前保存；以下为本次调用实录，保留完整提示词、输出正文和结构化结果。</p>'
    body += _details('样本原始字段与来源', _pre(doc['case'])) + _details('本次使用的版本', _pre(doc['versions']))
    if doc.get('previous_trace_ref'):
        body += slot(f'/api/trace/{exp_dir.parent.parent.name}/{exp_dir.name}/{doc["previous_trace_ref"]}', '上次执行中断或重试前的实录')
    if doc['case'].get('generation_trace_ref'):
        body += slot(f'/api/trace/{exp_dir.parent.parent.name}/{exp_dir.name}/{doc["case"]["generation_trace_ref"]}', '本题 AI 样本如何生成 · 冻结评估包实录')
    for index, op in enumerate(doc['operations'], 1):
        branch = '基线' if op['branch'] == 'baseline' else '候选'
        round_name = f'补测第 {op["round"]} 轮' if op['round'] else '初测'
        action = '生成回复' if op['kind'] == 'generation' else '裁判判定'
        state = {'ok': '已完成', 'failed': '失败', 'running': '执行中'}.get(op['status'], op['status'])
        result = _details('本步骤输入', _pre(op['input']))
        if 'result' in op:
            result += _details('本步骤结果', _pre(op['result']))
        if 'error' in op:
            result += '<div class="error-box"><h3>本步骤失败原因</h3>' + _pre(op['error']) + '</div>'
        call_number = 0
        for n, event in enumerate(op['events'], 1):
            if event['kind'] in ('llm', 'codex'):
                call_number += 1
            result += _event(event, index * 1000 + n, call_number)
        elapsed = f' · {op["elapsed_ms"] / 1000:.2f} 秒' if 'elapsed_ms' in op else ''
        body += f'<details data-detail-key="operation-{index}"><summary>{index}. {_e(branch + " · " + round_name + " · " + action + " · " + state + elapsed)}</summary><div class="detail-body">{result}</div></details>'
    response['html'] = body
    return response
