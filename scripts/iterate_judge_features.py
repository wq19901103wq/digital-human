#!/usr/bin/env python3
"""在来源核对过的开发样本上串行比较抽特征模型；不切换生产或开发指针。"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.digital_human.bootstrap.ingest import load_weflow_dir  # noqa: E402
from src.digital_human.config import ConfigError, load_settings, sha256_file  # noqa: E402
from src.digital_human.generator.generator import ReplyGenerator  # noqa: E402
from src.digital_human.iteration import experiment, report, runner, versions  # noqa: E402
from src.digital_human.llm import build_clients  # noqa: E402
from src.digital_human.tracing import CaseTrace  # noqa: E402


def clean_cases(pool, messages):
    """仅纳入一次对方发言后的一次本人回复，避开当前数据与 unread 的边界缺陷。"""
    chats = defaultdict(list)
    for message in messages:
        chats[message['chat_id']].append(message)
    valid, excluded = [], Counter()
    for case in pool:
        context = case.get('context') or []
        if not context or context[-1].get('is_self') is not False:
            excluded['self_boundary'] += 1
            continue
        chat, _, index = case['source_message_id'].rpartition(':')
        index = int(index)
        source = chats.get(chat, [])
        if index < len(context) or index >= len(source):
            excluded['source_missing'] += 1
            continue
        original = source[index - len(context):index]
        if (not source[index]['is_self'] or [source[index]['text']] != case['human_reply']
                or [(m['sender'], m['text'], m['is_self'], m['timestamp']) for m in original]
                != [(m['sender'], m['text'], m['is_self'], m['timestamp']) for m in context]):
            excluded['source_mismatch'] += 1
            continue
        if not 0 <= source[index]['timestamp'] - source[index - 1]['timestamp'] <= 600:
            excluded['late_reply'] += 1
            continue
        if index + 1 < len(source) and source[index + 1]['is_self']:
            excluded['consecutive_self_reply'] += 1
            continue
        if len(context) > 1 and context[-2].get('is_self') is not True:
            excluded['multiple_incoming'] += 1
            continue
        valid.append(case)
    return valid, dict(excluded)


def build_pack(args, study):
    settings = load_settings()
    ptr = versions.load_pointers()
    info = versions.load_generator(ptr['production_gen'])
    data = versions.data_version_dir(ptr['data'])
    pool = [json.loads(s) for s in (data / 'dev_pool.jsonl').read_text().splitlines() if s.strip()]
    valid, exclusions = clean_cases(pool, load_weflow_dir(args.source_exports))
    rng = random.Random(args.seed)
    groups = {kind: [c for c in valid if c['chat_type'] == kind] for kind in ('group', 'private')}
    for values in groups.values():
        rng.shuffle(values)
    counts = {'group': round(args.sample * .75), 'private': args.sample - round(args.sample * .75)}
    if any(counts[k] > len(groups[k]) for k in counts):
        raise ConfigError('边界核对通过的样本不足，不能静默缩小评估包')
    selected = groups['group'][:counts['group']] + groups['private'][:counts['private']]
    rng.shuffle(selected)
    pack_id = 'pack-calibration-features-' + study
    directory = versions.PRIVATE / 'judge_eval' / pack_id
    directory.mkdir(parents=True, exist_ok=True)
    generator = ReplyGenerator(settings, info['config'], build_clients(settings, info['config']['llm']),
                               prompt_root=info['dir'], pool_path=data / 'fewshot_pool.jsonl')
    path = directory / 'building.json'
    building = json.loads(path.read_text()) if path.exists() else {'rows': [], 'selected_ids': [c['case_id'] for c in selected]}
    if building['selected_ids'] != [c['case_id'] for c in selected]:
        raise ConfigError('续建时样本集合发生变化')
    done = {row['case_id'] for row in building['rows']}
    for case in selected:
        if case['case_id'] in done:
            continue
        trace = CaseTrace(directory, case, {'kind': 'judge_pack', 'dataset': 'development', 'baseline_ref': info['id'], 'data_ref': ptr['data']})
        try:
            with trace.operation('generation', 'baseline', 0, {'forced_reply': True}) as op:
                generated = generator.generate(case)
                op['result'] = generated
            trace.finish('ok')
        except BaseException:
            trace.finish('failed')
            raise
        building['rows'].append({**case, 'ai_replies': generated['replies'], 'generation_trace_ref': trace.ref})
        experiment._write_json(path, building)
        print(f'生成评估包 {len(building["rows"])}/{args.sample}', flush=True)
    pack = {'pack_id': pack_id, 'c0_gen_version': info['id'], 'data_ref': ptr['data'],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': building['rows'],
            'source_audit': {'source_total': len(pool), 'eligible': len(valid), 'excluded': exclusions,
                             'rule': '对方单条待回复消息 → 本人单条回复；来源逐字段一致；回复间隔不超过 600 秒',
                             'note': '受限开发子集，用于模型方向筛选；不是全量数据已清洗的声明'},
            'sampling': {'seed': args.seed, **counts},
            'identical_options': sum(r['human_reply'] == r['ai_replies'] for r in building['rows'])}
    experiment._write_json(directory / 'pack.json', pack)
    print(f'冻结评估包 {pack_id}：{len(pack["rows"])} 题；相同文本对 {pack["identical_options"]}', flush=True)
    return pack_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--source-exports', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--study', default=time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--models', nargs='+', default=['gpt-5.6-terra', 'gpt-5.6-sol'])
    parser.add_argument('--effort', default='high')
    args = parser.parse_args()
    if args.sample < 4:
        parser.error('--sample 至少 4')
    versions.switch_instance(args.instance)
    pointers = versions.load_pointers()
    pack_id = 'pack-calibration-features-' + args.study
    pack_path = versions.PRIVATE / 'judge_eval' / pack_id / 'pack.json'
    if not pack_path.exists():
        pack_id = build_pack(args, args.study)
    results = []
    for model in args.models:
        name = model.removeprefix('gpt-5.6-')
        exp_id = f'judge-development-{args.study}-{name}-{args.effort}'
        directory = versions.PRIVATE / 'experiments' / exp_id
        if not directory.exists():
            directory = experiment.create_judge_eval_experiment('development', pack_id,
                {'feature_llm': {'model': model, 'reasoning_effort': args.effort}},
                change=f'{args.sample}题开发抽样校准：仅抽特征升级为 {name}/{args.effort}，初判与小模型保持基线', exp_id=exp_id)
            spec = experiment.spec_of(directory)
            # 记录来源包的哈希，说明本次所有候选使用同一份冻结输入。
            spec['pack_sha256'] = sha256_file(pack_path)
            experiment._write_json(directory / 'spec.json', spec)
            (directory / 'traces').mkdir(exist_ok=True)
            for source in (pack_path.parent / 'traces').glob('*.json'):
                shutil.copy2(source, directory / 'traces' / source.name)
        if experiment.spec_of(directory)['pack_sha256'] != sha256_file(pack_path):
            raise ConfigError('评估包已被改写，拒绝混合结果')
        print(f'开始 {exp_id}', flush=True)
        report.write_run(directory)
        runner.run_judge_experiment(directory)
        state = experiment.state_of(directory)
        results.append({'experiment': exp_id, 'candidate': experiment.spec_of(directory)['candidate_ref'],
                        'feature_model': model, 'effort': args.effort, **state})
        experiment._write_json(pack_path.parent / 'comparison.json', {'pack': pack_id, 'results': results})
        print(json.dumps({'experiment': exp_id, 'verdict': state.get('verdict'), 'metrics': state.get('metrics')}, ensure_ascii=False), flush=True)
        if versions.load_pointers() != pointers:
            raise ConfigError('运行期间基线指针发生变化，停止后续比较')


if __name__ == '__main__':
    main()
