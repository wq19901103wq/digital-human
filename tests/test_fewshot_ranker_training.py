from copy import deepcopy
import json

import numpy as np
import pytest

import pytest
pytest.importorskip('torch')  # 可选重依赖
pytest.importorskip('xgboost')  # 可选重依赖
from src.config import ConfigError
from src.generator.fewshot_ranker import extraction, features, schema, training
from src.iteration.storage import read_json, write_json
from src.judge.embedding import fit_encoder, restore


def record(i=1):
    context = [dict(sender='person', text='hello', is_self=False, timestamp=i*100),
               dict(sender='person', text='question?', is_self=False, timestamp=i*100+10)]
    return dict(context=context, context_messages=deepcopy(context), reply=['yes', 'detail'],
        human_reply=['TARGET_SECRET'], generated_reply=['GENERATED_SECRET'], z=1,
        chat_type='private', relationship='private', chat_name='chat', source_chat_id='chat',
        input_cutoff={'timestamp': i*100+10}, source_span={'end_timestamp': i*100+20},
        context_message_ids=[f'c{i}'], reply_message_ids=[f'a{i}'])


def test_independent_views_keep_all_bubbles_and_reject_future_context():
    target, example = record(10), record()
    view = features.context_view(target)
    text = json.dumps(view)
    assert 'TARGET_SECRET' not in text and 'GENERATED_SECRET' not in text and 'z' not in view
    assert len(view['messages']) == 2
    assert features.reply_view(example)['reply'] == ['yes', 'detail']
    assert 'reply' not in features.context_view(example, example=True)
    changed = deepcopy(target)
    changed['human_reply'] = ['changed']
    changed['z'] = 0
    assert features.context_view(changed) == view
    changed['context'][0]['timestamp'] = 10000
    with pytest.raises(ConfigError, match='cutoff'):
        features.context_view(changed)


def test_content_cache_reuses_successes_and_invalidates_changed_requests(tmp_path):
    class Client:
        calls = 0
        def cache_identity(self):
            return {'model': 'test', 'effort': 'low'}
        def run(self, prompt, output_schema):
            self.calls += 1
            return json.dumps({k: 'unknown' for k in output_schema['properties']})
    client = Client()
    view = features.context_view(record())
    key, request = extraction.task('context', view, client.cache_identity())
    first = extraction.extract_one(tmp_path, key, request, client)
    assert extraction.extract_one(tmp_path, key, request, client) == first
    assert client.calls == 1
    assert extraction.task('context', {**view, 'chat_name': 'different'}, client.cache_identity())[0] != key
    assert extraction.task('context', view, {'model': 'other'})[0] != key
    mutated = read_json(tmp_path / (key+'.json'))
    mutated['features']['scene'] = 'work'
    write_json(tmp_path / (key+'.json'), mutated)
    with pytest.raises(ConfigError, match='binding'):
        extraction.cached(tmp_path, key, request)
    with pytest.raises(ConfigError):
        schema.validate('context', {'scene': 'unknown'})


def test_deduplication_and_categorical_crosses():
    target, example = record(10), dict(record(), id='e')
    groups = [dict(target_id='t', target=target, candidates=[{'example': example}]),
              dict(target_id='t2', target=deepcopy(target), candidates=[{'example': example}])]
    tasks, refs = extraction.prepare(groups, {'model': 'test'})
    assert len(tasks) == 3 and refs[('t', 'e')] == refs[('t2', 'e')]
    values = {key: {k: 'unknown' for k in request['schema']['properties']} for key, request in tasks.items()}
    rows = training.assemble(groups, refs, values)
    assert rows[0] == rows[1] and all(isinstance(v, str) for v in rows[0].values())
    assert rows[0]['cross.needs_intersection'] == 'unknown'
    assert rows[0]['cross.needs_jaccard'] == 'unknown'
    assert not any('label' in k or 'target_id' in k or 'example_id' in k for k in rows[0])
    future = deepcopy(example)
    future['source_span']['end_timestamp'] = target['input_cutoff']['timestamp']
    with pytest.raises(ConfigError, match='strictly earlier'):
        features.combine(target, future, {**values[refs[('t','e')][0]], **features.context_local(target)},
            {**values[refs[('t','e')][1]], **features.context_local(example, example=True)},
            {**values[refs[('t','e')][2]], **features.reply_local(example)})


def synthetic():
    groups, rows, obs = [], [], []
    for i in range(1, 16):
        split = 'fit' if i <= 10 else 'validation'
        group = dict(target_id=str(i), split=split, target=record(i), candidates=[], indices=[])
        for z in (0, 1):
            group['indices'].append(len(obs))
            rows.append({'quality': str(z), 'identity': str(i)})
            obs.append(dict(target_id=str(i), example_id=str(z), z=z, split=split, recall_rank=2-z))
        groups.append(group)
    return groups, rows, obs


def test_inner_time_split_and_fit_only_vocabulary():
    groups, rows, obs = synthetic()
    groups[0]['target']['context_message_ids'] = ['a9']
    fit, valid, report = training.inner_split(groups, .2)
    assert [g['target_id'] for g in valid] == ['9', '10']
    assert report['purged_contexts'] == ['1']
    assert all(g['split'] == 'fit' for g in fit+valid)
    encoder = fit_encoder([(rows[i], rows[i]) for g in fit for i in g['indices']])
    x = training.encode_rows(encoder, rows)
    assert x[-1, encoder['fields'].index('identity')] == 0
    assert encoder['embedding_dim'] == 8 and encoder['numeric_bypass'] is False


def test_pair_weights_equalize_contexts_and_metrics_handle_ties():
    obs = [dict(target_id='a', z=z) for z in (1, 0, 0)] + [dict(target_id='b', z=z) for z in (1, 0)]
    groups = [dict(indices=[0,1,2]), dict(indices=[3,4])]
    pairs, weights, mixed = training.pairs(groups, obs)
    assert pairs.tolist() == [[0,1], [0,2], [3,4]]
    assert weights.tolist() == [.5,.5,1] and mixed == 2
    perfect = training.metrics(obs, [1,0,0,1,0])
    assert perfect['roc_auc'] == perfect['group_auc_macro'] == perfect['ndcg_at_3'] == 1
    ties = training.metrics(obs, [0]*5)
    assert ties['roc_auc'] == ties['group_auc_macro'] == ties['pair_accuracy'] == .5
    assert ties['pr_auc_average_precision'] == .4
    assert ties['top1_positive_yield'] == pytest.approx((1/3+1/2)/2)
    negative = training.metrics([dict(target_id='c', z=0)], [0])
    assert negative['ndcg_contexts'] == 0 and negative['top1_positive_yield'] == 0


def test_train_serialization_and_outer_labels_do_not_change_weights(tmp_path):
    groups, rows, obs = synthetic()
    recipe = dict(training.RECIPE, max_epochs=3, patience=2, hidden=[8], dropout=0)
    report = training.train(groups, obs, rows, tmp_path / 'first', recipe)
    changed = deepcopy(obs)
    for row in changed:
        if row['split'] == 'validation':
            row['z'] = 1-row['z']
    training.train(groups, changed, rows, tmp_path / 'second', recipe)
    saved = read_json(tmp_path / 'first/model.json')
    assert saved == read_json(tmp_path / 'second/model.json')
    model = restore(saved)
    predicted = training.scores(model, training.encode_rows(model.encoder, rows))
    stored = read_json(tmp_path / 'first/predictions.json')['rows']
    np.testing.assert_allclose(predicted, [r['logit'] for r in stored])
    assert report['metrics']['validation']['dnn']['observations'] == 10
