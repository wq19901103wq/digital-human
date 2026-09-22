"""All-categorical, shared-option embedding ranker. No numeric bypass."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from ..config import ConfigError
from . import corrected_v1 as rt


def categories(features, reply, metadata):
    """Fixed raw categories; never consume case IDs, labels or option positions."""
    value = {k: str(v) for k, v in rt.validate_option_features(features).items()}
    text = ''.join(reply)
    # Predefined bins are independent of every training/evaluation answer.
    value['bubble_bin'] = str(min(len(reply), 4))
    value['length_bin'] = str(sum(len(text) >= x for x in (4, 8, 16, 32, 64, 128)))
    observed, names = rt.vectorize_observable_option_booleans(reply, metadata)
    value.update({k: str(bool(v)) for k, v in zip(names[5:], observed[5:])})
    value['group_pattern'] = str(bool(rt.reply_reuses_active_group_pattern(reply, metadata)))
    value['relationship'] = str(metadata.get('relationship') or '<MISSING>')
    value['group_name'] = str(metadata.get('group_name') or '<MISSING>')
    for key in ('latest_message_has_question_mark', 'latest_message_has_at_mention'):
        value[key] = str(bool(metadata.get(key)))
    speakers = list(metadata.get('recent_speakers') or [])[:5]
    speakers += [None] * (5 - len(speakers))
    for i, speaker in enumerate(speakers):
        value[f'speaker_{i}'] = str(speaker or '<MISSING>')
        value[f'speaker_role_{i}'] = 'missing' if speaker is None else ('self' if speaker == '__self__' else 'other')
    for i in range(5):
        for j in range(i):
            value[f'speaker_equal_{j}_{i}'] = str(speakers[i] is not None and speakers[i] == speakers[j])
    return value


def raw_pairs(rows, entries):
    return [tuple(categories(entries[r['case_id']]['features'][side], r['blind'][side], r['metadata'])
                  for side in ('option_A', 'option_B')) for r in rows]


def fit_encoder(pairs):
    """Only caller-supplied fitting rows contribute a vocabulary."""
    rows = [v for pair in pairs for v in pair]
    if not rows or any(set(v) != set(rows[0]) for v in rows):
        raise ConfigError('Embedding categorical fields differ')
    fields = sorted(rows[0])
    return dict(schema=1, embedding_dim=8, numeric_bypass=False, fields=fields,
        vocabularies={key: {token: i + 1 for i, token in enumerate(sorted({r[key] for r in rows}))}
                      for key in fields}, unknown_id=0)


def encode(encoder, pairs):
    if encoder.get('embedding_dim') != 8 or encoder.get('numeric_bypass') is not False:
        raise ConfigError('All fields must use dimension 8 embeddings without numeric bypass')
    fields, vocabs = encoder['fields'], encoder['vocabularies']
    result = np.asarray([[[vocabs[k].get(value[k], 0) for k in fields] for value in pair]
                         for pair in pairs], dtype=np.int64)
    if result.ndim != 3 or result.shape[1:] != (2, len(fields)):
        raise ConfigError('Embedding input must be categorical option pairs')
    return result


class Ranker(nn.Module):
    def __init__(self, encoder, recipe):
        super().__init__()
        if encoder.get('embedding_dim') != 8 or encoder.get('numeric_bypass') is not False:
            raise ConfigError('Embedding dimension or input policy differs')
        self.encoder, self.recipe = encoder, recipe
        self.embeddings = nn.ModuleList([nn.Embedding(len(encoder['vocabularies'][k]) + 1, 8, padding_idx=0)
                                        for k in encoder['fields']])
        width = len(self.embeddings) * 8
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.zeros(width))
                                              for _ in range(recipe['cross_layers'])])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(width))
                                             for _ in range(recipe['cross_layers'])])
        layers, previous = [], width
        for hidden in recipe['hidden']:
            layers.append(nn.Linear(previous, hidden))
            normalization = recipe['normalization']
            if normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden))
            elif normalization == 'layer':
                layers.append(nn.LayerNorm(hidden))
            elif normalization != 'none':
                raise ConfigError('Unknown embedding ranker normalization')
            layers.append({'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}[recipe['activation']]())
            layers.append(nn.Dropout(recipe['dropout']))
            previous = hidden
        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(previous + (width if recipe['cross_layers'] else 0), 1, bias=False)

    def score(self, x):
        if x.dtype != torch.long or x.ndim != 2 or x.shape[1] != len(self.embeddings):
            raise ConfigError('Ranker accepts categorical IDs only')
        embedded = torch.cat([layer(x[:, i]) for i, layer in enumerate(self.embeddings)], dim=1)
        deep = self.deep(embedded)
        if self.cross_weights:
            crossed = embedded
            for weight, bias in zip(self.cross_weights, self.cross_biases):
                crossed = embedded * (crossed @ weight)[:, None] + bias + crossed
            deep = torch.cat((deep, crossed), dim=1)
        return self.output(deep).squeeze(1)

    def forward(self, pairs):
        # Both options share parameters AND one BN batch. Evaluation freezes BN.
        scores = self.score(pairs.reshape(-1, pairs.shape[-1])).reshape(-1, 2)
        return scores[:, 0] - scores[:, 1]


def predict(model, pairs):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(pairs, dtype=torch.long))).numpy()


def metrics(probability, labels):
    p = np.clip(probability.astype(float), 1e-7, 1 - 1e-7)
    return dict(correct=int(np.sum((p >= .5) == labels)), total=len(labels),
                logloss=float(-np.mean(labels * np.log(p) + (1-labels) * np.log(1-p))))


def fit(encoder, recipe, pairs, labels, train_indices, validation_indices, seed, epochs=None):
    """Deterministic CPU fit. Full refit gets fixed epochs, never dev labels."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = Ranker(encoder, recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'],
                                 weight_decay=recipe['weight_decay'])
    x = torch.as_tensor(pairs, dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_loss, stale = None, 0, math.inf, 0
    limit = int(epochs or recipe['max_epochs'])
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(train_indices)
        for offset in range(0, len(order), recipe['batch_size']):
            ix = order[offset:offset + recipe['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[ix]), y[ix])
            if not torch.isfinite(loss):
                raise ConfigError('Nonfinite embedding training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if len(validation_indices):
            result = metrics(predict(model, pairs[validation_indices]), labels[validation_indices])
            if result['logloss'] < best_loss - 1e-5:
                best_loss, best_epoch, stale = result['logloss'], epoch, 0
                best = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= recipe['patience']:
                break
    if best is not None:
        model.load_state_dict(best)
    else:
        best_epoch = limit
    result = metrics(predict(model, pairs[validation_indices]), labels[validation_indices]) if len(validation_indices) else None
    return model, dict(seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                       validation=result, parameters=sum(p.numel() for p in model.parameters()))


def document(model):
    return dict(schema=1, kind='categorical_embedding_ranker_v1',
        scoring='sigmoid(score(A)-score(B))', encoder=model.encoder, recipe=model.recipe,
        torch_version=torch.__version__, weights={k: v.tolist() for k, v in model.state_dict().items()})


def restore(value):
    if (value.get('kind') != 'categorical_embedding_ranker_v1'
            or value.get('scoring') != 'sigmoid(score(A)-score(B))'
            or value.get('torch_version') != torch.__version__):
        raise ConfigError('Embedding model schema or runtime differs')
    model = Ranker(value['encoder'], value['recipe'])
    template = model.state_dict()
    model.load_state_dict({k: torch.tensor(v, dtype=template[k].dtype) for k, v in value['weights'].items()})
    model.eval()
    return model
