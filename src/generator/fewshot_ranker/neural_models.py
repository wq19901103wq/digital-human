"""Few-shot neural scorers, including a small dense RankMixer adaptation.

RankMixer follows https://arxiv.org/abs/2507.15551: parameter-free multi-head
token permutation, independent per-token FFNs, post-residual LayerNorm and
mean pooling. Domain-group projections adapt the architecture to this dataset;
there is no sparse MoE or claim to reproduce the industrial-scale model.
"""
import torch
from torch import nn

from ...judge import embedding
from ..history_sources import require


def token_mix(x):
    """H=T: exchange token and head axes, preserving [batch, tokens, width]."""
    batch, tokens, width = x.shape
    require(width % tokens == 0, 'RankMixer token width must be divisible by token count')
    return x.reshape(batch, tokens, tokens, width//tokens).transpose(1, 2).reshape_as(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens, width, expansion, dropout):
        super().__init__()
        self.mix_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(width, width*expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width*expansion, width), nn.Dropout(dropout)) for _ in range(tokens)])

    def forward(self, x):
        x = self.mix_norm(x + token_mix(x))
        transformed = torch.stack([ffn(x[:, i]) for i, ffn in enumerate(self.ffns)], dim=1)
        return self.ffn_norm(x + transformed)


def token_groups(fields):
    # Three independent views plus deterministic interactions. No target reply.
    prefixes = ('target.', 'example_context.', 'example_reply.')
    groups = [[i for i, key in enumerate(fields) if key.startswith(prefix)] for prefix in prefixes]
    groups.append([i for i, key in enumerate(fields) if not key.startswith(prefixes)])
    require(all(groups), 'RankMixer requires target, example context/reply and interaction fields')
    return groups


class RankMixer(nn.Module):
    def __init__(self, encoder, recipe):
        super().__init__()
        require(encoder['embedding_dim'] == recipe['embedding_dim'] == 8 and
                encoder['numeric_bypass'] is False, 'Only categorical embedding8 inputs are supported')
        self.encoder, self.recipe = encoder, recipe
        self.groups = token_groups(encoder['fields'])
        width = recipe['token_width']
        require(width > 0 and width % len(self.groups) == 0, 'Invalid RankMixer token width')
        self.embeddings = nn.ModuleList([nn.Embedding(len(encoder['vocabularies'][key])+1, 8,
            padding_idx=0) for key in encoder['fields']])
        self.projections = nn.ModuleList([nn.Linear(8*len(group), width) for group in self.groups])
        self.blocks = nn.Sequential(*[MixerBlock(len(self.groups), width,
            recipe['ffn_expansion'], recipe['dropout']) for _ in range(recipe['mixer_layers'])])
        layers, previous = [], width
        for hidden in recipe['hidden']:
            layers.append(nn.Linear(previous, hidden))
            if recipe['normalization'] == 'batch':
                layers.append(nn.BatchNorm1d(hidden))
            elif recipe['normalization'] == 'layer':
                layers.append(nn.LayerNorm(hidden))
            else:
                require(recipe['normalization'] == 'none', 'Unknown head normalization')
            layers.extend([nn.SiLU(), nn.Dropout(recipe['dropout'])])
            previous = hidden
        self.head = nn.Sequential(*layers, nn.Linear(previous, 1, bias=False))

    def score(self, x):
        require(x.dtype == torch.long and x.ndim == 2 and x.shape[1] == len(self.embeddings),
                'RankMixer accepts categorical IDs only')
        embedded = [layer(x[:, i]) for i, layer in enumerate(self.embeddings)]
        tokens = torch.stack([projection(torch.cat([embedded[i] for i in group], dim=1))
            for group, projection in zip(self.groups, self.projections)], dim=1)
        return self.head(self.blocks(tokens).mean(dim=1)).squeeze(1)

    def forward(self, pairs):
        scores = self.score(pairs.reshape(-1, pairs.shape[-1])).reshape(-1, 2)
        return scores[:, 0]-scores[:, 1]


def create(encoder, recipe):
    architecture = recipe.get('architecture', 'dnn')
    require(architecture in ('dnn', 'rankmixer'), 'Unknown neural architecture')
    return RankMixer(encoder, recipe) if architecture == 'rankmixer' else embedding.Ranker(encoder, recipe)


def document(model):
    value = embedding.document(model)
    if isinstance(model, RankMixer):
        value['kind'] = 'fewshot_dense_rankmixer_v1'
        value['token_fields'] = [[model.encoder['fields'][i] for i in group] for group in model.groups]
    return value


def restore(value):
    if value['kind'] == 'categorical_embedding_ranker_v1':
        return embedding.restore(value)
    require(value['kind'] == 'fewshot_dense_rankmixer_v1' and
            value['scoring'] == 'sigmoid(score(A)-score(B))' and
            value['torch_version'] == torch.__version__, 'RankMixer schema or runtime differs')
    model = RankMixer(value['encoder'], value['recipe'])
    require(value['token_fields'] == [[model.encoder['fields'][i] for i in group]
            for group in model.groups], 'RankMixer token fields differ')
    template = model.state_dict()
    model.load_state_dict({key: torch.tensor(v, dtype=template[key].dtype)
                          for key, v in value['weights'].items()})
    model.eval()
    return model
