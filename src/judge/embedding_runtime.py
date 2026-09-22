"""Frozen categorical embedding scorer on the shared feature replay channel."""
import json
from pathlib import Path

from ..config import ConfigError
from .lr_retrain import LRJudge
from . import embedding as net


class EmbeddingJudge(LRJudge):
    decision_policy = 'embedding_only'

    def __init__(self, config, version_dir, *args, **kwargs):
        name = config.get('embedding_file')
        if (config.get('decision_policy') != self.decision_policy or not isinstance(name, str)
                or Path(name).name != name or name not in config.get('assets', {})):
            raise ConfigError('Embedding scorer must be a bound frozen asset')
        super().__init__(config, version_dir, *args, **kwargs)
        net.torch.set_num_threads(1)
        self.network = net.restore(json.loads((Path(version_dir) / name).read_text()))
        self._check_sources()

    def probability_a(self, features, blind, metadata):
        pair = tuple(net.categories(features[k], blind[k], metadata)
                     for k in ('option_A', 'option_B'))
        encoded = net.encode(self.network.encoder, [pair])
        # The shared scorer must tie identical inputs, including OOV collisions.
        # Float32 GEMM can otherwise round identical rows differently at 0.5.
        if (encoded[0, 0] == encoded[0, 1]).all():
            return 0.5
        return float(net.predict(self.network, encoded)[0])
