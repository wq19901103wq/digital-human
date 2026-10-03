"""冻结边界守护：ranker SERVING_RUNTIME 文件字节与 pin 一致性（learned_sources.serving_runtime 原始 sha256 绑定）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# g-0029..g-0032 的 ranker provenance 以原始 sha256 钉死这组推理实现文件
# （learned_sources.SERVING_RUNTIME，不经过 cache._CODE_DIGEST_ALIASES）。
# 任何字节变化都会使现有生成器的 serving 核验失败——重构时禁止修改。
SERVING_RUNTIME_SHA256 = {
    '__init__': '67ed902559ace50963f2353ab1442155153121ec13bada7ebbc306e2c7b1a0ae',
    'features': 'ed11c437ad8806aa6456b202c66067d667557316087653dca6b50940b6bfc371',
    'schema': 'fc3ff6d64a6006507ca7c0c45728d8c1669f02ad7ffbeede7565c563d6595714',
    'extraction': 'eede9fad6434bab744a5e0ac984ec20c5cb38823b36d69e570701c904fc57aa5',
    'crosses': '3e96a90d59ab457f5104342b350dae2cfeb1ebfa21881e12576c4a4a15b1cbd1',
    'identity_crosses': 'b994ee9814e3138dc8bc4950a82a98868322d34bcbbae72ef591ccf8d5aa0eca',
    'boosting': 'f9818dd560059bf393ededee91819fcf2280a1ced68718d11129b347515e9862',
    'action_crosses': '5d8f2bab2e0b4377ddeb0f69306cbd893b3b96fe2ce130ae8d52b0053b8193fe',
    'concern': '2567b64ada4d50b142eac75aea04bad3f9bd8060bdf8dbb0fbb8a1d38c3e6d48',
    'supplemental': '5e2179f987e8c33859b133800dc5833207354dec8f543f592d55a306dc83f500',
}


def frozen_violations() -> list[str]:
    problems = []
    for name, expected in SERVING_RUNTIME_SHA256.items():
        path = ROOT / 'src' / 'generator' / 'fewshot_ranker' / (name + '.py')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f'{name}.py 字节变化：{actual[:12]}… ≠ pin {expected[:12]}…')
    return problems
