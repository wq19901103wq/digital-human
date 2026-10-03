"""历史来源还原包：从原始消息重建历史输入（实现见 sources.py）。"""
from .sources import HistorySources, Message, POLICY, _load, digest, load, prompt_case, require, stamp

__all__ = ['HistorySources', 'Message', 'POLICY', 'digest', 'load', 'prompt_case', 'require', 'stamp']
