"""兼容层：服务 Fewshot 冻结特征文件与 legacy 脚本的 import，新代码一律用 src.data.history。"""
from src.data import conversations
from src.data.ingest import message_identity
from src.infrastructure.config import sha256_file
from src.data.history.sources import HistorySources
from src.data.history.sources import Message
from src.data.history.sources import POLICY
from src.data.history.sources import _load
from src.data.history.sources import digest
from src.data.history.sources import load
from src.data.history.sources import prompt_case
from src.data.history.sources import require
from src.data.history.sources import stamp

__all__ = ['HistorySources', 'Message', 'POLICY', 'conversations', 'digest', 'load', 'message_identity',
           'prompt_case', 'require', 'sha256_file', 'stamp']
