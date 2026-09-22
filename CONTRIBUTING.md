# 贡献指南

感谢贡献！请遵守以下约定：

1. **校验门禁**：提交前运行 `python scripts/check.py code --staged`；
   发布相关改动额外运行 `python scripts/check_public.py --history` 必须 0 findings。
2. **边界规则**：`src/` 不得 import `scripts/`/`instances/`/`private`
   （含 `__import__`/`importlib` 动态导入，扫描器强制）；
   `instances/`、`.env`、`data/`、`runs/` 永不入库（见 .gitignore）。
3. **测试**：新增机制行为必须带回归测试（tests/，使用合成数据/mocks）。
4. **历史纪律**：版本与证据 write-once，改动走新版本，不修改已冻结内容。
