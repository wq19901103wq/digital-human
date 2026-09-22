# digital-human

> **2026-09-18 已合并为单一仓库**：原 `digital-human-framework` 的通用机制已并入本仓库（代码在 `src/`，import 前缀 `src.`）；私有边界由 `.gitignore` 保证（`instances/`、`.env` 等不进 git）。本仓库保持私有；将来如需公开，从干净状态重新导出一份新历史。

对抗式（GAN-like）文字数字人系统：生成器与 Judge 互相迭代，唯一优化目标是 **Judge 盲测下更像真人**。移植自 wechat-mac 的对抗 workflow，核心架构为**四对象模型**。

## 架构：四对象，三指针

| 对象 | 存储 | 职责 |
|------|------|------|
| 数据版本 `d-XXXX` | `private/data/`（不可变） | 切分结果、few-shot 池、人格工作区、manifest |
| 生成器版本 `g-XXXX` | `private/generators/`（不可变） | 行为配置 + persona/场景**自包含快照** |
| Judge 版本 `j-XXXX` | `private/judges/`（不可变） | 判别配置 + 模型 + 训练来源 meta |
| 实验 | `private/experiments/<id>/` | 唯一执行状态源：spec/state/cases/账单 |

`private/pointers.json` 只有四个字段（data / production_gen / iteration_gen / judge）。
**版本创建后只读，晋升只是切指针**——没有需要互相同步的多份状态。

三层 git 边界：
- **机制层**（提交）：`src/`、`prompts/*.template.md`、`config/settings.yaml`（超参）、`scripts/`、`docs/`、测试
- **版本层 + 数据层**（gitignore）：`private/` 全部

## 流程一句话

```
bootstrap（数据→数据版本+初始版本+指针）
→ 实验（开发 A/B：iteration vs iteration+改动；固定测试：production vs iteration）
→ 晋升（promote.py：净胜≥10/1000 进开发指针；固定双条件进生产指针）
→ Judge 环（train 候选 → 同包比较 → 达标转正）
```

## Quickstart

连接配置可以写入项目根目录的 `.env`，命令行会自动读取；已显式设置的环境变量优先。
Coding Plan 通道使用 `https://ark.cn-beijing.volces.com/api/coding`，
客户端走 Anthropic 协议。其模型名和密钥不能直接套用普通 `/api/v3` 端点。

```bash
pip install -r requirements.txt
export DH_LLM_BASE_URL=... DH_LLM_API_KEY=...
cp your_export.jsonl data/chat_export.jsonl        # 统一 jsonl，schema 见 ingest.py

python scripts/bootstrap.py --data data/chat_export.jsonl --model <生成模型>
python scripts/run.py --dataset development --change "few-shot 开启 X" \
    --override retriever.X=true --limit 5           # 冒烟
python scripts/run.py --dataset development --change "…" --override retriever.X=true
python scripts/promote.py gen --exp <实验id>        # 通过则开发基线前进
python scripts/serve_dashboard.py                 # http://localhost:8080/dashboard/
```

合成示例数据冒烟：`cp examples/chat_export.sample.jsonl data/chat_export.jsonl`。
规则与约束见 **docs/SOP.md**（5 节），文件格式与状态机见 **docs/IMPLEMENTATION.md**。
跨实验结果复用、并发请求去重和历史查询见 [实验缓存说明](docs/RESULT-CACHE.md)。
内部研究待办见 [人物介绍与提示词恢复实验备忘](docs/PERSONA-RESTORATION-2026-09-22.md)：保存旧人设入口、移除原因、当前提示词结构及后续恢复的实验条件。
