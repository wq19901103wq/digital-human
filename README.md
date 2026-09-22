# Digital Human · 对抗式文字数字人迭代框架

让 AI 生成器在**盲测对抗**中迭代得越来越像真人：生成器与盲测 Judge 互相较量，
唯一优化目标是 **Judge 在固定测试题上把 AI 回复识别出来的比例（越低越像人）**。

## 为什么是这套框架

- **对抗式迭代**：不是一次性评测，而是"生成器版本 ↔ Judge 版本"交替升级的
  受控循环，每轮只改一个方向，改动必须可证伪。
- **证据优先**：实验的证据（逐题记录）与结论（指标/裁决）分层存储，版本
  write-once、晋升只切指针，任何结论可从冻结证据逐题重算。
- **按侧测量复用**：judge 对每侧的判定是绝对测量，测量库让同条件对比零成本
  复现、补轮只跑缺失题（见 `docs/DESIGN-SIDE-MEASUREMENT-BANK.md`）。
- **防刷固定集**：开发轮免费反复试，固定验收批次一次性消耗，晋升多层把关。

## 5 分钟跑通（冒烟）

```bash
pip install -r requirements.txt
cp examples/chat_export.sample.jsonl data/chat_export.jsonl   # 合成示例数据
export DH_LLM_BASE_URL=<你的 OpenAI 兼容端点> DH_LLM_API_KEY=<你的密钥>

python scripts/bootstrap.py --data data/chat_export.jsonl --model <生成模型>
python scripts/run.py --dataset development --change "冒烟接线验证" --limit 2
python scripts/serve_dashboard.py --port 8080   # http://localhost:8080/dashboard/
```

## 目录

| 路径 | 职责 |
|---|---|
| `src/` | 机制层（bootstrap 数据构建 / generator 生成 / judge 盲测 / iteration 迭代引擎 / dashboard） |
| `scripts/` | 命令入口（bootstrap/run/promote/evaluate_judge/iterate_branches/check.py 统一校验） |
| `docs/` | SOP、架构、数据契约、测量库设计 |
| `tests/` | 回归测试 |
| `prompts/` `config/` | 提示词模板与超参模板 |
| `examples/` | 合成示例数据（与任何真实个人无关） |

## 文档导航

- 流程与约束：`docs/SOP.md`；文件格式与状态机：`docs/IMPLEMENTATION.md`
- 架构与数据模型：`docs/ARCHITECTURE.md`、`docs/DATA_MODEL.md`
- 多分支并行迭代：`docs/ASYNC-ITERATION.md`；运维：`docs/OPERATIONS.md`
- 发布与边界：`scripts/check_public.py`（贡献门禁）、`scripts/scrub_sensitive.py`

## 许可证

MIT（见 LICENSE）。示例数据为合成样本，与任何真实个人无关。
