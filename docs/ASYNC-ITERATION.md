# 多分支异步迭代

每个方向有自己的候选与评测进度。有一个方向通过固定验收，就推进全量；其余方向将自己的改动
重新应用到新全量上，再验证是否仍有增益。配置、提示词与冻结的模型资产都可以成为实验改动。
这里的全量是 `instances/<实例>/pointers.json` 中的 production 版本；不包含线上流量分桶或部署。

（验收记录留在内部仓库）：两个真实 worker 并行、两次达标推全、
其他分支跟随新全量重评。模型输出为隔离模拟，评分和采用规则沿用现有实现。

```text
全量 P0 ───── 实验 A：冒烟 → 开发 → 固定通过 ───── 全量 P1
       └──── 实验 B：基于 P0 运行                  │
                  保留旧结果，重放 B 的改动 ──────┘
                              ↓
                  B 基于 P1：冒烟 → 开发 → 固定 → P2
```

## 提交和启用

以下命令在仓库根目录执行。`python` 使用已安装项目依赖的 Python 环境。
示例参数用于说明接口，不代表已验证的候选方案。

```bash
# 两个独立生成器方向；提交只冻结提案，不调用模型、不推全。
python scripts/iterate_branches.py --instance example-agent submit \
  --name gen-model --kind gen --change '验证候选模型对生成器识别率的影响' \
  --overrides '{"llm":{"model":"<候选模型>"}}'

python scripts/iterate_branches.py --instance example-agent submit \
  --name gen-budget --kind gen --change '验证示例字符预算的影响' \
  --overrides '{"shots_char_budget":12000}'

# Judge 提案必须提供生产生成器生成的开发包和固定验证包。
python scripts/iterate_branches.py --instance example-agent submit \
  --name judge-model --kind judge --change '验证候选 Judge 的识别增益' \
  --overrides '{"llm":{"model":"<候选模型>"}}' \
  --development-pack '<pack-calibration-ID>' --validation-pack '<pack-validation-ID>'

# 只读查看；生产版本、分支轮次、调度任务都在输出中。
python scripts/iterate_branches.py --instance example-agent status

# 启用调度：已提交分支通过固定验收后自动推全。
python scripts/iterate_branches.py --instance example-agent run \
  --max-experiments 2 --workers 4 --max-attempts 3
```

`--candidate g-XXXX` 或 `--candidate j-XXXX` 可替代 `--overrides`，提交已有不可变版本，
用于提示词、场景或 Judge 模型权重资产变化。候选应基于当前全量准备，提交时冻结其相对全量的差异。
给已有分支名再次 `submit` 会创建新提案，旧实验保留；解决冲突也用这种方式。
分支名最长 80 字符，只能包含字母、数字、下划线和连字符，首字符为字母或数字。

`--max-experiments` 限制本实例同时执行的实验/生成包任务总数，也计入旧入口的在途任务。
`--workers` 是每个 Judge 评分或评估包生成任务内部的题目并发数；生成器 A/B 内部仍按题串行，
不同生成器分支可以同时运行。不要将这两个并发数混为一谈。
`run --once` 只推进/调度一次，已启动的独立子进程继续运行；持续自动推进需要保持调度器运行。

## 采用与重测

- 每轮先跑 2 题冒烟，无失败才进入开发全量；开发通过才允许固定验收。识别率方向、确认净胜、
  失败率与原来的门槛相同，协议在提交时冻结。开发通过不修改共享 iteration 指针。
- 固定通过后在锁内核对完整生产版本，随后原子推进 production 并写入采用凭证。
  若旧 iteration 仍等于旧 production，会一起推进；已有其他单线候选则保留。
- 多个实验同时达标只能先采用其中一个，其余分支必须基于新全量重测，不能把旧基线的提升直接相加。
  不相交的配置改动自动合并；同一字段或同一资产文件改成不同内容时显示 `conflict`。
- 生成器推全后，Judge 分支需要新的 AI 回复，会按原包相同题目重建；原包与原结果不改。
  明确的模型拒答作为失败题保留，其他异常可按断点重试。数据版本变化则显示 `blocked`。
- 固定集仍为 one-shot。同一候选只允许一个固定实验，已完成的候选不能重新验固定集；
  换分支名、来源信息或只换评判基线都不能绕过。这种情况显示阻塞原因，需要开发侧提出新候选。
  固定逐题答案及验证包生成断点不通过后台提供。

## 暂停与恢复

```bash
python scripts/iterate_branches.py --instance example-agent pause --name gen-model
python scripts/iterate_branches.py --instance example-agent resume --name gen-model

# 停止调度器后，修复任务错误，再重置该任务的调度尝试次数。
python scripts/iterate_branches.py --instance example-agent retry \
  --kind experiment --job '<status 中的任务 ID>'
```

暂停阻止后续调度和推全，不中断已运行的子进程。停止调度器同样不会停止在途实验；重新运行调度器
会识别已有进程和锁，避免重复执行。瞬时错误最多尝试 `--max-attempts` 次，成功题断点保留；
耗尽后显示 `retry_exhausted`，查看对应目录的 `scheduler.log`。`retry` 仅重置尝试次数，
不会解除 one-shot 或冲突门槛；评估包任务使用 `--kind pack`。

文件位于 `instances/<实例>/`：

| 路径 | 内容 |
| --- | --- |
| `branches/<name>/revisions/v-XXXX.json` | 冻结的提案、原基线、候选、评分协议和源包指纹 |
| `branches/<name>/rounds/r-XXXX.json` | 每一轮的生产基线、合并候选、阶段与阻塞原因 |
| `branches/<name>/state.json` | 当前提案、轮次列表与暂停状态 |
| `experiments/branch-<name>-r-XXXX-{smoke,dev,fixed}/` | 原有 spec/cases/state 和后台报告 |
| `judge_eval/pack-*-rebase-*/` | 按原题重建的回复包、断点与生成实录 |
| `scheduling/` | 调度尝试次数与子进程记录 |
| `pointers.json` | 当前生产/开发版本以及 `branch_promotions` 采用凭证 |

现有单线入口继续可用，不会被自动纳入分支推全。调度器不会自动训练新模型或提出下一项改动；
学习得到的候选先冻结为版本，再提交到分支。各进程仍共享本仓库的 Python 实现和全局资源，
本能力没有为不同运行时代码提供独立 checkout；运行期间不要用改共享代码的方式区分实验。
