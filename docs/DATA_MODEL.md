# 数据模型与事实来源

一个实例对应 `DH_INSTANCES_ROOT/<instance>`。原始消息、人格、模型权重、连接配置与运行证据都属于私有实例；框架只包含机制、通用模板和合成示例。

## 目录与写入责任

| 路径 | 写入者 | 内容与生命周期 |
| --- | --- | --- |
| `data_policy.json` | 实例维护者 | 新数据版本的时间窗口、规模和抽样比例；不改变已有版本 |
| `data/d-*/` | history 构建器 | 原始消息快照、历史池、用途清单和指纹；构建结束后冻结 |
| `generators/g-*/` | 版本创建器 | 生成配置、人格、场景、学习来源；不可变 |
| `judges/j-*/` | 版本创建器 | 判别配置、提示词、特征权重、学习来源；不可变 |
| `pointers.json` | 晋升或迁移入口 | 当前数据，以及 Gen/Judge 的生产、开发指针 |
| `judge_training/<study>/` | 训练工作进程 | 冻结方案和规格、生成及特征断点、候选、审计与进度 |
| `judge_eval/<pack>/` | 回复包构建器 | 冻结 recipe、可恢复 building、完成后的 pack |
| `experiments/<id>/` | 评估工作进程 | 冻结 spec、逐题 cases、可变 state、派生 bill/index |
| `branches/<name>/revisions/` | 人工提案入口 | 每次提案的冻结版本、起点和候选 |
| `branches/<name>/rounds/` | 分支推进器 | 各轮阶段、实验引用、准入结果；过程记录可更新 |
| `scheduling/` | 调度器 | 尝试次数、启动和退出记录；不是实验结论 |
| `acceptance/d-*/` | 封存/分配入口 | 冻结 manifest；可变 ledger 记录批次所有者及使用状态 |
| `runtimes/<hash>/` | 运行冻结器 | 内容寻址的源代码、设置副本和环境 manifest |
| `evidence_blobs/` | 证据归档入口 | 按原哈希归档的历史代码；数据仍在原绑定路径核验 |
| `baseline_migrations/` | 迁移入口 | 迁移前后指针、来源证明和应用记录 |
| `resource_policy.json`、`resource_usage.json` | 实例维护者 / 请求控制器 | 额度策略 / 累计预留、实际调用尝试和活跃请求 |
| `.cache/`、各任务的 `traces/` | 传输及执行层 | 请求复用和实际调用证据；缓存不能替代来源核验 |

`generator_builds` 记录同一行为生成器曾被哪些数据构建使用；数据变化不会必然新建行为版本。编号不是采用记录，当前使用哪个版本以 `pointers.json` 为准。

实验里的 `baseline_ref` 记录冻结的**实验对照**；控制实验的对照也可以是未晋级的重训版本。它不等于 `production_judge` 或 `iteration_judge`。后台分别展示实验对照和当前基线，只有晋级入口能够把正式实验的候选采用到对应基线；显式初始化/迁移另有凭证。普通数据构建与训练不会更换指针。

## 六种数据用途

| 用途键 | 文件 | 作用 |
| --- | --- | --- |
| `gen_learning` | `gen_learning.jsonl` | 构造人格、提示词等静态学习材料 |
| `gen_optimization` | `gen_optimization.jsonl` | 调整生成策略；其评估结果不能直接晋升 |
| `judge_training` | `judge_training.jsonl` | 构建词表、抽特征、拟合 LR |
| `development` | `dev_pool.jsonl` | Gen 开发比较 |
| `judge_development` | `judge_dev_pool.jsonl` | Judge 开发比较 |
| `fixed_test` | `fixed_test.jsonl` | 固定验收池，按封存批次使用 |

`purposes.json.roles` 保存文件名、样本量和 SHA-256；`protocol` 保存时间边界、共享说明和拆分参数。Gen/Judge 的学习用途允许声明重叠；开发用途共享题目时不构成两次独立验证。时间与同秒消息的处理见 [数据边界](DATA_BOUNDARIES.md)。

完整历史池可以包含所有导入历史，**物理收录不等于每道题可见**：检索必须按题目输入时间、来源和排除条件过滤。

## 消息、题目与回复包

统一导入消息包含 `chat_id`、`sender`、`timestamp`、`text`、`is_self`；时间使用 Unix 秒。详见 [合成导入示例](../examples/chat_export.sample.jsonl)。

历史题目保留 `case_id`、上下文、真人连续回复、`context_message_ids`、`reply_message_ids`、`source_span` 和 `input_cutoff`。`source_span` 表达完整片段在原始聊天中的位置；`input_cutoff` 表达回答之前模型能看到的输入时刻。内容 ID/哈希只能证明一致性，原始记录重建负责证明内容来源。

回复包的 `rows` 在原题基础上添加 `ai_replies` 及生成 trace/失败字段；`data_ref` 和 `c0_gen_version` 锁定数据与生成器，`generation_proof` 绑定生成 recipe 和完整行内容。固定包还必须绑定验收 receipt。比较 Judge 时两臂使用同一个包。

## 规格、状态、结果

- `proposal.json`：人的意图。训练臂、模型、评分策略和拟合配方在此指定。
- `spec.json`：执行契约。锁定数据/模型引用、协议、输入指纹和来源证明；新实验还对应 `runtime.json`。
- `state.json`：工作进度、错误和最终结论。评估任务的逐题进度位于 `progress`；训练直接记录 `phase`；回复包进度位于 `progress.json`。`jobs.snapshot` 统一读取这三种已有格式。
- `cases.jsonl`：追加式逐题记录；重试保留旧行，按 case ID 取最后一条。损坏的尾行先备份再恢复，中间损坏拒绝继续。
- `spec.record_contract`：冻结 case ID 清单。统一写入器检查成员、判定字段和成功断点；完成入口对照题单重新核对结果。
- `spec.comparison`：专项比较两侧的特征方案、分类器配方和策略，不伪装成已创建或已晋级的 Judge 版本。专项比较同样位于 `experiments`，逐题使用相同字段及完成检查。
- `spec.provenance`：历史补录记录原任务、原比较 ID 和内容哈希。补录时的运行快照只证明导入代码身份，不证明原训练过程；这类结果禁止进入晋级和固定准入。
- `generations.json`、`features.json`：训练断点；每条记录绑定输入、输出及实际 trace，不能只靠 ID 跳过。
- `bill.md`、`index.html`、后台汇总：派生展示，不能作为晋升的唯一依据。

新状态时间为 Unix 秒，记录 PID 和进程启动身份。旧字符串时间仅做读取兼容；没有足够证据的旧状态不补写成新证据。`running` 是生产者记录，实际是否存活由统一任务视图另行判断。

## 发生变化时创建什么

| 变化 | 正确动作 |
| --- | --- |
| 数据内容、用途或时间窗口变化 | 新建数据版本，重新核验兼容性 |
| 模型、人格、场景或行为配置变化 | 新建 Gen/Judge 版本并提交候选 |
| 代码或依赖变化 | 新任务使用新快照；旧任务使用旧执行器 |
| 并行候选改变了生产基线 | 新一轮直接比较，不改旧实验对照 |
| 请求暂时失败 | 修复传输问题后恢复同一任务和成功断点 |
| 固定批次耗尽 | 补充并封存新数据；不能退回已暴露批次 |

`schema`、来源种类和缺失字段共同决定历史格式的兼容路径。文件只读权限是防误操作措施，真正的运行和采用条件仍由哈希、重建和准入检查执行。
