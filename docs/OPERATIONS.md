# 运行与维护

已完成的 embedding 搜索用 `evaluate_judge.py embedding --study <study> --change <说明>`
冻结训练内部选定的主候选。入口只重建选定模型以确认训练来源，不重新搜索参数或调用大模型。
普通 `compare --candidate <judge> --saved-draw-manifest <path>` 默认只允许已存观察；
显式加 `--supplement-missing-rounds --workers 4` 才补齐缺失的 r1/r2，两个纯特征判别器
共享同轮观察。初测缺失、已有证据损坏或配置变化均拒绝，不能通过请求回退掩盖。
后续开发晋级、固定准入和正式采用继续使用同一门槛入口和独立补验规则。
Embedding 推理中，两侧离散编码完全相同则概率精确为 0.5，沿用既有平局选 A 的规则；
不同编码仍使用网络原始概率，不做阈值容差。修复不改变冻结训练代码及模型权重。

规范分工及人工判断步骤见 [SOP](SOP.md)，可执行检查和测试映射见 [规则清单](rules.json)。

```bash
python scripts/check.py code
python scripts/check.py experiment --instance demo
python scripts/import_comparisons.py --instance demo --study historical-study
# 上一步核验通过后，才写入 experiments；原始文件保持原样。
python scripts/import_comparisons.py --instance demo --study historical-study --apply
```

新专项比较生产者先调用 `comparisons.register`，用返回的实验目录进入 `record_contract.writer` 写逐题记录，最后调用 `comparisons.complete` 从记录重算指标。禁止自行把比较结果写入训练目录并标记完成。

需要 Python 3.10 或更高版本，以及支持 `fcntl` 文件锁的 macOS/Linux。所有命令从框架根目录执行。实例名和各类任务名只使用字母、数字、下划线与连字符。

## 先验证框架

```bash
python -m pip install -r requirements.txt
python scripts/check.py code --full --history --demo-output /tmp/digital-human-demo
```

演示输出目录必须为空。演示使用合成消息和模拟模型传输，实际构建材料、抽特征、拟合 LR、独立审计和评估，不消耗外部模型调用；不代表模型效果验收。

## 统一校验入口

所有主动校验使用 `scripts/check.py`，复用运行时的规则实现。普通实验创建、续跑、完成、晋级已自动检查数据内容、来源、模型资产和证据；日常推进无需先手动再跑整套检查。新增规则补进现有模块和回归测试，不为具体实验另写校验脚本。原检查模块保留兼容入口。

```bash
# 只在修改框架后，执行静态检查和相关回归；CI 使用 --full。
python scripts/check.py code --tests tests/test_draw_replay.py tests/test_check.py
# 可选预检/排障：冻结开发包、双方学习来源、当前基线上下文及已存观察。
python scripts/check.py --output /path/to/private/check-report.json judge \
  --instance demo --pack PACK_ID --candidate JUDGE_ID \
  --saved-draw-manifest /path/to/private/saved-draws.json
# 正式开发证据和分层门槛；固定准入要求已有生产版的直接对比。
python scripts/check.py experiment --instance demo --exp EXP_ID --gate development
python scripts/check.py experiment --instance demo --exp EXP_ID --gate fixed-entry
python scripts/check.py materials --instance demo --data DATA_REF --judge JUDGE_REF --generator GEN_REF
```

`judge --baseline JUDGE_ID` 可检查待衔接的对照；材料检查和当前指针是否匹配分别报告，指针不匹配时总状态仍失败。此命令仅适用于开发包，既有观察核验不代表所有必需独立轮次齐全；正式运行器继续按分歧要求补验，复用模式缺失轮次会停止，不自动发出模型请求。`experiment` 省略 `--exp` 时只检查实例历史结果是否已登记。

`materials` 将聊天对象重合与答案、未来信息泄漏分别报告。熟悉对象的早期聊天允许用于学习；对象重合不阻塞评测，但相关样本不能解释为模型从未见过该对象。数据版本变化时，模型仍按原训练版本核验来源，再检查新评测答案与时间边界；不重写训练证明，也不把旧数据上的成绩当作新数据成绩。

运行时检查不请求模型、不生成实验结果、不写基线指针，也不读取固定验收答案。JSON 报告给出每项通过或失败以及原因；任一失败均返回非零，互不依赖的检查继续汇总。`--output` 只能写入新报告路径。检查通过是当前材料/证据通过相应检查，实际晋级仍从晋级入口重验当前条件。

已冻结候选和观察可以交给普通实验命令，沿用创建和执行时的自动保护：

```bash
python scripts/evaluate_judge.py --instance demo compare --pack PACK_ID \
  --candidate JUDGE_ID --saved-draw-manifest /path/to/private/saved-draws.json \
  --change "复用已存特征比较冻结候选"
```

并行 Judge 分支的 `iterate_branches.py submit` 同样支持 `--saved-draw-manifest`，
将观察绑定到该提案的开发包和双方版本。冒烟与完整开发轮都复用已有请求，开发通过后
保留分支自己的开发版；相对生产版达到固定准入门槛才进入固定验收。开发观察不会带入
固定轮；生产条件变化时须提交匹配新条件的提案，旧观察不会自动回退为实时请求。

### 固定轮的统一执行流程

候选提交后，由 `scripts/iterate_branches.py run` 连续推进；`status` 和后台历史表读取同一份状态。正常推进不需要负责人依次手动执行预检、登记、算分或晋级命令。

| 阶段 | 代码负责的动作 | 重复执行与复用边界 |
| --- | --- | --- |
| 开发晋级、固定准入 | `branches.advance`、`gates` 检查完整开发记录和独立补验；达到冻结开发门槛后保留分支开发版；固定准入使用候选对生产版的直接比较 | 已有同条件直接比较复用；不能累加不同对照的净胜。准入不足就停在开发版 |
| 分配固定批次、准备回复包 | `acceptance`、`branch_packs` 分配封存批次，冻结数据、Gen、Judge 和协议 | 同一任务恢复原批次、原成功回复；开发观察不得充当固定观察 |
| 创建、启动或续跑 | `branches._ensure_trial`、`learning_guard.execution_seal` 自动检查学习来源、时间边界及内容指纹 | 创建时冻结；实际启动或续跑重验当前输入。调度轮询已有任务不重建来源、不重验开发准入 |
| 初测、分歧补验、失败重试 | `runner`、`scheduler` 完成两侧初测，分歧题按冻结规则独立补验，失败按重试上限续跑 | 同条件成功步骤与相应独立轮次复用；运行中用轻量封印检测输入变化 |
| 完成、正式采用 | `experiment.finish`、`branches.promote_experiment` 汇总完整记录，按冻结门槛决定；通过后在事务内切生产指针 | 完成核对记录；采用时复核当前生产基线、来源和准入凭证，防止过期结果覆盖新版本。未达标保留现用版 |
| 历史与后续分支 | 后台从实验记录、共享指针及分支开发记录生成；其他分支在新生产基线上继续 | 不手填表，不把已结束固定轮重跑到达标 |

曾经在这些自动入口之外重复运行 `check.py judge/experiment`、手工读取逐题数据重算、逐个查版本整理表，均不是正常流程的必需步骤。`check.py` 用于明确故障的排查和代码改动后的相关回归，不作为每轮前置清单。创建、续跑、完成、采用跨越了输入或状态可能变化的边界，其自动保护保留；轮询等待本身不触发完整审计。

需要负责人判断的是下一项假设和候选方向、预算、异常的共同原因与修复方案、结果适用范围，以及是否变更样本/评分规则/门槛。既定条件下的计数、缓存命中、重试、门槛判定、登记、晋级和历史更新全部由代码完成，不增加逐步人工批准。具体分工见 [SOP §10](SOP.md)。

当前配置三个门槛均为 `0.005`，即每 1000 有效题确认净胜至少 5 题。负责人明确修订已结束固定轮的门槛后，通过同一晋级入口复用成绩：

```bash
python scripts/promote.py --instance demo judge --exp FIXED_EXPERIMENT_ID \
  --threshold-reason "用户决定：开发及固定门槛统一为 +5/1000"
```

此命令不发起模型请求，不改写原结论；新规则仍要求固定初测严格改善、原有独立补验和失败率达标。没有明确门槛修订时，普通晋级继续使用冻结协议。

## 连接一个实例

### 全离散 embedding DNN 批量调参

安装可选 PyTorch 依赖后，运行同一个入口；相同命令可续跑成功断点：

入口默认设置 `OMP_NUM_THREADS=1`，避免 PyTorch 与 XGBoost 同进程时竞争 OpenMP 线程池；实验并行数由 `--workers` 控制。

```bash
python scripts/tune_embedding_judge.py --instance demo --study EMBEDDING_STUDY \
  --source CLEAN_TRAINING_STUDY --arm luna --pack DEVELOPMENT_PACK \
  --saved-draws /path/to/private/saved-draws.json --workers 3
```

复用原训练对和已有 LLM 特征，所有字段（包括布尔、分桶长度、类别和上下文）进入独立的 8 维 embedding，没有数值旁路。A/B 共用网络，损失使用两侧 logit 之差。BN 的同一次前向包含两侧回复，推理冻结统计量。未知类别使用零向量，词表仅从拟合数据构建。

一次调用共用训练来源检查、数组准备和开发观察读取：36 组 MLP/交叉层、none/BN/LN 及优化参数先按训练内部后 20% 的 logloss 筛选；前 6 组补 2 个种子；按 3 种子平均选前 4 组，在完整训练集各重训 3 种子。完整训练轮数取内部最优轮数中位数。共 60 次拟合、12 项原正式开发包比较；选择在观察开发结果前冻结。内部留出只是缓存特征上的分类头筛选，抽特征参考材料曾使用整个训练区，因此不宣称它是端到端时间留出。最终结论来自原正式开发集。

批次绑定输入、特征、代码、配方和运行版本；续跑复用成功模型文件，变化时要求新任务 ID。来源检查只在批次启动/恢复时运行，单配置使用轻量变更检测。正式比较在 `experiments` 登记后写入逐题结果，共用同一批独立观察，不请求大模型。缺失分歧补验标为 `awaiting_independent_evidence`；`judge_training/STUDY/report.md`、`results.json` 自动列出完整初测与缺口，不能把初测收益直接当作确认净胜或晋级。模型选择与固定轮门槛沿用原流程。

```bash
export DH_INSTANCES_ROOT=/path/to/private/instances
export DH_ENV_FILE=/path/to/private/connection.env
export DH_SETTINGS_FILE=/path/to/private/settings.yaml
python scripts/iterate_branches.py --instance demo status
python scripts/serve_dashboard.py --daemon --port 8080
```

`DH_INSTANCES_ROOT` 指向实例的父目录。`DH_ENV_FILE` 提供连接配置，已有环境变量优先；`DH_SETTINGS_FILE` 提供评估规模、比例和门槛。没有显式设置时使用框架默认路径。模型选择、数据窗口和额度均由实例明确配置。

后台为 `/dashboard/demo/index.html#workflows`；`scripts/serve_dashboard.py --status` 查看服务，`--stop` 停止服务。后台服务与实验进程分开，停止后台不等于停止实验。

## 首次建数据与基线

参考 [数据策略模板](../examples/data-policy.template.json)，按真实历史填写窗口和规模。

```bash
python scripts/bootstrap.py --instance demo \
  --data /path/to/private/messages.jsonl \
  --policy /path/to/private/data_policy.json \
  --model GENERATOR_MODEL --judge-model JUDGE_MODEL --initialize
```

这个入口使用统一 JSONL 消息，生成数据版本和通用机械指令基线；不调用模型，已有基线不会被覆盖。对于 WeFlow 目录导出，使用下面的容量检查，确认参数后增加 `--build`：

```bash
python scripts/prepare_history.py --instance demo \
  --exports /path/to/private/exports --policy /path/to/private/data_policy.json
```

已有实例更换数据或建立新的兼容起点时，先准备迁移凭证，再用返回的凭证 ID 应用：

```bash
python scripts/migrate_baseline.py --instance demo prepare \
  --data d-0002 --generator g-0002 --judge j-0002 --reason "建立已核验的新数据起点"
python scripts/migrate_baseline.py --instance demo apply --receipt RECEIPT_ID
```

迁移会重验材料和原指针，不是效果晋升。不能为了让命令通过而给未知来源的旧资产补写 `verified: true`。

## 提交候选并自动执行

先按 [训练方案模板](../examples/training-plan.template.json) 明确数据、生成器、特征模型、初判模型及配方。训练题数来自用途清单。

```bash
python scripts/iterate_branches.py --instance demo train \
  --name study-a --plan /path/to/private/training-plan.json
python scripts/iterate_branches.py --instance demo run --max-experiments 2 --workers 4
```

训练依次执行两题预检、训练回复、各臂特征和 LR、开发回复及独立审计。多个研究任务可并行；一个研究内部的训练臂顺序执行，各臂的题目可并行。

方案设置 `branch_name` 和 `candidate_arm` 后，会将人指定的那一臂交给候选分支；未设置时完成训练与审计后停止。数据与当前基线不兼容时显示 `needs_attention`，不会自动更换真实基线。

生成器的人工配置改动直接提交：

```bash
python scripts/iterate_branches.py --instance demo submit --name generator-a \
  --kind gen --change "调整示例条数" --overrides '{"max_shots_per_case":4}'
```

Judge 候选需要开发回复包。训练结果会给出开发包引用；独立构建时也使用同一个可恢复构建器：

```bash
python scripts/evaluate_judge.py --instance demo build --workers 2
python scripts/iterate_branches.py --instance demo seal --data d-0001 --batch-size 1000
python scripts/iterate_branches.py --instance demo submit --name judge-a \
  --kind judge --change "人工选择的判别改动" --candidate j-0002 \
  --development-pack PACK_ID
```

上面的 `--sample` 已不是抽样开关；可省略，提供时只校验是否等于完整冻结清单。固定包由分支在准入通过、分配批次之后构建，独立 `build --kind validation` 会拒绝提前生成固定答案。旧任务已保存的固定包仍按兼容路径核验。

`run --once` 只推进一次，已经启动的子进程继续执行；默认 `run` 持续调度。`max-experiments` 包括训练、回复包和评测三类任务，`workers` 是每个任务内的题目并发，不是实例总请求并发。

## 生成器提示词分支

```bash
python scripts/iterate_branches.py --instance demo submit --name conversation --kind gen \
  --prompt-recipe conversation --change "保持原立场，改善上下文承接"
```

`--prompt-recipe` 与 `--candidate`、`--overrides` 三选一，仅用于生成器。
当前提供 `mechanical`、`conversation`、`conversation_partner` 和 `conversation_rhythm` 四个已审阅的通用指令配方。后两者分别在接话指令上增加按对象互动、消息节奏说明，不加入具体答案、事实或学习样本。对象只能根据当前输入和已提供历史判断，材料不足时不猜关系。指令精确内容仍由材料保护绑定，不能靠重签来源文件放行任意内容。
新候选继承该分支当前有效开发版的模型、few-shot 数量及预算；没有开发版时继承生产版。创建不更改指针，后续沿用统一分支调度。
并行试验可设置 `--from-branch conversation`，以该来源分支已采用的开发版作为共同对照及改动起点。提交自动核对同一生产条件、正式采用记录、未变的材料和数据、完整补验净胜；不重新请求历史结果，不重签原实验。来源记录冻结在提案中，新实验仍执行当前材料保护。无效或过期来源会拒绝，不能用任意候选充当开发基线。
实例可在新配置版本设置 `dev_min_net_win_rate: 0.0`，表示确认净胜严格大于零即可开发晋级。`fixed_entry_min_net_win_rate` 与 `formal_min_net_win_rate` 仍可保持 `0.005`：先直接对生产版确认累计收益，再进入固定轮；不能相加不同对照下的净胜。配置通过 `DH_SETTINGS_FILE` 加载，旧实验使用冻结门槛。

已冻结的示例排序生成器可通过 `scripts/evaluate_learned_fewshot.py start --candidate <版本> --base <现用生产版> --data <当前数据版> --name <分支> --instance <实例> --output <续跑目录> --stage fixed_test` 验收。该模式复用原候选，开发、固定准入、固定验收及晋级沿用分支状态机；重复启动使用相同分支和目录，只补缺项。默认 `--stage development` 仍在开发结论后停止。新数据使用新分支，历史结果继续绑定原数据。

## 生成器并行开发比较

few-shot 默认仍使用规则召回。版本配置可显式设置 `retriever.reranker` 为标准 ChatClient 模型配置；先按原规则与历史时间过滤召回，再由模型选择 `max_shots_per_case` 条。当前输入只包含聊天类型、对象和消息正文，不包含评测答案；例子必须来自可验证的历史。例子数量与展示预算不变，重复请求使用现有缓存，无效选择重试一次后记失败。

版本也可设置 `retriever.selection: context_mmr_v1`，使用本地选择策略，不能与 `retriever.reranker` 同时启用。分别用最近三句、最后一句在同一历史检索器中召回最多 12 条，沿用相同的时间、对象及样本排除保护。按例子 ID 合并后，用 0.35/0.65 权重融合两路排名，再逐条选择相关性高且与已选内容重复较少的示例。该策略不调用重排模型；数量与字符预算仍由原配置控制，超长例子跳过后继续寻找能完整放入的例子，选择依据写入逐题 trace。未设置此策略的版本保持原选择行为。

`python scripts/iterate_branches.py --instance demo limit --name BRANCH --stage development` 持久限制该分支只做开发比较。有收益的开发版仍自动保留，可作为同分支下一提案的起点；不会自动创建固定轮或改生产指针。限制不会中断已在运行的任务。

`run --timeout-seconds 180` 或 `worker --kind experiment --timeout-seconds 180` 可提高传输等待下限。每次启动记录操作凭证，仍运行原代码快照，保留请求身份、模型配置和成功断点；等待上限更高的客户端保持原值。

## 已保存的 GBDT 判别器

运行器支持 `decision_policy: gbdt_only`，通过 `scorer_file` 指定版本内的评分资产，并在 `assets` 绑定其 SHA-256。评分资产使用 schema 1、`rank:pairwise` 的 XGBoost JSON 权重，依赖可选的 `xgboost>=2.0`，运行版本必须与权重内记录的版本一致。特征沿用冻结的结构化定义及顺序；分别计算两个选项的分数，再计算 `sigmoid(score(A)-score(B))`。不对特征差直接运行树模型。

该入口只加载权重，不进行训练；已有特征可按原样本、回复和独立轮次复用。运行支持本身不授予来源合格或晋级资格：历史专项比较仍为诊断记录，候选进入正式实验前仍须通过学习来源核验，不能通过新增模型文件或重登记绕过验收。

来源核验先重建原始训练、参考例子及特征请求，再按冻结的 GBDT 配方在内存中复现全部树权重。此核验不请求大模型、不改变已保存模型、不读取评测答案。候选须绑定 `scorer_source.json`、训练矩阵清单和权重凭证，并保持原有特征配置、材料及学习时间边界；旧比较记录不会因此自动获得采用资格。

混合对照或纯 LR/GBDT 对照与纯 LR/GBDT 候选可通过 `create_judge_eval_experiment(..., saved_draw_manifest=...)` 复用已保存的开发观察。观察来源不必是当前对照版本；双方使用各自冻结判别器对相同特征重评分。清单须绑定完整开发包、原观察来源配置与资产、双方一致的特征请求配置、每题每轮的原始请求证据，以及所有证据文件和缓存条目的内容哈希。运行器在已登记实验内重新评分、写入逐题结果并核验完成；不导入历史汇总或改写其采用标记。原记录后续步骤失败，不影响其中已成功且来源核验通过的步骤。

该入口只用于开发轮。初测产生分歧时，仍读取两轮独立补验；缺失轮次直接拒绝，不能用初测代替，也不会自动请求大模型。样本、模型配置、生成回复、证据或轮次发生变化均拒绝复用。固定轮仍通过既定封存、分配和独立验收流程执行。

### GBDT 与 LR 分数融合

```bash
python scripts/evaluate_judge.py --instance demo fuse --gbdt GBDT_ID --lr LR_ID \
  --data DATA_ID --change "固定 80% GBDT 与 20% LR 分数融合"
```

此命令只创建冻结候选，不请求大模型、不改生产指针。两个父版本须来自同一份可重建的训练、特征及参考材料；各自原始分数以完整训练集的配对分差 RMS 归一化，再按 0.8/0.2 融合。两个回复分别过模型后才求分差，评分为 `sigmoid(score(A)-score(B))`。权重固定，开发和固定评测不参与尺度估计。

新候选仍通过普通分支提交、开发补验、固定准入和正式采用流程；开发可使用同一份 `--saved-draw-manifest`，不会重新抽特征或用初测替代独立补验。来源保护自动重建父模型及训练尺度，绑定父版本全部学习证据并检测运行中变更。

## 查看、取消与恢复

```bash
python scripts/iterate_branches.py --instance demo status
python scripts/iterate_branches.py --instance demo cancel --kind training --job study-a
python scripts/iterate_branches.py --instance demo retry --kind training --job study-a
```

`kind` 可以是 `training`、`pack` 或 `experiment`。重试只重置调度重试额度和取消标记，不改样本、规格、已成功断点或已消耗验收批次；之后运行调度器，由它选择原代码快照。直接创建的回复包和实验也可以通过这个重试入口交给调度器。

| 状态或原因 | 处理 |
| --- | --- |
| `waiting_retry` | 调度器会按重试间隔重新启动 |
| `retry_exhausted` | 查 scheduler.log/任务错误，修复后显式 retry |
| `runtime_unavailable` | 恢复原解释器、依赖及完整快照；不重新签名旧结果 |
| `legacy_runtime_not_frozen` | 使用旧执行器处理旧任务，或创建新任务 |
| `request_budget_exhausted`、`cost_budget_exhausted` | 检查实例额度和已消耗记录，调整后显式 retry |
| `deadline_reached`、`cancelled` | 按需要调整截止时间或解除取消，再 retry |
| `acceptance_exhausted` | 补充新的封存数据，已使用批次不能回收 |
| `conflict`、`blocked`、`baseline_review` | 检查候选、当前基线及来源条件，由人决定下一方案 |

资源策略参考 [额度模板](../examples/resource-policy.template.json)。失败请求仍消耗额度；缓存命中不新增传输调用，但仍检查取消和截止时间。成本单位是保守预留，不是服务商实付账单。取消的实际生效时间受在途外部调用影响；运行中的调度器可以停止自己持有且身份匹配的任务进程组。

## 维护与发布

修改框架前保留可恢复版本，避免改动旧任务绑定的文件。运行快照能保存执行代码，环境仍需要维护；不要把安装最新版依赖当作旧任务恢复方案。

本地启用 `git config core.hooksPath .githooks`，CI 再执行公开边界、文档和回归检查。新增公开文件须逐项进入 `public-files.json`。检查器覆盖常见秘密、个人路径、私有依赖和 Git 历史，不能认证任意自然语言内容适合公开。含私有历史的原仓库不能通过删除当前文件就直接公开。

本框架没有规定开源许可证；真正发布前由所有者决定许可证和目标仓库。
