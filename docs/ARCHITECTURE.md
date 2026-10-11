# 代码架构与目录说明

> 核查日期：2026-10-10。本文描述实际源码布局，不是待实施方案。
> 运行配置与操作见 [README](../README.md)、[OPERATIONS](OPERATIONS.md)；
> 可执行约束见 [rules.json](rules.json)。

## 总览

仓库只有一个 Python 包 `src`，import 前缀为 `src.`。一级目录按业务责任划分，
不是按某次实验、某代模型或某个 Dashboard 页面组件划分：

```text
src/
├── data/                      原始记录、会话、数据构建与来源核验
│   └── history/               HistorySources 来源信任边界
├── share/                     可版本化的共享知识材料
│   ├── versions.py            Share 快照、验证与版本绑定
│   ├── knowledge_view.py      共用知识卡片表示，不维护知识或执行查询
│   ├── wiki/                  Wiki 抽取、称呼、关系、修订与批次处理
│   └── memory/                只读知识查询、来源准入与历史记录检索
├── generator/                 生成回复
│   ├── generator.py           生成入口与提示组装
│   ├── context/               人格、查询规划与回复材料编排
│   ├── fewshot/               召回、排序、选取提示示例
│   │   ├── prepare.py         一次性准备示例池
│   │   ├── recall_engines.py  内存/SQLite 召回引擎
│   │   ├── rank.py            逐题规则评分与选择
│   │   ├── session.py         逐题召回 API
│   │   ├── recall_cache.py    受保护的精确召回结果缓存
│   │   ├── learned_selection.py  已有模型评分、排序、去重与预算选取
│   │   ├── provenance.py     排序模型及材料来源证明
│   │   ├── features.py / crosses.py / boosting.py  现用特征与模型评分
│   │   ├── runtime_layout.py 历史证据路径的精确映射
│   │   └── training/         样本、标签、拟合、对比、报告与断点复用
├── judge/                     盲测判别、特征抽取与评分
│   ├── features/             背景、对话归属与身份特征
│   ├── scoring/              在线评分执行器与通道分派
│   └── diagnostics/          特征检查与诊断报告，不拟合模型
├── iteration/                 跨业务域的实验、训练任务、验收、晋级与调度
│   └── materials/            材料密封与来源验证
├── dashboard/                 工作台 HTTP、页面、逐题证据与派生报告
├── infrastructure/            配置、缓存、追踪、存储、JSON schema 原语
└── 兼容入口与冻结模块          bootstrap/、history/、config/cache/tracing、llm.py

scripts/                       现行命令入口
scripts/legacy/                已退役实现和一次性脚本，仅本地保留
tests/                        按业务域分组的回归测试及公共支撑文件
```

`data/`、`share/` 对应工作台中的数据迭代与共享材料管理，但业务实现不依赖页面。
`iteration/` 管理 Data / Share / Gen / Judge 的版本、实验和晋级，不替它们实现业务算法。
`infrastructure/` 是通用机制，不是另一个业务产品。

## 各业务域的边界

### Data：原始记录到可用数据

`src/data/` 负责导入（ingest）、会话边界（conversations）、质量与覆盖
（data_quality、import_coverage）、切分（partition）、示例池与测试集
（build_fewshot_pool、build_testsets）、历史材料构建（history_build）和
验收刷新（acceptance_refresh）。`data/history/sources.py` 负责来源加载、
字段核验、时间与消息边界、进程内 sha256 stamp 记忆。

Data 不决定生成提示如何排序，也不决定模型是否晋级。冻结的
`iteration/datasets.py` 仍承接数据版本操作，不为目录整齐迁移历史绑定入口。

### Share：有来源、可冻结的共享知识

`versions.py` 负责共享快照的构建、验证和配置绑定。Wiki 与 Memory 消费
同一份共享知识，不是两套知识库：

- `wiki/` 负责抽取、整理、修订、维护知识，以及离线分包和审阅导出。
  `wiki_prompt.py` 只导出审阅卡片，不执行在线按题查询。
- `memory/` 负责只读查询已有知识和历史记录。`wiki_lookup.WikiLookup`
  按账号、来源、时间、答案与留出范围筛选可用知识，并检查查询前后的来源
  与 Share 内容是否变化；`search_memory` 负责按需检索原始历史。
- `knowledge_view.py` 提供共用数据表示和卡片序列化，不决定查询准入，
  不维护知识，也不组织生成回复的提示词。

Gen 的 `context/background.py` 只规划查询、调用 Memory、组织回复材料，
不再自行加载 Wiki 或重复实现知识准入。Memory 不反向依赖 Wiki 的维护管线。

Share 不等于 Few-shot：前者提供人物事实、称呼、关系和背景知识，后者提供
“面对类似上下文如何回复”的历史示例。Gen 与 Judge 各自通过冻结配置绑定
`share_ref` 和 `share_sha256`，可使用不同版本；升级 Share 不隐式改变旧模型。
共享记忆工具不依赖 Judge 或 Gen 的私有实现。Wiki 当前复用的模型传输以及
Share 对消费配置的显式核验仍保留，不通过复制代码强行消除必要依赖。

### Generator：上下文材料 + Few-shot + 回复生成

`generator.py` 负责调用与提示组装；`context/` 负责人格、查询规划和回复材料编排；
`fewshot/` 负责从有资格的历史示例中找候选、评分、选出进入提示词的少量示例。
这里的“选例”就是选历史回复示例，不是抽取评估题目。

训练编排通过 `GenerationConstraints` 向 `build_prompt` / `generate` 逐题传入
排除示例 ID 和来源核验函数，不修改召回器私有状态。约束覆盖普通召回、双路选取、
已有模型评分和重排，以及最终示例展示；任何召回或核验失败都禁止降级生成。
约束仅作用于当次调用，不污染共享索引或下一题；模型请求、重试和缓存结果复用
前后仍核验来源。普通非历史、无约束调用保留原有无示例降级行为。
已有模型评分的特征请求、纠错及逐项特征缓存复用也核验当次来源；
通过局部客户端适配，不修改冻结抽取器字节或共享客户端状态。

- `prepare.py` 是池加载、逐行核验、特征物化和深只读冻结的唯一准备入口。
- `recall_engines.py` 实现内存与 SQLite 两种召回；索引异常按既有规则处理。
- `rank.py` 是逐题纯评分；`session.py` 的 `open_session()` 先探测索引绑定，
  再水合，只有干净缺失才用内存装载。
- selection、reranker、shared_history 是选取策略及共享历史适配。
- `recall_cache.py` 缓存相同输入和受保护来源下的精确召回调用；来源、排除 ID
  等条件变化必须失效。它不是独立业务域，也不是第二套召回实现。

**在线选取与离线训练的边界**：`learned_selection.py` 在逐题运行时使用已有模型
评分、排序并选取，`provenance.py` 核验模型与训练材料来源；`training/` 才负责
造样本、打标签、提取训练特征、拟合权重、候选对比、训练报告和断点复用。
逐题服务不训练模型；没有独立用途的 `ranker/` 包装层已删除。
checkpoint_reuse 与 label_reuse 分别处理准备断点与标注
复用，不再为它们各建一个只有单件实现的目录。

生产 g-0040 启用 `pairwise_xgb_v1`，仍需为历史回复示例评分。现用特征与
评分实现已直接归入 `fewshot/`，由 `learned_selection.py` 调用；旧
`fewshot_ranker/` 源码目录及 logistic、training 兼容入口已删除。

十文件（__init__、features、schema、extraction、crosses、identity_crosses、
boosting、action_crosses、concern、supplemental）受已注册模型的来源证明
约束，迁移保留原字节；旧 initializer 以 `_frozen_init.py` 仅作历史证据，
真正的包入口仍为 Fewshot 的 `__init__.py`。`runtime_layout.py` 对这十件做
精确路径映射，历史证明中的逻辑路径与 SHA 不变，不能用归档字节掩盖
现用评分代码变化。冻结内核依赖的优化 API 由正常的 Fewshot 和 training
包入口惰性提供，不再保留旧包。实际执行的桥接文件和来源重构依赖
（样本、标签、数据集、报告及对比模块的依赖闭包）另纳入运行封印，
实现变化不得继续复用旧核验缓存，也不改写历史证明或要求重跑实验。
训练 runtime 清单统一由 `training/runtime_sources.py` 生成。

### Judge：特征、在线评分、诊断

`features/` 负责背景、对话归属、身份联合特征；`scoring/` 负责贡献、归属、
身份通道与在线执行器；`diagnostics/` 负责特征检查和诊断报告，不承担拟合。
因此不使用含混的 `models/` 或误称训练的 `training/`。实验与训练任务的
编排由 Iteration 承担，不在 Judge 内再次实现晋级流程。

现行方向是大模型抽取特征，小模型预测概率；身份混淆作为特征，不新增规则
改判层。身份联合通道按冻结配置显式分派，未知通道或显式空值不得静默回退。
历史归属覆盖模型仍有真实消费者，保留其执行语义，不将历史模型直接等同无用代码。

judge、corrected、corrected_v1、gbdt、lr_retrain、fusion、embedding、
embedding_runtime 留在原位置：这些入口或模型文件存在冻结字节/路径绑定。
`corrected_runtime_sources.json` 仍与 `corrected.py` 同目录；codex_transport
仍满足归档的动态加载路径。migration 仍由 import_wechat 使用，不因名字像
旧工具而删除。

### Iteration：跨业务域编排与结论

`src/iteration/` 保留一套编排入口，按用途理解这些模块：

| 用途 | 主要模块 |
|---|---|
| 版本与生命周期 | versions、baseline、experiment、runner、acceptance |
| 协议、结果与晋级 | protocol、record_contract、metrics、gates、promote |
| 分支、任务与调度 | branches、branch_packs、jobs、scheduler、control、task_state、parallel、progress、journal |
| 数据与材料保护 | datasets、data_guard、materials、material_compatibility |
| 归档与传输 | runtime、runtime_auth、pack_transport、generation_transport、retrieval_profile、retrieval_compare |
| 训练与来源证据 | training、training_operations、training_evidence、learned_gen、gbdt_evidence、fusion、embedding_adoption、embedding_tuning |
| 已有测量与复用 | measurements、draw_replay、generation_reuse、comparisons、reply_model_report |
| 只读在线评分排障 | inference_check（已有权重加载和合成预测，不训练、不产生实验结论） |

这里的 training 是跨模型训练任务管理，不是 Gen 的排序器拟合算法；
reply_model_report 是可供工作台消费的尝试与速度数据，不生成 HTML。
存储原语已归 `infrastructure/storage.py`，Share 业务已归 `share/versions.py`。
其余模块继续同包协作，避免为流程步骤再建立一组相互循环的子目录。

`materials/` 的 seals/validation 是材料验证正式实现；learning_guard、
material_reuse、transport_profile 保留归档或冻结代码所需的适配入口。
pack_transport 按已认证 manifest 分流旧/新检索代际，混用或缺失拒绝；
当前召回缓存动态加载指向真正的 `fewshot/recall_cache.py`。
完成的历史运行仍按其认证归档核验，源码迁移不意味着重跑模型或重签证据。

### Dashboard：展示，不产生实验真相

`server.py` 负责 HTTP 与路由，`report.py` 负责派生报告，changes、version_view、
trace_view 负责对应视图。执行器不 import Dashboard：runner 写实验记录，由
experiment.finish 保存指标与结论；命令入口在执行返回后调用 report.publish_safely。
报告失败只记录展示错误，不回写实验失败或覆盖完成进度。模型、材料、协议
和执行错误仍按原有机制处理。直接调用 runner 不会自动生成 HTML。

### Infrastructure：通用机制

config 解析配置和实例路径；cache 管理请求缓存与代码摘要；tracing 管理追踪；
storage 提供原子读写与哈希；json_schema 提供严格对象 schema。
这里不导入业务规则来判定结果或修改生产指针。
根 `src/llm.py` 是冻结传输入口，继续保留而不是为了视觉统一迁移。

## 已清除实现与兼容边界

- 无现行生产消费者的 bootstrap/analyze、judge/pairwise_models 已退出 `src/`，
  连同对应历史测试在本地 `scripts/legacy/` 保留，不入公开仓。
- 旧 bootstrap/wiki 实现、ranker_expansion、ranker_preparation 及散落的 ranker
  样本/标签/报告实现已迁至上述业务域，不留两份算法实现。
- 根 config/cache/tracing、history、bootstrap.acceptance_refresh、iteration
  storage/shares、Gen/Judge 的必要旧路径是薄兼容入口。模块别名指向同一对象，
  保持缓存状态、异常身份与 monkeypatch 一致；不是复制实现。
- 历史检索适配器 generator/{disk_history,indexed_retrieval,disk_source_reads,
  retrieval_cache,history_sources}.py 仅供冻结/归档调用，新消费者使用新业务路径。
- corrected 系、Gen 主入口、历史池及检索四件核心、部分实验/证据文件和
  ranker 十件内核保持本次迁移前字节；本次没有更改模型、协议、实例或生产指针，
  没有新增摘要别名、重签 provenance 或发起模型请求。

## 命令、测试与持续约束

`scripts/` 只提供现行命令入口；`scripts/legacy/` 保留历史一次性代码，不作为
新的业务依赖。代码校验统一经 `scripts/check.py`，不为迁移增加临时校验脚本。
`tests/` 按 data / share / generator / judge / iteration / dashboard 分组，
根保留通用机制测试、冻结边界与公共测试支撑。

`tests/test_architecture_layout.py` 持续核验：兼容入口共享同一模块对象、
训练 runtime 清单完整、冻结 ranker 字节不变、当前缓存加载真实实现、
业务执行不依赖 Dashboard、共享记忆不依赖 Gen/Judge、退役实现没有生产 import。
同时防止重新引入无独立用途的 fewshot/ranker 包装层或 Judge models/training
误名，并核验材料清单、身份通道指向真实源码。`check.py inference` 可独立核验
现有评分资产的加载与合成预测，不替代来源证明、真实模型请求或实验验收。
文档、规则符号与公开文件清单随目录同步；公开边界最终由 gitignore 与
`scripts/check_public.py` 强制，历史归档字节仍由各自认证流程核验。
