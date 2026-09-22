# 实现细节（IMPLEMENTATION）

本文档描述文件格式、状态机与校验细节，供维护与排障。**约束与合法步骤以 SOP.md 为准**；
本文与 SOP 冲突时以 SOP 为准，以代码为事实。

## 目录结构（全部 gitignore）

```
instances/<实例名>/          # 每个数字人一套（gitignore）
├── pointers.json      # {"data","production_gen","iteration_gen","production_judge","iteration_judge"}
├── data/d-XXXX/                     # 数据版本（创建后只读）
│   ├── manifest.json                #   id/seed/切分统计/refreeze_reason/testsets/池
│   ├── fixed_test.jsonl dev_pool.jsonl fewshot_pool.jsonl report.json
│   └── persona.md scenarios/        #   工作区（生成器版本创建时快照）
├── generators/g-XXXX/               # 生成器版本（创建后只读，候选同样在这里）
│   ├── config.json                  #   llm/retriever{enabled}/shots/data_version/persona_sha256
│   └── persona.md scenarios/        #   自包含快照
├── judges/j-XXXX/                   # Judge 版本（创建后只读，候选同样在这里）
│   ├── config.json                  #   mode=pairwise_llm 或 corrected_pairwise / llm / 资产指纹
│   ├── prompt.md                    #   提示词快照（随版本冻结）
│   └── meta.json                    #   来源（bootstrap / calibration + changed diff）
├── judge_eval/pack-XXXX/pack.json   # 冻结校准包（当前生产基线生成的 AI 回复）
└── experiments/<e-id>/              # 实验（唯一执行状态源）
    ├── spec.json                    #   输入快照：kind/dataset/change/baseline_ref
    │                                #   candidate_ref/judge_ref/data_ref/config_diff/fingerprint
    ├── state.json                   #   进度与结论：status/verdict/metrics
    └── cases.jsonl bill.md index.html
```

## 不变量（代码强制，违反即 ConfigError）

1. 版本目录 write-once；`create_*` 遇已存在即拒绝。
2. 生成器版本 config 不得含 `baseline_id/adopted_from_run/adopted_via`。
3. 指针指向不存在的版本 = 唯一指针错误。
4. 池审批：`report.json.review_status == approved` 且 `examples_sha256 == sha256(池文件)`。
5. 检索策略冻结：`PersonaFewShotRetriever(path)` 无策略参数（SOP §1.5）。

## LLM 连接与错误

`load_settings()` 自动读取项目根目录 `.env`，不覆盖已设置的环境变量。
ChatClient 支持 OpenAI 与 Anthropic；Coding Plan 端点走 Anthropic。
timeout_seconds、temperature、max_tokens 由冻结模型配置控制；重试只有一层。
网络超时、429 和服务端临时错误最多重试两次；401/402/403/404 及输出预算耗尽直接报出原因。
空输出、截断 JSON 和纯思考输出不能当成功结果进入评测。
模型或密钥必须使用其所属接口，不能把 Coding Plan 模型名套用普通 Ark 或 DeepSeek 接口。

## 实验状态机

```
run.py 新建（预检/准入/diff/one-shot 全过才建目录与候选版本）
  → running（cases.jsonl 追加写；中断后 run.py --exp <id> 续，成功题跳过）
  → finish（唯一入口；experiment_incomplete 保持 running，恢复续跑；
     恢复只跳过 status=ok 的题，failed 题重试）
```
- 准入：非冒烟时构成 == settings 声明；池 ≥ 5000 条。
- one-shot：fixed 实验按 candidate_fingerprint 查历史实验。
- 失败：单题异常 → failed 行；失败率 = 最终状态 failed / 全量题；每题取最后一行，
  retries = 行数 − 题数。
- 产物唯一事实：spec（含 protocol 快照：flip 轮数/门槛/失败率上限/force_reply，
  运行与决策不再读当前 settings）/ cases.jsonl / state.json；bill 与页面由它们生成。

## 指标口径

- 净胜只计翻转补验确认且非 contested 的题；初测识别数仅描述。
- 正式采用双条件：identified_candidate < identified_baseline 且
  net_win_confirmed ≥ formal_min_net_win。
- Judge 校准：识别率 = 正确指出 AI 回复的比例；候选 ≥ 现用即 adopt。

## 命令速查

外部部署导入使用 `scripts/import_wechat.py --source-root <来源部署根目录>`，读取
`config/current_judge.json`，拒绝静默套用默认裁判。已有实例先核对：
`python scripts/import_wechat.py --instance example-agent --judge-only`；加 `--adopt` 后
补齐数据元信息、创建 Judge 快照并切双指针，生成器和历史实验保持原版本。
后台“展开裁判配置与来源”可直接查看源配置、规则、参考资料和校正模型。
`judge/corrected_v1.py` 只保留来源的纯推理定义；迁移时逐定义核对来源指纹，
来源算法变化时必须更新适配，不能用旧特征定义装载新权重。

```bash
python scripts/bootstrap.py --data <导出.jsonl> --model <模型>   # 首次：初始化；之后：只建数据版本
python scripts/bootstrap.py ... --adopt                          # 采用新数据（重建快照版本+切指针）
python scripts/run.py --dataset development --change "…" --override k=v [--limit 5]
python scripts/run.py --exp <实验id>                             # 恢复
python scripts/promote.py gen --exp <实验id>
python scripts/evaluate_judge.py build --sample 500              # 冻结校准包（需 LLM）
python scripts/evaluate_judge.py compare --pack <id> --change "换模型" --override llm.model=<新>
python scripts/promote.py judge --exp <实验id>
python scripts/serve_dashboard.py
```
