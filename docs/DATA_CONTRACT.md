# 数据契约（DATA CONTRACT）

各模块对接的唯一语义来源。格式变更必须先改本文档。

## 统一消息（ingest 出口，已断言）
{chat_id, chat_type: group|private, chat_name, sender, is_self, timestamp(int,秒), text, source_chat_id?}
- chat_id 私聊 = `private:{账号wxid}:{会话wxid}`，群聊 = `group:{账号wxid}:{群wxid}`；账号=isSend=1 的 senderUsername
- timestamp 是消息级唯一身份线索；跨模块匹配必须带内容或使用哈希映射，禁止裸时间戳比对

## 测试 case（data 版本 dev_pool/fixed_test.jsonl）
{case_id, chat_type, chat_name, context: [{sender,text,is_self,timestamp}], human_reply: [str], source_message_id, source_chat_id?}
- human_reply 是原始消息文本，无发送者前缀
- is_self / timestamp 来自原始消息，不根据昵称猜测。RPA Judge 遇旧 case 缺少这些字段时拒绝评分。
- WeFlow 的 source_chat_id 沿用 RPA 的 `chat_` + 归一化导出文件名 SHA-256 前 10 位，
  用于已冻结的群成员特征；不是生成器的 chat_id，也不从测试答案反查。
- 已有实例用 `import_wechat.py --judge-only --adopt` 按 source_message_id 核对并补齐字段，
  生成新数据快照；题目 ID、上下文正文、答案、分集和风格池不变。

## RPA Judge 快照
- config.json：`mode=rpa_corrected_pairwise`，llm 保存 provider/model/reasoning_effort/
  timeout_seconds/codex_cli_version；correction_threshold 保存来源阈值。
- prompt.md / profile.json / reference.json / correction.json / source_judge.json 全部随版本冻结；
  config 记录资产及推理代码 SHA-256，meta 记录 RPA 基线 ID、来源路径与采用日期。
- 模型与参考资产只读复制，不重新训练，不导入 RPA 训练集或固定集答案；推理不依赖 RPA 目录。
- Judge 评估包同时保留 chat_type、chat_name、source_chat_id 和完整 context 元数据。

## few-shot 池行（data 版本 fewshot_pool.jsonl）
{id, context: ["sender: text"], reply: [str], relationship: group|private, chat_name,
 source_message_id, source_provenance, timestamp}
- context 带 "sender: " 前缀；任何文本匹配必须先剥前缀归一化
- 可信来源仅 explicit_human_marker / before_automation_cutoff；report.json 的
  examples_sha256 必须等于文件现算 hash，review_status=approved 才可用

## 实验 cases.jsonl
- 生成器：{case_id, status: ok|failed, context, human_reply, baseline/candidate_replies,
  identified_baseline/candidate, flip_verified{...}}
- Judge：{case_id, status, baseline_correct, candidate_correct, flip_verified,
  baseline/candidate_identified_final}
- 约定：每题记录原子完整（初测+补测同条）；重试 = 同 case_id 新行；每题取最后一条

## 冒烟失败的后续验证
- 旧 state 可追加 `resolution: {verified_by, reason, verified_at}`，指向同一实例中的成功冒烟。
- 必须核对：验证任务已结束且零失败，覆盖原失败 case ID，原题上下文正文与真人答案相同。
- 原 state 的 status/verdict/metrics 和 cases.jsonl 保持历史事实；resolution 只说明
  问题已在新配置下验证，不把旧失败改成成功，不用于正式评测或晋升。
- 后台分别展示待处理失败与已验证的历史失败，并提供页内详情和关联任务链接。
