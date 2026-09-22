# 安全策略

- 发现安全漏洞请通过 GitHub Private Vulnerability Reporting 或仓库主页
  提供的渠道私下报告，勿公开开 issue。
- 响应承诺：确认后 48 小时内回复，修复随下一个补丁版本发布并署名致谢。
- 红线：`instances/`、`.env`、密钥类内容永远不接受进入仓库
  （`scripts/check_public.py` 会在 CI 强制拦截）。
