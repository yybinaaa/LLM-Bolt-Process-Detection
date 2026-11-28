# 实验日志

正式实验的结果登记在这里，每次运行一行。每次 `boltcheck run` 或 `boltcheck evaluate` 结束时都会打印这一行（同时保存在 `runs/<运行 ID>/log_row.md`），直接粘贴到表格末尾即可。登记规则见 [CONTRIBUTING.md](../CONTRIBUTING.md#实验运行与登记)。

- 提交号后带 `*` 表示运行时工作区有未提交改动，结果不作为正式结论。
- 多次重复的运行，指标为均值 ± 标准差（百分点）。
- 第 1 组（程序规则基线）是流水线自检，所有准确率都应为 100%。
- 不同提示词版本的结果不能直接比较；"提示词"列记录运行时的提示词版本（`PROMPT_VERSION`）。
- 代码 0.9.0 起，区块抄写准确率逐个区块统计（缺失或形状不对的区块按全错计）。提示词 v1.5 试跑按此重评，进入第二阶段的 2 条所抄区块约 14% 正确。
- 代码 0.8.0 起，试跑改用 `configs/pilot.toml` 中固定的 8 条样本（合规、不合规各 4 条），与此前 `--limit 5`（4 条合规、1 条不合规）的判断类指标不能直接比较；区块最大数准确率在第 5 组矩阵不是 1～64 排列时也统计，提示词 v1.4 的两次试跑按此重评，全对比例均为 50%。
- 代码 0.7.1 起，复原类回复的 cells 须按模板的格子顺序填写，否则记为格式无效，报告增加"cells 未按模板顺序的比例"。按此重评提示词 v1.4 试跑，结果不变。
- 代码 0.7.0 起报告增加"逐格清单准确率（宽松提取，辅助）"：提示词 v1.3 试跑按此重评为 96.9%，说明复原内容基本正确，0 分来自 JSON 格式错误。
- 代码 0.6.0 起修正评分口径：格式无效的样本，判断类指标一律按答错计。此前登记的提示词 v1.2 试跑按新口径重评，合规判断准确率为 0.0%（登记值 20.0% 按代码 0.5.0 计算，保留原样），见 CHANGELOG。
- 试跑必须在备注中注明样本限制，以下标注为“试跑”的记录仅用于排查流程与输出，不作为正式实验结论。

| 日期 | 运行 ID | 提交 | 数据 | 提示词 | 组 | 模型 / 配置 | 重复 | 格式有效 | 判断准确 | 检出率 | 误报率 | 首违步 | 整矩阵 | 自洽 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | [20261115-175155_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-175155_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | 8c866a3 | v1.1 | v1.1 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），非正式结果；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-175155) |
| 2026-11-15 | [20261115-184124_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-184124_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | af7ac79 | v1.1 | v1.2 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 20.0% | 100.0% | 25.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.2；指标按代码 0.5.0 原样登记，计分说明见报告；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-184124) |
| 2026-11-15 | [20261115-190725_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-190725_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | ec71430 | v1.1 | v1.3 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.3；第一阶段均无可解析矩阵，第二阶段未调用；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-190725) |
| 2026-11-15 | [20261115-200224_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-200224_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | a85f2ab | v1.1 | v1.4 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.4；自动试跑；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-200224) |
| 2026-11-15 | [20261115-210203_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-210203_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | 2b95ed2 | v1.1 | v1.4 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.4；自动试跑；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-210203) |
| 2026-11-15 | [20261115-213225_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-213225_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | 90d34ff | v1.1 | v1.5 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.5；自动试跑；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-213225) |
| 2026-11-15 | [20261115-220216_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-220216_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | 971b168 | v1.1 | v1.6 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.6；自动试跑；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-220216) |
| 2026-11-15 | [20261115-233234_pipeline_qwen3-4b-instruct-2507-int4_greedy](experiments/20261115-233234_pipeline_qwen3-4b-instruct-2507-int4_greedy.md) | 3cd514f | v1.1 | v1.6 | pipeline | qwen3-4b-instruct-2507-int4 / greedy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 试跑（limit=5），提示词 v1.6；自动试跑；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261115-233234) |
| 2026-11-28 | [20261128-133749_judge_deepseek_api](experiments/20261128-133749_judge_deepseek_api.md) | f7b1354 | v1.1 | v1.6 | judge | deepseek / api | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | — | — | 10 条试看（大模型10条），非正式结论；开启思考；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261128-133749) |
| 2026-11-28 | [20261128-134509_pipeline_deepseek_api](experiments/20261128-134509_pipeline_deepseek_api.md) | f7b1354 | v1.1 | v1.6 | pipeline | deepseek / api | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 10 条试看（大模型10条），非正式结论；开启思考；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261128-134509) |
| 2026-11-28 | [20261128-142020_judge_deepseek_api-nothink](experiments/20261128-142020_judge_deepseek_api-nothink.md) | f7b1354 | v1.1 | v1.6 | judge | deepseek / api-nothink | 1 | 100.0% | 90.0% | 80.0% | 0.0% | 40.0% | — | — | 10 条试看（大模型10条），非正式结论；关闭思考对照；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261128-142020) |
| 2026-11-28 | [20261128-142036_pipeline_deepseek_api-nothink](experiments/20261128-142036_pipeline_deepseek_api-nothink.md) | f7b1354 | v1.1 | v1.6 | pipeline | deepseek / api-nothink | 1 | 100.0% | 90.0% | 80.0% | 0.0% | 20.0% | 100.0% | 60.0% | 10 条试看（大模型10条），非正式结论；关闭思考对照；[完整归档](https://github.com/yybinaaa/LLM-Bolt-Process-Detection/releases/tag/exp-20261128-142036) |
