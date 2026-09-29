# 匿名结果

Git 提交仅包含汇总统计、评价指标、图和用于核验的轻量冻结模型。原始转写、可回溯样本编号、逐样本预测、完整特征表与审计明细保留在本地，不纳入提交。

| 部分 | 从哪里读起 | 内容 |
|---|---|---|
| 问题一 | [`public_summary.json`](public_summary.json) | 100 条样本、5000 × 843 特征表的匿名构建统计与覆盖率 |
| 问题二 | [`q2/test/outputs/q2_final_test_metrics.csv`](q2/test/outputs/q2_final_test_metrics.csv) | 基线与主模型的独立 test 指标；另附 36 组缺失条件的汇总指标与曲线 |
| 问题三 | [`q3/model/outputs/q3_final_metrics.json`](q3/model/outputs/q3_final_metrics.json) | 冻结 GMF-MT 的 valid/test 指标；另附解释忠实性汇总 |

`q2/model/models/q2_mafmt_best.pt` 与 `q3/model/outputs/q3_final_model.pt` 是轻量冻结模型。问题二完整推理另需外部 BERT 权重。Git 忽略 `results/` 下的新文件，新增公开结果需逐项检查后明确纳入提交。
