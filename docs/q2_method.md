# 问题二：局部模态缺失下的鲁棒预测

## 数据与评价协议

使用官方附件 2 `aligned_50.pkl`：`train/valid/test = 3395/728/727`，模型共同输入为 `text_bert[3,50]`、`audio[50,74]`、`vision[50,35]`。冻结 BERT 将 `text_bert` 重建为文本向量；训练集拟合模型与预处理，验证集选择模型与校正，test 仅在锁定后评估。附件 3 的 30 条样本无标签，只做最终推理。[冻结数据协议](../results/q2/protocol/outputs/q2_protocol.json)保留详细字段规则。

## 缺失模拟与模型

[`q2_missingness.py`](../src/data/q2_missingness.py)对原始特征**另存显式人工 mask**，可遮挡文本、音频、视觉或同步音视频的连续位置。固定的 4 种模式 × 3 个名义缺失率 × 3 个位置组成 36 组验证条件；原始全零行不自动视作人工缺失。

传统基线使用冻结 BERT、特征拼接、仅用训练集拟合的 StandardScaler、Balanced SVC 和 Ridge。[`q2_mafmt.py`](../src/models/q2_mafmt.py) 实现主模型 MAF-MT：三模态投影、基于可用性的动态门控、BiGRU 和时间注意力，共享表示后分别输出三类极性与带符号强度。训练用缺失增强、分类/回归联合损失和一致性约束；随机种子与参数见 [`q2.yaml`](../configs/q2.yaml)。

附件 3 最终版本使用经未对齐原文核对的 `[UNK]` 缺失表示，以及只凭附件 2 `valid` 选择的中性 logit 减量 `0.10`。这一步不重训模型，不用附件 3 的标签或预测类别分布选参。代码见 [`q2_inference.py`](../src/models/q2_inference.py)与[`q2_final_calibration.py`](../src/models/q2_final_calibration.py)。

## 结果

| 附件 2 独立 test，727 条 | 基线 | MAF-MT |
|---|---:|---:|
| Accuracy | 0.6162 | **0.6437** |
| Macro-F1 | 0.5689 | **0.6303** |
| MAE ↓ | 0.7628 | **0.6201** |
| Pearson ↑ | 0.5424 | **0.6849** |

完整小数见 [test 指标 CSV](../results/q2/test/outputs/q2_final_test_metrics.csv)。[36 组曲线](../results/q2/model/figures/main_vs_baseline_missing_rate_curves.png)和[消融摘要](../results/q2/ablation/outputs/q2_ablation_36grid_summary.csv)用于分析缺失类型、时长和位置的影响。附件 3 完成 30 条无标签推理，数量记录在[匿名汇总](../results/public_summary.json)中；逐样本预测留在本地，不能据此计算准确率。

`python scripts/train_q2.py` 在 `runs/q2/model/` 重训；`python scripts/predict_q2.py` 在 `runs/q2/` 重做附件 3 原版与最终校正。二者均需[官方数据和 BERT 权重](../data/README.md)，重新运行不会覆盖 `results/` 的冻结模型。
