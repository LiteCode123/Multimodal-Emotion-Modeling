# 问题三：多任务预测与解释核验

## 预测模型

使用官方附件 2 `aligned_50.pkl` 的同一官方 `train/valid/test` 划分。三模态特征为 `text[50,768]`、`audio[50,74]`、`vision[50,35]`；由 `text_bert` 的注意力位置构造结构内容 mask，排除边界与 padding。训练集内容位置拟合均值和标准差，之后进行 mask 均值池化。

GMF-MT 分别编码三模态，用门控加权融合，共享表示上进行三类情感极性分类和带符号强度回归。验证集选择结构与 epoch；独立 test 在模型锁定后评价。实现见 [`q3_gmf.py`](../src/models/q3_gmf.py)，冻结设置在 [`q3.yaml`](../configs/q3.yaml)。test 727 条的 Accuracy **0.6465**、Macro-F1 **0.6244**、MAE **0.6767**、Pearson **0.6255**，精确值见 [指标 JSON](../results/q3/model/outputs/q3_final_metrics.json)。

## 解释和验证

模态级扰动以训练统计量对应的中性基线替换单个模态，观察原预测类别概率和回归输出变化。位置级扰动在结构有效位置逐位进行；各模态归一化并平滑重要性，再按累计贡献提取关键区间。代码在 [`src/explainability/`](../src/explainability/)；解释的是冻结模型对输入的**敏感性**，不是对情绪的因果归因。

Key–Random–Low 验证以等量随机和低重要性位置作对照，比较分类概率下降与回归输出变化；汇总见 [忠实性结果](../results/q3/faithfulness/outputs/q3_faithfulness_summary.json)。附件 4 的 20 条无标签样本由冻结模型生成预测与解释，数量记录在[匿名汇总](../results/public_summary.json)中；逐样本结果留在本地。`python scripts/explain_q3.py` 需官方对齐附件 4、冻结模型及未发布的逐样本审计结果，输出到 `runs/q3/inference/`。

## 原始证据回看边界

20 条样本的关键文本片段可由原文字符位置核实。音频与视觉可通过 ASR 词时间和真实解码帧 PTS 给出**回看参考**，但官方附件没有公布预计算特征行对应的采样时钟、裁剪起点和窗口。因此本地原始证据审计中的音视频时间与帧不被写成“已核实的特征提取位置”。局部文本片段若仅含标点，也不能解释为有语义的情感证据。
