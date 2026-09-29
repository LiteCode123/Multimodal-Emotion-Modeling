# 问题一：官方文本与多模态特征构建

## 输入和目标

附件 1 提供 100 条视频及 `label-100.xlsx`。以样本 ID 核对官方转写、情感标签和媒体；为每条视频构建 50 个时间单元，以 `(sample_id, bin_id)` 作唯一键。这里的情感标签是随结果保留的标注字段，**不是 BERT 的输入特征**。

## 最终采用的方法

1. 用有效媒体时长划分 50 个 bin；在各 bin 内计算声学统计、低层视觉统计和人脸几何特征。实现位于 [`q1_media.py`](../src/features/q1_media.py) 与 [`q1_face.py`](../src/features/q1_face.py)。
2. 将官方转写切成词；`faster-whisper small.en` 的逐词识别只提供时间锚点。规范化官方词与 ASR 词做 LCS 匹配，未匹配词在锚点间插值；ASR 为空、匹配不足或时间逆序时，对该条样本标记比例回退。
3. 用 `bert-base-uncased` 对**官方转写原文**编码，将 WordPiece 向量按字符位置聚合为词向量，再按词区间与 bin 的重叠时长聚合为 768 维文本特征。无词覆盖的 bin 置零并记录 `text_present=0`。实现位于 [`q1_official_text.py`](../src/features/q1_official_text.py)。
4. 保留音频、视觉、人脸与标签字段，替换文本特征列，形成 100 × 50 行、843 列的最终表。

公开仓库保留[匿名汇总](../results/public_summary.json)和[schema](../results/q1/official/outputs/q1_official_text_schema.json)，可检查字段与覆盖口径。含样本编号或官方转写的 100 条摘要、验证明细和完整 5000 × 843 特征表留在本地，不进入 Git 提交。

## 复现入口与限制

`python scripts/run_q1.py` 在本地准备好的 843 列媒体与人脸基表上重建官方文本版本，输出到 `runs/q1/official/`。输入位置见[数据说明](../data/README.md)。

4414/5000 个 bin 有本轮生成的文本表示；其中 26 条样本采用比例回退，产生的 1300 个 bin 只是**估计覆盖**。词匹配率和 bin 覆盖率都不是人工标注的词时间准确率。原声学实现还存在媒体有效时长与解码音频长度的差异，因此本项目不宣称三模态达到精确物理秒级对齐。
