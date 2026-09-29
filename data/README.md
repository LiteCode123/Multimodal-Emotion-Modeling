# 数据与外部模型

仓库只保留小型验证结果和轻量冻结模型。官方视频、特征 `.pkl`、BERT/ASR 权重以及问题一的媒体中间表放在本地 `data/`，由 `.gitignore` 排除。

主要放置路径：

```text
data/raw/E题数据/
  附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/
    label-100.xlsx
    <视频ID>/*.mp4
  附件2-数据集特征文件/aligned_50.pkl
  附件3-模态缺失特征样本/对齐版本/*.pkl
  附件3-模态缺失特征样本/未对齐版本/*.pkl
  附件4-可解释专项视频样本与特征文件/
    附件4-可解释专项视频样本与特征文件/对齐版本/*.pkl
data/models/bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594/
  model.safetensors
  config.json
  tokenizer.json
data/intermediate/q1/multimodal_features_50bin.csv
```

问题二要求 BERT revision `86b5e0934494bd15c9632b12f734a8a67f723594`，`model.safetensors` 的 SHA-256 为 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。问题一另外需要本地 `faster-whisper small.en` 与 `bert-base-uncased`；问题三原始证据回看另需 `base.en` ASR 权重和附件 4 原视频。

问题一的 `data/intermediate/q1/multimodal_features_50bin.csv` 是由 [`q1_media.py`](../src/features/q1_media.py) 与 [`q1_face.py`](../src/features/q1_face.py) 对原媒体提取的 843 列基表。`scripts/run_q1.py` 在此基础上使用官方转写重建文本列。该中间表已从原工作区复制到本地新目录，后续 Git 会忽略它。重建时还需原视频，因为 ASR 时间锚点需要重新提取。

`.pkl` 使用 Python pickle 格式，只加载可信的官方数据包。配置文件中的数据路径是此目录的新位置；`results/` 保留原实验协议的字节内容，`src/data/paths.py` 在运行时将其中记录的旧路径映射到这里。
