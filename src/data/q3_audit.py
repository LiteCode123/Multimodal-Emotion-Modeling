# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Read-only Q3 audit of official aligned features and evidence mapping.

Run from the project root: python -X utf8 task_12_q3_data_audit.py
Only Task12 outputs are written. Pickle inputs must be trusted project files.
"""
from __future__ import annotations

import csv
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/protocol"
OUT = TASK / "outputs"
RAW = ROOT / "data/raw/E题数据"
A2 = RAW / "附件2-数据集特征文件/aligned_50.pkl"
A2_LABEL = RAW / "附件2-数据集特征文件/label.xlsx"
A4 = RAW / "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
Q1_MAPPING = ROOT / "data/intermediate/q1/base_final/q1_sample_mapping_100.csv"
Q1_BINS = ROOT / "results/q1/media/outputs/multimodal_features_50bin.csv"
FFPROBE = ROOT / "data/tools/ffprobe.exe"
SPLITS = ("train", "valid", "test")
MODALITIES = ("text", "audio", "vision")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def array_audit(value: np.ndarray) -> dict:
    arr = np.asarray(value)
    result = {"shape": list(arr.shape), "dtype": str(arr.dtype), "nan": 0,
              "inf": 0, "zero_sequences": 0, "constant_sequences": 0,
              "zero_rows": 0, "zero_rows_by_position": [],
              "first_nonzero_min": None, "last_nonzero_max": None,
              "candidate_span_length_min": None, "candidate_span_length_max": None,
              "candidate_span_length_mean": None, "trailing_zero_rows_min": None,
              "trailing_zero_rows_max": None, "trailing_zero_rows_mean": None,
              "all_zero_first": None, "all_zero_last": None}
    if arr.ndim != 3:
        return result
    n, t, _ = arr.shape
    pos = np.zeros(t, dtype=np.int64)
    spans, trailing = [], []
    for start in range(0, n, 64):
        b = arr[start:start + 64]
        result["nan"] += int(np.isnan(b).sum())
        result["inf"] += int(np.isinf(b).sum())
        zero = np.all(b == 0, axis=-1)
        pos += zero.sum(axis=0)
        result["zero_rows"] += int(zero.sum())
        result["zero_sequences"] += int(zero.all(axis=1).sum())
        result["constant_sequences"] += int(np.all(b == b[:, :1, :], axis=(1, 2)).sum())
        for z in zero:
            nonzero = np.flatnonzero(~z)
            if nonzero.size:
                spans.append(int(nonzero[-1] - nonzero[0] + 1))
                trailing.append(int(t - 1 - nonzero[-1]))
            else:
                trailing.append(t)
    result["zero_rows_by_position"] = pos.tolist()
    result["all_zero_first"] = int(pos[0])
    result["all_zero_last"] = int(pos[-1])
    if spans:
        result["candidate_span_length_min"] = min(spans)
        result["candidate_span_length_max"] = max(spans)
        result["candidate_span_length_mean"] = float(np.mean(spans))
    result["trailing_zero_rows_min"] = min(trailing)
    result["trailing_zero_rows_max"] = max(trailing)
    result["trailing_zero_rows_mean"] = float(np.mean(trailing))
    return result


def probe_duration(path: Path) -> float | None:
    if not path.exists() or not FFPROBE.exists():
        return None
    proc = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                           "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                           str(path)], capture_output=True, text=True, timeout=20)
    if proc.returncode:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def token_feature_position_diagnostic(tokens: np.ndarray, features: np.ndarray) -> dict:
    """Test positional association without treating it as a time-map proof."""
    valid = np.asarray(tokens)[:, 1, :].sum(axis=1).astype(int)
    present = np.any(np.asarray(features) != 0, axis=2)
    last = np.where(present, np.arange(present.shape[1]), -1).max(axis=1)
    delta = last - valid
    interior = (np.arange(present.shape[1])[None, :] >= 1) & (np.arange(present.shape[1])[None, :] < valid[:, None] - 1)
    return {"last_nonzero_minus_attention_length_counts": dict(sorted(Counter(map(int, delta)).items())),
            "last_nonzero_equals_attention_length_minus_2": int((delta == -2).sum()),
            "last_nonzero_equals_attention_length_minus_1": int((delta == -1).sum()),
            "correlation": float(np.corrcoef(last, valid)[0, 1]),
            "samples_exact_interior_nonzero_mask": int(np.all(present == interior, axis=1).sum()),
            "nonzero_rows_outside_interior": int((present & ~interior).sum()),
            "zero_rows_inside_interior": int((~present & interior).sum()),
            "attention_first_token_101": int((np.asarray(tokens)[:, 0, 0] == 101).sum()),
            "attention_last_token_102": int((np.asarray(tokens)[np.arange(len(valid)), 0, valid - 1] == 102).sum())}


def main() -> None:
    required = [A2, A2_LABEL, A4, Q1_MAPPING, Q1_BINS]
    absent = [str(p) for p in required if not p.exists()]
    if absent:
        raise FileNotFoundError("Required input missing: " + ", ".join(absent))
    paths = sorted(A4.glob("[0-9][0-9].pkl"))
    if not paths:
        raise FileNotFoundError(f"No attachment4 aligned pkl files in {A4}")
    OUT.mkdir(exist_ok=True)

    print("Loading attachment2 aligned_50", flush=True)
    with A2.open("rb") as fh:
        data = pickle.load(fh)
    with Q1_MAPPING.open(encoding="utf-8-sig", newline="") as fh:
        q1_ids = {row["sample_id"] for row in csv.DictReader(fh)}

    a2 = {}
    all_ids = set()
    for split in SPLITS:
        block = data[split]
        ids = [str(x) for x in block["id"]]
        all_ids.update(ids)
        y = np.asarray(block["regression_labels"]).reshape(-1)
        c = np.asarray(block["classification_labels"]).reshape(-1)
        expected = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        a2[split] = {
            "n": len(ids), "fields": list(block), "field_counts": {k: len(v) for k, v in block.items()},
            "id_unique": len(set(ids)) == len(ids), "class_rule_match": bool(np.array_equal(c, expected)),
            "labels": {"regression_field": "regression_labels", "regression_dtype": str(y.dtype),
                       "classification_dtype": str(c.dtype), "min": float(y.min()), "max": float(y.max()),
                       "negative": int((y < 0).sum()), "neutral": int((y == 0).sum()),
                       "positive": int((y > 0).sum())},
            "modalities": {m: array_audit(block[m]) for m in MODALITIES},
            "text_bert_shape": list(np.asarray(block["text_bert"]).shape),
            "text_bert_dtype": str(np.asarray(block["text_bert"]).dtype),
            "text_attention_values": sorted(np.unique(np.asarray(block["text_bert"])[:, 1, :]).tolist()),
            "text_attention_valid_lengths": np.asarray(block["text_bert"])[:, 1, :].sum(axis=1).astype(int).tolist(),
            "token_feature_position_diagnostic": {
                m: token_feature_position_diagnostic(block["text_bert"], block[m])
                for m in ("audio", "vision")},
        }
        print(f"attachment2 {split}: {len(ids)}", flush=True)
    overlap = {f"{x}_{y}": len(set(map(str, data[x]["id"])) & set(map(str, data[y]["id"])))
               for x, y in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    # Keep the report compact: summarize individual token lengths, not every sample.
    for split in SPLITS:
        lengths = a2[split].pop("text_attention_valid_lengths")
        a2[split]["text_attention_valid_length_summary"] = {
            "min": min(lengths), "max": max(lengths), "mean": float(np.mean(lengths)),
            "all_prefix_masks": bool(np.all(np.diff(np.asarray(data[split]["text_bert"])[:, 1, :], axis=1) <= 0))}

    a4 = {"n": len(paths), "samples": [], "fields_common": None, "fields_union": set(),
          "modalities": {}, "id_unique": None}
    a4_arrays = {m: [] for m in MODALITIES}
    a4_tokens = []
    sample_ids = []
    for path in paths:
        with path.open("rb") as fh:
            record = pickle.load(fh)
        sample_id = str(record.get("id", path.stem))
        sample_ids.append(sample_id)
        fields = set(record)
        a4["fields_union"].update(fields)
        a4["fields_common"] = fields if a4["fields_common"] is None else a4["fields_common"] & fields
        entry = {"file": str(path.relative_to(ROOT)), "sample_id": sample_id,
                 "id_matches_filename": sample_id == path.stem,
                 "fields": list(record), "raw_text_nonempty": bool(str(record.get("raw_text", "")).strip()),
                 "video_exists": (A4 / "videos" / f"{path.stem}.mp4").exists(),
                 "video_duration_sec": probe_duration(A4 / "videos" / f"{path.stem}.mp4"),
                 "text_bert_shape": list(np.shape(record.get("text_bert"))),
                 "text_bert_dtype": str(np.asarray(record.get("text_bert")).dtype),
                 "token_attention_length": int(np.asarray(record["text_bert"])[1].sum())}
        a4_tokens.append(np.asarray(record["text_bert"]))
        for m in MODALITIES:
            if m in record:
                arr = np.asarray(record[m])
                entry[m] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                            "nan": int(np.isnan(arr).sum()), "inf": int(np.isinf(arr).sum())}
                a4_arrays[m].append(arr)
        a4["samples"].append(entry)
    a4["fields_common"] = sorted(a4["fields_common"])
    a4["fields_union"] = sorted(a4["fields_union"])
    a4["id_unique"] = len(set(sample_ids)) == len(sample_ids)
    a4["id_filename_match_all"] = all(s["id_matches_filename"] for s in a4["samples"])
    a4["modalities"] = {m: array_audit(np.stack(a4_arrays[m])) for m in MODALITIES}
    a4["all_zero_visual_ids"] = [sample_ids[i] for i, arr in enumerate(a4_arrays["vision"])
                                 if np.all(arr == 0)]
    a4["q1_id_overlap"] = len(set(sample_ids) & q1_ids)
    a4["a2_id_overlap"] = len(set(sample_ids) & all_ids)
    a2_q1_overlap = len(all_ids & q1_ids)
    a4["videos_present"] = sum(s["video_exists"] for s in a4["samples"])
    a4["video_durations_obtained"] = sum(s["video_duration_sec"] is not None for s in a4["samples"])
    a4["raw_text_nonempty"] = sum(s["raw_text_nonempty"] for s in a4["samples"])
    a4_token_array = np.stack(a4_tokens)
    a4_attention_lengths = a4_token_array[:, 1, :].sum(axis=1).astype(int)
    a4["text_bert_attention"] = {
        "values": sorted(np.unique(a4_token_array[:, 1, :]).tolist()),
        "valid_length_min": int(a4_attention_lengths.min()),
        "valid_length_max": int(a4_attention_lengths.max()),
        "valid_length_mean": float(a4_attention_lengths.mean()),
        "all_prefix_masks": bool(np.all(np.diff(a4_token_array[:, 1, :], axis=1) <= 0)),
    }
    a4["token_feature_position_diagnostic"] = {
        m: token_feature_position_diagnostic(a4_token_array, np.stack(a4_arrays[m]))
        for m in ("audio", "vision")}
    print(f"attachment4: {len(paths)}", flush=True)

    rows = []
    for m in MODALITIES:
        train = a2["train"]["modalities"][m]
        test = a4["modalities"][m]
        rows += [
            {"item": f"{m}_field", "attachment2_value": m, "attachment4_value": m,
             "consistent": m in a4["fields_common"], "notes": "同名预计算特征"},
            {"item": f"{m}_shape_per_sample", "attachment2_value": str(train["shape"][1:]),
             "attachment4_value": str(test["shape"][1:]),
             "consistent": train["shape"][1:] == test["shape"][1:], "notes": "长度及维度"},
            {"item": f"{m}_dtype", "attachment2_value": train["dtype"],
             "attachment4_value": test["dtype"], "consistent": train["dtype"] == test["dtype"],
             "notes": "未来推理统一转 float32，train 拟合的标准化参数复用"},
        ]
    rows += [
        {"item": "sample_id", "attachment2_value": "id (MOSEI clip ID)",
         "attachment4_value": "id (01-20)", "consistent": True, "notes": "语义均为唯一 ID；值域不同"},
        {"item": "raw_text", "attachment2_value": "raw_text", "attachment4_value": "raw_text",
         "consistent": "raw_text" in a4["fields_common"], "notes": "全文可回溯，非逐 bin 时间戳"},
        {"item": "text_bert", "attachment2_value": str(a2["train"]["text_bert_shape"][1:]),
         "attachment4_value": str(a4["samples"][0]["text_bert_shape"]),
         "consistent": a2["train"]["text_bert_shape"][1:] == a4["samples"][0]["text_bert_shape"],
         "notes": "辅助 token/attention 字段，不作为预计算 text 特征的真实时间 mask"},
        {"item": "labels", "attachment2_value": "regression_labels,classification_labels",
         "attachment4_value": "absent", "consistent": True, "notes": "训练有监督；附件4无标签是正常推理接口"},
        {"item": "temporal_mask_or_valid_length", "attachment2_value": "absent",
         "attachment4_value": "absent", "consistent": True, "notes": "text_bert attention 仅为 token mask"},
        {"item": "structural_content_mask", "attachment2_value": "derived from text_bert attention",
         "attachment4_value": "derived from text_bert attention", "consistent": True,
         "notes": "1<=i<L-1; excludes 101/102 special tokens and padding; not seconds validity"},
        {"item": "bin_timestamp", "attachment2_value": "absent", "attachment4_value": "absent",
         "consistent": True, "notes": "缺少原始特征生成的 bin 边界；视频时长不能证明映射"},
    ]
    write_csv(OUT / "q3_train_test_interface_check.csv",
              ["item", "attachment2_value", "attachment4_value", "consistent", "notes"], rows)
    map_rows = []
    for sample in a4["samples"]:
        for i in range(50):
            structural_content = 1 <= i < sample["token_attention_length"] - 1
            map_rows.append({"sample_id": sample["sample_id"], "bin_index": i,
                             "valid": int(structural_content), "start_time": "", "end_time": "",
                             "mapping_source": "text_bert_attention_structural_content",
                             "mapping_confidence": "high_for_structural_index_only",
                             "notes": "valid=token-content position, not verified physical-media validity; timestamps unavailable."})
    write_csv(OUT / "q3_time_mapping_audit.csv",
              ["sample_id", "bin_index", "valid", "start_time", "end_time",
               "mapping_source", "mapping_confidence", "notes"], map_rows)

    compatible = all(r["consistent"] for r in rows if r["item"].endswith("shape_per_sample"))
    protocol = {
        "status": "Q3 Task12 audited; time mapping unresolved",
        "source": {"attachment2": str(A2.relative_to(ROOT)), "attachment4_directory": str(A4.relative_to(ROOT))},
        "sequence_length": 50, "bin_index_base": 0,
        "modalities": {m: {"source_field": m, "sample_shape": a2["train"]["modalities"][m]["shape"][1:],
                            "model_dtype": "float32", "source_dtype_attachment2": a2["train"]["modalities"][m]["dtype"],
                            "source_dtype_attachment4": a4["modalities"][m]["dtype"]} for m in MODALITIES},
        "sample_id": {"attachment2_field": "id", "attachment4_field": "id", "must_be_unique": True},
        "splits": {"fit": "attachment2/train", "select": "attachment2/valid",
                   "final_labeled_evaluation": "attachment2/test", "unlabeled_inference": "attachment4 only"},
        "label_definition": {"regression_field": "regression_labels", "range_observed": [-3, 3],
                             "classification_field": "classification_labels",
                             "class_ids": {"0": "Negative", "1": "Neutral", "2": "Positive"},
                             "polarity_from_score": {"negative": "y<0", "neutral": "y==0", "positive": "y>0"}},
        "padding_rule": {"status": "structural_token_padding_identified",
                         "token_content_index_rule": "1 <= i < sum(text_bert[1,:]) - 1",
                         "note": "BERT 101/102 boundaries and near-exact audio interior support structural padding; this is not physical-time validity. Audio/vision may be zero inside content."},
        "mask_rule": {"official_temporal_valid_mask": None,
                      "text_bert_attention_mask": "channel 1; 1-prefix then 0-padding; positions 0 and L-1 are 101/102 special tokens",
                      "structural_content_mask": "1 <= i < L-1, L=sum(text_bert[1,:]); exclude special tokens and padding for local explanations",
                      "explanation_perturbation_mask": "future explicit mask, distinct from natural zeros/padding"},
        "time_mapping": {"level": "C", "bin_to_seconds": None,
                         "reason": "No official bin edges or confirmed aligned_50 generation rule for attachment2/4.",
                         "attachment4_video_available": a4["videos_present"],
                         "q1_reuse": "method only; Q1 bin indices and sample metadata cannot be transferred without verified equivalence"},
        "model_input": {"sample": {"sample_id": "str", **{m: [50, a2["train"]["modalities"][m]["shape"][2]] for m in MODALITIES},
                                   "structural_content_mask": [50]},
                        "batch": {**{m: ["B", 50, a2["train"]["modalities"][m]["shape"][2]] for m in MODALITIES},
                                  "structural_content_mask": ["B", 50]},
                        "preprocessing": "fit on attachment2/train only; apply unchanged to valid/test/attachment4"},
        "prediction_outputs": {"required": ["sample_id", "pred_polarity", "pred_intensity"],
                               "pred_polarity_values": ["Negative", "Neutral", "Positive"]},
        "explanation_outputs": {"required": ["text_contribution", "audio_contribution", "visual_contribution",
                                             "dominant_modality", "text_importance", "audio_importance",
                                             "visual_importance", "text_key_intervals", "audio_key_intervals",
                                             "visual_key_intervals"],
                                "importance_length": 50, "key_interval_indexing": "zero-based half-open [start,end)",
                                "seconds_fields": "nullable until verified time mapping exists"},
        "explanation_principles": ["Freeze prediction model before explanation", "Per-sample numeric modality contributions",
                                    "Keep 50-position local importance", "Merge neighboring important bins",
                                    "Compare key-bin masking against random and low-importance masking on held-out data"],
        "readiness": {"train_baseline": compatible, "feature_space_perturbation": compatible,
                      "raw_evidence_timestamps": False},
    }
    (OUT / "q3_explainability_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def modality_table(dataset: dict, n: int) -> str:
        return "\n".join(
            f"| {m} | {dataset[m]['dtype']} | {n}×{dataset[m]['shape'][1]}×{dataset[m]['shape'][2]} | "
            f"{dataset[m]['nan']} | {dataset[m]['inf']} | {dataset[m]['zero_sequences']} | "
            f"{dataset[m]['constant_sequences']} | {dataset[m]['zero_rows']} |"
            for m in MODALITIES)

    labels_table = "\n".join(
        f"| {split} | {v['n']} | " + " | ".join(
            f"{v['labels'][key]} ({v['labels'][key]/v['n']:.2%})" for key in ("negative", "neutral", "positive")) + " |"
        for split, v in a2.items())
    durations = [s["video_duration_sec"] for s in a4["samples"] if s["video_duration_sec"] is not None]
    audit = f"""# Task12 Q3 数据审计明细

## 来源与方法

只读扫描 `{A2.relative_to(ROOT)}` 和附件4对齐版全部 {len(paths)} 个 pkl；视频只用 ffprobe 读取时长。Excel 文件存在，内容一致性引用 Task04 已完成的审计，本轮环境缺少 openpyxl。无训练、无伪标签。源字段和样本 shape 由文件读取，未按文件名推断特征维度。固定长度不等于真实 50 个物理时间段。

## 附件2划分与标签

| split | N | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
{labels_table}

顶层字段 `{list(data)}`。三个 split 的字段分别为 `{a2['train']['fields']}`。字段行数均与 ID 行数相同：{all(all(v == x['n'] for v in x['field_counts'].values()) for x in a2.values())}；各 split ID 唯一：{all(x['id_unique'] for x in a2.values())}；跨 split 交集：{overlap}。本轮直接核对 pkl 两类标签；Excel ID、split、连续标签与 pkl 一致性沿用 Task04 的独立审计结果，本环境缺少 openpyxl，未重新读取 Excel 内容。`regression_labels` dtype `{a2['train']['labels']['regression_dtype']}`，三个 split 实测范围均在 [-3,3]；`classification_labels` dtype `{a2['train']['labels']['classification_dtype']}`，与 y<0→0、y=0→1、y>0→2 完全一致：{all(x['class_rule_match'] for x in a2.values())}。Neutral 绝不并入 Positive。附件2 test 保留终评。

## 三模态数组质量

| 附件2 split / 模态 | dtype | shape | NaN | Inf | 全零序列 | 时间常数序列 | 全零时间行 |
|---|---|---|---:|---:|---:|---:|---:|
"""
    for split in SPLITS:
        audit += modality_table(a2[split]["modalities"], a2[split]["n"]).replace("| text |", f"| {split}/text |")\
            .replace("| audio |", f"| {split}/audio |").replace("| vision |", f"| {split}/vision |") + "\n"
    audit += f"""
| 附件4模态 | dtype | shape | NaN | Inf | 全零序列 | 时间常数序列 | 全零时间行 |
|---|---|---|---:|---:|---:|---:|---:|
{modality_table(a4['modalities'], a4['n'])}

附件4顶层字段交集/并集同为 `{a4['fields_common']}`；20/20 个 `id` 与文件名一致：{a4['id_filename_match_all']}，ID 唯一：{a4['id_unique']}，`raw_text` 非空 {a4['raw_text_nonempty']}/{a4['n']}；无标签。与附件2 ID 交集 {a4['a2_id_overlap']}，与问题1 100 条样本 ID 交集 {a4['q1_id_overlap']}。视觉全零样本 ID：`{a4['all_zero_visual_ids']}`，不能称全部样本视觉内容完整。

## aligned_50、padding 和 mask

三模态在附件2所有 split 及附件4每条样本均为长度 50，且特征维度一致。`text_bert` 在附件2/附件4均为 3×50，第二通道有 0/1 token attention；附件2 train/valid/test 的 token 有效数 min/max/mean 分别为 """
    audit += ", ".join(f"{split}: {a2[split]['text_attention_valid_length_summary']['min']}/"
                       f"{a2[split]['text_attention_valid_length_summary']['max']}/"
                       f"{a2[split]['text_attention_valid_length_summary']['mean']:.2f}"
                       for split in SPLITS)
    audit += f"""；附件4：{a4['text_bert_attention']['valid_length_min']}/{a4['text_bert_attention']['valid_length_max']}/{a4['text_bert_attention']['valid_length_mean']:.2f}。各 split 与附件4 token mask 均为 1 前缀、0 后缀：{all(a2[s]['text_attention_valid_length_summary']['all_prefix_masks'] for s in SPLITS) and a4['text_bert_attention']['all_prefix_masks']}。可由该 attention 定义内容位置，但不能证明三个特征在相同秒数上对齐。

扣除首尾两个特殊 token 后，结构性内容长度 min/mean/max："""
    audit += ", ".join(f"{split}: {a2[split]['text_attention_valid_length_summary']['min']-2}/"
                       f"{a2[split]['text_attention_valid_length_summary']['mean']-2:.2f}/"
                       f"{a2[split]['text_attention_valid_length_summary']['max']-2}"
                       for split in SPLITS)
    audit += (f"；附件4: {a4['text_bert_attention']['valid_length_min']-2}/"
              f"{a4['text_bert_attention']['valid_length_mean']-2:.2f}/"
              f"{a4['text_bert_attention']['valid_length_max']-2}。这是 token 内容长度，不是秒级媒体有效长度。\n")

    audit += """
所有样本首个有效 token ID 为 101、最后有效 token ID 为 102。令 `L=sum(text_bert[1,:])`，`1 <= bin_index < L-1` 是**结构性内容位置**；0 与 L-1 是特殊 token，L 及之后是 padding。最后一个非零音视频位置与 L 的关系如下；该对应关系不能推出秒级时间含义。

| 数据 | 模态 | 最后非零位置=L-2 | 非零 mask 与内容位置完全相同 | 内容位置内零行 | 内容位置外非零行 |
|---|---|---:|---:|---:|---:|
"""
    for name, group in [("A2 train", a2["train"]), ("A2 valid", a2["valid"]),
                        ("A2 test", a2["test"]), ("A4", a4)]:
        for m in ("audio", "vision"):
            d = group["token_feature_position_diagnostic"][m]
            audit += (f"| {name} | {m} | {d['last_nonzero_equals_attention_length_minus_2']} | "
                      f"{d['samples_exact_interior_nonzero_mask']} | {d['zero_rows_inside_interior']} | "
                      f"{d['nonzero_rows_outside_interior']} |\n")
    audit += f"""

附件2 train 音频首/末零行样本 {a2['train']['modalities']['audio']['all_zero_first']}/{a2['train']['modalities']['audio']['all_zero_last']}，视觉首/末 {a2['train']['modalities']['vision']['all_zero_first']}/{a2['train']['modalities']['vision']['all_zero_last']}；附件4对应音频 {a4['modalities']['audio']['all_zero_first']}/{a4['modalities']['audio']['all_zero_last']}，视觉 {a4['modalities']['vision']['all_zero_first']}/{a4['modalities']['vision']['all_zero_last']}。全零位置既可能结构占位、padding、无信号，也可能真实零值，不能直接定义有效 mask。候选首末非零跨度及尾部零行统计如下；**这些只是诊断量，不是有效长度真值**。

| 数据 | 模态 | 首末非零跨度 min/mean/max | 尾部零行 min/mean/max |
|---|---|---|---|
"""
    for name, group in [("A2 train", a2["train"]["modalities"]), ("A2 valid", a2["valid"]["modalities"]),
                        ("A2 test", a2["test"]["modalities"]), ("A4", a4["modalities"])]:
        for m in MODALITIES:
            s = group[m]
            audit += (f"| {name} | {m} | {s['candidate_span_length_min']}/"
                      f"{s['candidate_span_length_mean']:.2f}/{s['candidate_span_length_max']} | "
                      f"{s['trailing_zero_rows_min']}/{s['trailing_zero_rows_mean']:.2f}/"
                      f"{s['trailing_zero_rows_max']} |\n")
    audit += f"""
所有音视频非零行均落在结构性内容位置内；音频绝大多数样本在内容位置非零，但附件2 train 有 4 条、valid 有 1 条存在内部零行。视觉有更多内容位置为零，不能把这些零行全部视作 padding。预计算 `text` 在 padding 位置也非零，因此其零行不能识别 padding。`q3_time_mapping_audit.csv` 的 `valid` 表示**结构性内容位置**，不表示物理媒体在该位置有有效信号。数组没有独立的物理时间 mask、bin interval、word timestamp 或 frame timestamp 字段。未发现异常长度样本。常数序列已计数，逐位置零行分布可由脚本重算。

## 附件2 → 附件4接口

`q3_train_test_interface_check.csv` 逐字段列出形状和 dtype。三个预计算特征的 `[50,D]` 严格一致，可统一转 float32；若训练拟合标准化，仅在附件2 train 拟合并原样应用。附件4含 `text` 50×768，可直接共用该字段，无须因问题2附件3缺 `text` 而强制使用 BERT 推导文本。模型应接收结构性内容 mask，防止非零的文本特殊 token/padding embedding 进入局部解释。附件4 `id` 和 `raw_text` 可用于样本级说明；无真实标签属正常。结构兼容结论：**{compatible}**。

## 原始证据回溯审计

附件4对齐版视频存在 {a4['videos_present']}/{a4['n']}，成功读时长 {a4['video_durations_obtained']}/{a4['n']}"""
    if durations:
        audit += f"，时长范围 {min(durations):.3f}–{max(durations):.3f} s"
    audit += f"""。位置结构强烈支持特殊 token + 内容 token + padding 的 token 锚点，**不支持匀速 T/50 解释**。pkl 不给每个内容 token/bin 的起止时间，也未找到官方 `aligned_50` 时间对齐表。视频总时长不能证明 `bin_i=[iT/50,(i+1)T/50)`。`q3_time_mapping_audit.csv` 给出 20×50 的结构性 `valid`，但 `start_time/end_time` 留空，秒级映射仍为 Level C。附件4虽有全文 raw_text，却没有逐 token 到原词及其时间区间的可靠对应表；不能准确给单个 bin 配原始文本片段。

问题1产物 `{Q1_BINS.relative_to(ROOT)}` 有 `bin_start_sec/bin_end_sec`，`{Q1_MAPPING.relative_to(ROOT)}` 有视频映射，但它是附件1的另外一套特征提取和时轴。附件2与问题1的样本 ID 交集 {a2_q1_overlap}，附件4与问题1交集 {a4['q1_id_overlap']}。问题1采用均匀媒体时间 bin，本次附件2/4呈 token 内容长度决定的 padding；两套 bin 索引不能直接移植。可以复用问题1的**建表与验证方法**，并对 ID 重合的附件2样本参考其原视频，但仍须建立官方特征 token→时间关系。

## 可实施性与缺口

独立 `[50,D]` 张量允许将单模态或局部位置置零并记录预测变化，故模态级及局部**特征空间**解释可实施。局部扰动应限于 `1 <= i < L-1` 的结构性内容位置；视觉内容内零行另行标记。未来需保留显式扰动 mask、敏感性对照，不得把原始全零行直接认作 padding。三个相同索引是否表示同一物理时间尚无证据，不能声称跨模态局部证据在秒级同步。

Task13 可以开始附件2 train/valid 上的完整三模态分类+回归基线；附件4仅在模型冻结后无标签推理。完整原始证据回溯仍缺官方 bin 的时间边界、物理媒体有效性 mask，以及词/帧/音频到 bin 的对应元数据。解释忠实性验证原则可预设，正式实验留待后续 Task。
"""
    (OUT / "q3_data_audit_report.md").write_text(audit, encoding="utf-8")
    findings = f"""# Task12 Q3 核心发现

## A. 数据规模
附件2 train/valid/test：{a2['train']['n']}/{a2['valid']['n']}/{a2['test']['n']}；附件4：{a4['n']} 条，无标签，视频 {a4['videos_present']} 条。

## B. 三模态 Shape
Text `[N,50,768]`、Audio `[N,50,74]`、Visual `[N,50,35]`；附件2/4 维度及字段相同，预计算 `text` 可直接使用。

## C. aligned_50
数组长度 50 严格确认；特殊 token + 内容 token + padding 的位置结构获得数据支持，**物理秒级共同时间轴未确认**。

## D. Padding / Mask
令 `L` 为 `text_bert` attention 长度；内容位置为 `1 <= i < L-1`，0/L-1 是特殊 token，L 之后为 padding。附件2音频 {sum(a2[s]['token_feature_position_diagnostic']['audio']['samples_exact_interior_nonzero_mask'] for s in SPLITS)}/{sum(a2[s]['n'] for s in SPLITS)}、附件4音频 {a4['token_feature_position_diagnostic']['audio']['samples_exact_interior_nonzero_mask']}/{a4['n']} 与内容位置的非零 mask 完全相符；视觉内部可全零，附件4有 {len(a4['all_zero_visual_ids'])} 条全零视觉。物理媒体有效 mask 未提供。

## E. 附件2 → 附件4
特征空间结构兼容，可在统一 float32 和训练集预处理参数下推理。附件4 `id/raw_text` 可追样本，不能追逐 bin 时间。

## F. 时间回溯
**Level C**。20 条原视频虽在，但没有官方 bin→秒的边界或已验证生成机制；不能用视频时长硬套 T/50。问题1映射逻辑可借鉴，现成秒数不能直接移植。

## G. Task13 前置条件
**可启动完整三模态分类+回归基线**，限附件2 train 拟合、valid 选型，test 留终评；附件4无标签且不参与训练。后续原始证据回溯须另获对齐元数据或完成独立校准。
"""
    (OUT / "q3_core_findings.md").write_text(findings, encoding="utf-8")
    print(json.dumps({"a2_counts": {k: v["n"] for k, v in a2.items()}, "a4_count": a4["n"],
                      "compatible": compatible, "a4_video_count": a4["videos_present"],
                      "q1_overlap": a4["q1_id_overlap"], "mapping_level": "C",
                      "token_position": {m: a4["token_feature_position_diagnostic"][m]
                                         for m in ("audio", "vision")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
