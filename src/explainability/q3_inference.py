# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task17: frozen Q3 inference and explanations for all Attachment 4 aligned samples."""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import torch

from src.models.q3_gmf import MODS, OUT as MODEL_OUT, PROTOCOL, Q3Model, structural_mask
from src.explainability.q3_modality import CLASSES, NAMES, normalize_contributions, softmax
from src.explainability.q3_local import (
    intervals, local_perturbations, normalize_local, select_positions, smoothed,
)


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/inference"
OUT = TASK / "outputs"
FIG = TASK / "figures"
TEST14 = ROOT / "results/q3/contribution/outputs/q3_modality_contributions_test.csv"
TEST15 = ROOT / "results/q3/local/outputs/q3_local_explanation_samples_test.csv"


def load_inputs():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    folder = resolve_recorded_path(protocol["source"]["attachment4_directory"])
    paths = sorted(folder.glob("*.pkl"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 aligned Attachment 4 pkl files, got {len(paths)}")
    items = []
    for path in paths:
        with path.open("rb") as stream:
            item = pickle.load(stream)
        if set(("id", "raw_text", "text", "audio", "vision", "text_bert")) - set(item):
            raise ValueError(f"Missing input fields: {path}")
        if str(item["id"]) != path.stem:
            raise ValueError(f"ID/filename mismatch: {path}")
        for mod, dim in (("text", 768), ("audio", 74), ("vision", 35)):
            item[mod] = np.asarray(item[mod], dtype=np.float32)
            if item[mod].shape != (50, dim) or not np.isfinite(item[mod]).all():
                raise ValueError(f"Invalid {mod}: {path}")
        if np.asarray(item["text_bert"]).shape != (3, 50):
            raise ValueError(f"Invalid text_bert: {path}")
        items.append(item)
    ids = [str(item["id"]) for item in items]
    if len(set(ids)) != 20:
        raise ValueError("Duplicate Attachment 4 IDs")
    block = {mod: np.stack([item[mod] for item in items]) for mod in MODS}
    block["text_bert"] = np.stack([item["text_bert"] for item in items])
    block["id"] = ids
    block["raw_text"] = [str(item["raw_text"]) for item in items]
    return block


def preprocess(block, stats):
    mask = structural_mask(block)
    count = mask.sum(axis=1).astype(np.float32)[:, None]
    x = {}
    for mod in MODS:
        pooled = (block[mod] * mask[:, :, None]).sum(axis=1) / count
        x[mod] = ((pooled - stats[f"{mod}_mean"]) / stats[f"{mod}_std"]).astype(np.float32)
        if not np.isfinite(x[mod]).all():
            raise ValueError(f"Nonfinite preprocessed {mod}")
    return {"x": x, "mask": mask, "ids": block["id"], "raw_text": block["raw_text"],
            "visual_all_zero": np.all(block["vision"] == 0, axis=(1, 2))}


def frozen_predictions(model, data):
    with torch.no_grad():
        logits_tensor, raw_tensor = model({mod: torch.from_numpy(data["x"][mod]) for mod in MODS})
    logits = logits_tensor.numpy()
    raw_score = raw_tensor.numpy()
    score = np.clip(raw_score, -3, 3)
    probabilities = softmax(logits)
    pred = logits.argmax(axis=1)
    full_prob = probabilities[np.arange(len(pred)), pred]
    cls_delta, reg_delta = [], []
    for mod in MODS:
        changed = {name: torch.from_numpy(data["x"][name]) for name in MODS}
        changed[mod] = torch.zeros_like(changed[mod])
        with torch.no_grad():
            altered_logits_tensor, altered_score_tensor = model(changed)
        altered_logits = altered_logits_tensor.numpy()
        altered_score = np.clip(altered_score_tensor.numpy(), -3, 3)
        altered_probability = softmax(altered_logits)[np.arange(len(pred)), pred]
        cls_delta.append(full_prob - altered_probability)
        reg_delta.append(score - altered_score)
    cls_delta = np.stack(cls_delta, axis=1)
    reg_delta = np.stack(reg_delta, axis=1)
    cls_support = np.maximum(cls_delta, 0)
    fallback = cls_support.sum(axis=1) <= 1e-12
    cls_support[fallback] = np.abs(cls_delta[fallback])
    cls_norm, _ = normalize_contributions(cls_support)
    reg_magnitude = np.abs(reg_delta)
    reg_norm, _ = normalize_contributions(reg_magnitude)
    joint = 0.5 * (cls_norm + reg_norm)
    rank = np.argsort(joint, axis=1)
    dominant = rank[:, -1]
    margin = joint[np.arange(len(pred)), rank[:, -1]] - joint[np.arange(len(pred)), rank[:, -2]]
    return locals()


def jdump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_explanation(row):
    shares = ", ".join(f"{name} {row[name.lower() + '_contribution']:.3f}" for name in NAMES)
    if row["low_modality_margin"]:
        modality_note = "多模态贡献较接近，不宜强调单一主导模态。"
    else:
        modality_note = f"模型主要参考 {row['dominant_modality']} 模态。"
    spans = "；".join(f"{name} {row[name.lower() + '_key_intervals']}" for name in NAMES)
    warnings = []
    if row["visual_all_zero"]:
        warnings.append("原始视觉序列全零，视觉响应不能视为真实视觉内容证据。")
    else:
        warnings.append("视觉回归局部解释对基线较敏感。")
    if row["neutral_prediction_flag"]:
        warnings.append("Neutral 是模型在有标签测试中的弱项。")
    if row["extreme_intensity_flag"]:
        warnings.append("该预测处于模型已知的极端强度薄弱区域。")
    return (f"模型预测{row['pred_polarity']}，强度 {row['pred_intensity']:.3f}，类别概率 "
            f"{row['pred_confidence']:.3f}。模态贡献：{shares}。{modality_note}"
            f"关键区间（零基、左闭右开 aligned feature bins）：{spans}。"
            "关键位置在有标签测试集上经过等量扰动的模型决策忠实性验证；不对应精确词、帧或秒，也不是因果解释。"
            + "".join(warnings))


def build_outputs(block, data, pred_result, cls, reg, within, global_values, selected):
    p = pred_result
    rows, details, records = [], [], []
    for i, sample_id in enumerate(data["ids"]):
        rank = np.unravel_index(int(np.argmax(global_values[i])), (3, 50))
        flag_list = []
        if p["margin"][i] <= 0.1:
            flag_list.append("LOW_MODALITY_MARGIN")
        if data["visual_all_zero"][i]:
            flag_list.append("VISUAL_ALL_ZERO")
        else:
            flag_list.append("VISUAL_BASELINE_SENSITIVE")
        if p["pred"][i] == 1:
            flag_list.append("NEUTRAL_PREDICTION")
        if abs(p["score"][i]) >= 2:
            flag_list.append("EXTREME_INTENSITY")
        row = {"sample_id": sample_id, "raw_text": data["raw_text"][i],
               "pred_class_id": int(p["pred"][i]), "pred_polarity": CLASSES[int(p["pred"][i])],
               "pred_confidence": float(p["full_prob"][i]),
               "prob_negative": float(p["probabilities"][i, 0]),
               "prob_neutral": float(p["probabilities"][i, 1]),
               "prob_positive": float(p["probabilities"][i, 2]),
               "pred_intensity_raw": float(p["raw_score"][i]),
               "pred_intensity": float(p["score"][i])}
        for j, name in enumerate(NAMES):
            mod = name.lower()
            row[f"{mod}_cls_delta"] = float(p["cls_delta"][i, j])
            row[f"{mod}_reg_signed_delta"] = float(p["reg_delta"][i, j])
            row[f"{mod}_reg_magnitude"] = float(p["reg_magnitude"][i, j])
            row[f"{mod}_contribution"] = float(p["joint"][i, j])
        row.update({"dominant_modality": NAMES[int(p["dominant"][i])],
                    "dominance_margin": float(p["margin"][i]),
                    "low_modality_margin": bool(p["margin"][i] <= 0.1)})
        span_map = {}
        for j, name in enumerate(NAMES):
            mod = name.lower()
            positions = np.flatnonzero(selected[i, j]).astype(int).tolist()
            spans = [list(pair) for pair in intervals(selected[i, j])]
            span_map[mod] = spans
            row[f"{mod}_importance_50"] = jdump(within[i, j].astype(float).tolist())
            row[f"{mod}_within_importance_50"] = row[f"{mod}_importance_50"]
            row[f"{mod}_global_importance_50"] = jdump(global_values[i, j].astype(float).tolist())
            row[f"{mod}_key_positions"] = jdump(positions)
            row[f"{mod}_key_intervals"] = jdump(spans)
            interval_map = {t: k + 1 for k, (start, end) in enumerate(spans) for t in range(start, end)}
            for t in range(50):
                details.append({"sample_id": sample_id, "modality": name, "bin_index": t,
                                "structural_valid": bool(data["mask"][i, t]),
                                "cls_delta": float(cls[i, j, t]),
                                "reg_signed_delta": float(reg[i, j, t]),
                                "reg_magnitude": float(abs(reg[i, j, t])),
                                "within_modality_importance": float(within[i, j, t]),
                                "global_local_importance": float(global_values[i, j, t]),
                                "selected_key_position": bool(selected[i, j, t]),
                                "interval_id": interval_map.get(t, ""),
                                "dominant_modality": NAMES[int(p["dominant"][i])],
                                "visual_all_zero": bool(data["visual_all_zero"][i])})
        row.update({"global_peak_modality": NAMES[rank[0]], "global_peak_bin": rank[1],
                    "global_peak_importance": float(global_values[i][rank]),
                    "visual_all_zero": bool(data["visual_all_zero"][i]),
                    "visual_explanation_valid": not bool(data["visual_all_zero"][i]),
                    "text_faithfulness_level": "strong",
                    "audio_faithfulness_level": "supported_but_small_effect",
                    "visual_faithfulness_level": ("not_interpretable_as_visual_content" if data["visual_all_zero"][i]
                                                  else "supported_with_baseline_caution"),
                    "neutral_prediction_flag": bool(p["pred"][i] == 1),
                    "extreme_intensity_flag": bool(abs(p["score"][i]) >= 2),
                    "explanation_quality_flags": jdump(flag_list)})
        row["explanation_text"] = make_explanation(row)
        rows.append(row)
        records.append({"sample_id": sample_id,
                        "raw_text": row["raw_text"],
                        "prediction": {"class_id": row["pred_class_id"], "polarity": row["pred_polarity"],
                                       "confidence": row["pred_confidence"],
                                       "probabilities": {name.lower(): row[f"prob_{name.lower()}"] for name in CLASSES},
                                       "intensity_raw": row["pred_intensity_raw"],
                                       "intensity": row["pred_intensity"]},
                        "modality_contribution": {name.lower(): row[f"{name.lower()}_contribution"] for name in NAMES},
                        "dominant_modality": row["dominant_modality"],
                        "dominance_margin": row["dominance_margin"],
                        "local_importance": {name.lower(): within[i, j].astype(float).tolist() for j, name in enumerate(NAMES)},
                        "global_local_importance": {name.lower(): global_values[i, j].astype(float).tolist() for j, name in enumerate(NAMES)},
                        "key_intervals": span_map, "quality_flags": flag_list,
                        "explanation_text": row["explanation_text"]})
    return rows, details, records


def validate_submission(rows, expected_ids):
    ids = [r["sample_id"] for r in rows]
    numeric = np.array([[float(r["intensity"])] for r in rows])
    result = {"row_count": len(rows), "unique_id_count": len(set(ids)),
              "missing_ids": sorted(set(expected_ids) - set(ids)),
              "extra_ids": sorted(set(ids) - set(expected_ids)),
              "duplicate_ids": sorted(k for k, v in Counter(ids).items() if v > 1),
              "nan_count": int(np.isnan(numeric).sum()), "inf_count": int(np.isinf(numeric).sum()),
              "intensity_range": [float(numeric.min()), float(numeric.max())],
              "class_values": sorted({r["polarity"] for r in rows}),
              "column_names": list(rows[0]), "encoding": "UTF-8 with BOM",
              "format_basis": "Official problem requests Attachment 4 prediction and explanation CSV without fixed column template",
              "id_order_matches_input": ids == expected_ids}
    result["validation_passed"] = (len(rows) == 20 and len(set(ids)) == 20 and not result["missing_ids"]
                                   and not result["extra_ids"] and not result["duplicate_ids"]
                                   and result["nan_count"] == 0 and result["inf_count"] == 0
                                   and numeric.min() >= -3 and numeric.max() <= 3
                                   and set(result["class_values"]) <= set(CLASSES)
                                   and result["id_order_matches_input"]
                                   and all(bool(r["explanation"]) for r in rows))
    return result


def select_examples(rows):
    nonzero = [r for r in rows if not r["visual_all_zero"]]
    text = max((r for r in nonzero if r["dominant_modality"] == "Text"),
               key=lambda r: r["dominance_margin"])
    visual = max((r for r in nonzero if r["sample_id"] != text["sample_id"]),
                 key=lambda r: r["visual_contribution"])
    low = min((r for r in nonzero if r["sample_id"] not in (text["sample_id"], visual["sample_id"])),
              key=lambda r: r["dominance_margin"])
    zero = next(r for r in rows if r["visual_all_zero"])
    return {"text_dominant": text["sample_id"], "visual_relative_high": visual["sample_id"],
            "closest_modalities": low["sample_id"], "visual_all_zero": zero["sample_id"]}


def write_example_cards(rows, examples):
    by_id = {row["sample_id"]: row for row in rows}
    labels = {"text_dominant": "Text 高贡献",
              "visual_relative_high": "Visual 相对较高",
              "closest_modalities": "多模态贡献接近",
              "visual_all_zero": "视觉特征全零"}
    lines = ["# 附件4典型解释卡", "",
             "以下样本仅按解释结构选取。区间均为零基、左闭右开的 aligned feature bins；不对应词、帧或秒。", ""]
    for kind, sample_id in examples.items():
        row = by_id[sample_id]
        lines += [f"## {labels[kind]}：样本 {sample_id}", "",
                  f"- Raw Text：{row['raw_text']}",
                  f"- 预测：{row['pred_polarity']}；强度 {row['pred_intensity']:.3f}；类别概率 {row['pred_confidence']:.3f}",
                  f"- 贡献：Text {row['text_contribution']:.3f}；Audio {row['audio_contribution']:.3f}；Visual {row['visual_contribution']:.3f}",
                  f"- 数值主导模态：{row['dominant_modality']}；margin {row['dominance_margin']:.3f}",
                  f"- 关键区间：Text {row['text_key_intervals']}；Audio {row['audio_key_intervals']}；Visual {row['visual_key_intervals']}",
                  f"- 质量标记：{row['explanation_quality_flags']}",
                  "- 忠实性说明：关键位置在附件2有标签 test 上相对等量 Random / Low 扰动具有模型决策忠实性。Text 证据最强；Audio 绝对效应小；Visual 回归解释对基线敏感。", ""]
    (OUT / "q3_attachment4_example_cards.md").write_text("\n".join(lines), encoding="utf-8")


def plot_modality_contribution(rows):
    ids = [r["sample_id"] for r in rows]
    shares = np.array([[r[f"{name.lower()}_contribution"] for name in NAMES] for r in rows], dtype=float)
    colors = ("#345995", "#e7a13d", "#6caa8a")
    fig, ax = plt.subplots(figsize=(12.5, 5))
    base = np.zeros(len(rows))
    for j, name in enumerate(NAMES):
        ax.bar(ids, shares[:, j], bottom=base, label=name, color=colors[j])
        base += shares[:, j]
    ax.set(ylabel="Joint contribution", xlabel="Attachment 4 sample ID", ylim=(0, 1))
    ax.set_xlim(right=len(rows) + 1.5)
    ax.legend(ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "q3_attachment4_modality_contribution.png", dpi=170)
    plt.close(fig)


def plot_prediction_distribution(rows):
    """Show the frozen Attachment 4 polarities and per-sample intensities."""
    if [str(row["sample_id"]) for row in rows] != [f"{i:02d}" for i in range(1, 21)]:
        raise ValueError("Expected one ordered prediction for each Attachment 4 sample")
    values = np.array([float(row["pred_intensity"]) for row in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 3):
        raise ValueError("Invalid Attachment 4 prediction intensity")
    if any(row["pred_polarity"] not in CLASSES for row in rows):
        raise ValueError("Invalid Attachment 4 prediction polarity")

    counts = Counter(row["pred_polarity"] for row in rows)
    labels = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}
    colors = {"Negative": "#C55855", "Neutral": "#8393A3", "Positive": "#218C87"}
    font_path = Path("fonts/msyh.ttc")
    font_family = FontProperties(fname=str(font_path)).get_name() if font_path.exists() else "DejaVu Sans"

    with plt.rc_context({"font.family": font_family, "font.size": 12,
                         "axes.unicode_minus": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"}):
        fig = plt.figure(figsize=(14, 7.4), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 2.3], hspace=0.17)
        ax_count = fig.add_subplot(grid[0])
        ax_value = fig.add_subplot(grid[1])

        bars = ax_count.bar(range(3), [counts[name] for name in CLASSES], width=0.48,
                            color=[colors[name] for name in CLASSES], edgecolor="none")
        ax_count.set_xticks(range(3), [f"{labels[name]} ({name})" for name in CLASSES])
        ax_count.set_ylim(0, 10)
        ax_count.set_yticks([0, 2, 4, 6, 8, 10])
        ax_count.set_ylabel("样本数", fontsize=14)
        ax_count.set_title("预测类别分布", loc="left", fontsize=12, fontweight="bold", pad=18)
        ax_count.grid(axis="y", color="#E7EBEF", linewidth=0.8)
        ax_count.set_axisbelow(True)
        ax_count.bar_label(bars, labels=[f"{counts[name]}  ({counts[name] / 20:.1%})" for name in CLASSES],
                           padding=4, fontsize=12)

        x = np.arange(1, 21)
        for name in CLASSES:
            selected = np.array([i for i, row in enumerate(rows) if row["pred_polarity"] == name])
            ax_value.vlines(x[selected], 0, values[selected], color=colors[name], alpha=0.55, linewidth=2)
            ax_value.scatter(x[selected], values[selected], color=colors[name], s=54,
                             label=labels[name], zorder=3)
        ax_value.axhline(0, color="#6A7480", linewidth=1)
        ax_value.set_xlim(0.3, 20.7)
        ax_value.set_ylim(values.min() - 0.2, values.max() + 0.55)
        ax_value.set_xticks(x)
        ax_value.set_xticklabels([f"{i:02d}" for i in x], fontsize=12)
        ax_value.set_xlabel("附件4样本编号", fontsize=14)
        ax_value.set_ylabel("预测强度", fontsize=14)
        ax_value.set_title("逐样本预测强度（颜色表示预测类别）", loc="left",
                           fontsize=12, fontweight="bold", pad=18)
        ax_value.grid(axis="y", color="#E7EBEF", linewidth=0.8)
        ax_value.set_axisbelow(True)
        ax_value.legend(ncol=3, frameon=False, loc="upper right", fontsize=12)
        for ax in (ax_count, ax_value):
            ax.tick_params(axis="both", labelsize=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#BBC4CC")
        fig.savefig(FIG / "q3_attachment4_prediction_distribution.png", dpi=190, bbox_inches="tight")
        plt.close(fig)


def write_prediction_markdown(rows):
    """Summarize the same frozen predictions shown in the distribution figure."""
    if [str(row["sample_id"]) for row in rows] != [f"{i:02d}" for i in range(1, 21)]:
        raise ValueError("Expected 20 ordered Attachment 4 predictions")
    values = np.array([float(row["pred_intensity"]) for row in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 3):
        raise ValueError("Invalid prediction intensity")
    if any(row["pred_polarity"] not in CLASSES for row in rows):
        raise ValueError("Invalid prediction polarity")
    if not (FIG / "q3_attachment4_prediction_distribution.png").is_file():
        raise FileNotFoundError("Prediction distribution figure is missing")

    counts = Counter(row["pred_polarity"] for row in rows)
    summary = json.loads((OUT / "q3_attachment4_summary.json").read_text(encoding="utf-8"))
    if summary["sample_count"] != len(rows) or summary["prediction_distribution"] != dict(counts):
        raise ValueError("Prediction CSV disagrees with Task17 summary")
    for key, actual in (("mean", values.mean()), ("min", values.min()), ("max", values.max())):
        if not np.isclose(summary["intensity_statistics"][key], actual, atol=1e-9):
            raise ValueError(f"Intensity {key} disagrees with Task17 summary")

    labels = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}
    count_lines = "\n".join(
        f"| {labels[name]}（{name}） | {counts[name]} | {counts[name] / 20:.1%} |"
        for name in CLASSES
    )
    detail_lines = "\n".join(
        f"| {index:02d} | {row['sample_id']} | {labels[row['pred_polarity']]}（{row['pred_polarity']}） | {float(row['pred_intensity']):.6f} |"
        for index, row in enumerate(rows, 1)
    )
    content = f"""# 附件4测试集全量预测结果汇总与展示

## 推理说明

使用已冻结的 Task13 三模态多任务模型 `C_gated_weighted`，对附件4对齐版的 20 条无标签样本进行情感极性与强度推理。本表直接汇总 Task17 已保存的最终结果；预测极性与强度取自 [`outputs/q3_attachment4_predictions_and_explanations.csv`](outputs/q3_attachment4_predictions_and_explanations.csv)，没有重新训练或调整模型。强度按原任务规则裁剪至 `[-3,3]`。

## 结果概览

20 条样本均有预测结果：负向 **{counts['Negative']}** 条、中性 **{counts['Neutral']}** 条、正向 **{counts['Positive']}** 条。预测强度均值为 **{values.mean():.4f}**，范围为 **[{values.min():.4f}, {values.max():.4f}]**。

**表1　附件4预测类别汇总**

| 预测类别 | 样本数 | 占比 |
|:---|---:|---:|
{count_lines}
| 合计 | **20** | **100.0%** |

**表2　附件4全部样本的预测结果**

| 序号 | 样本编号 | 预测极性 | 预测强度 |
|---:|:---|:---|---:|
{detail_lines}

## 可视化

![附件4预测类别分布和逐样本强度](figures/q3_attachment4_prediction_distribution.png)

**图1　附件4预测结果概览。** 上图展示三类预测数量与占比，下图按样本编号展示强度，点的颜色对应预测极性。分类与强度由不同输出头给出，因此中性预测的强度不必恰好为 0。

> **解释边界：**附件4没有真实极性和强度标签；上述表格和图只描述模型输出，不能据此计算或声称附件4的预测准确率。逐样本置信度、模态贡献及解释详见 Task17 主 CSV。
"""
    (TASK / "推理.md").write_text(content, encoding="utf-8")


def make_figures(rows, global_values, selected, examples):
    plot_modality_contribution(rows)
    colors = ("#345995", "#e7a13d", "#6caa8a")
    counts = Counter(r["dominant_modality"] for r in rows)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(NAMES, [counts[name] for name in NAMES], color=colors)
    ax.bar_label(bars)
    ax.set(ylabel="Sample count", ylim=(0, 20))
    fig.tight_layout()
    fig.savefig(FIG / "q3_attachment4_dominant_modality_distribution.png", dpi=170)
    plt.close(fig)

    plot_prediction_distribution(rows)

    i = ids.index(examples["text_dominant"])
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, name in enumerate(NAMES):
        ax.plot(np.arange(50), global_values[i, j], label=name, color=colors[j], lw=1.8)
        ax.scatter(np.flatnonzero(selected[i, j]), global_values[i, j, selected[i, j]],
                   color=colors[j], s=13)
    ax.set(xlabel="Aligned feature bin index (zero based)", ylabel="Global local importance",
           title=f"Sample {ids[i]} | {rows[i]['pred_polarity']} | intensity {rows[i]['pred_intensity']:.2f}")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "q3_attachment4_example_explanation.png", dpi=170)
    plt.close(fig)


def summarize(rows, validation, examples):
    intensity = np.array([r["pred_intensity"] for r in rows])
    summary = {"sample_count": len(rows),
               "prediction_distribution": dict(Counter(r["pred_polarity"] for r in rows)),
               "intensity_statistics": {"mean": float(intensity.mean()), "std": float(intensity.std()),
                                        "min": float(intensity.min()), "max": float(intensity.max()),
                                        "median": float(np.median(intensity))},
               "mean_modality_contribution": {name: float(np.mean([r[f"{name.lower()}_contribution"] for r in rows])) for name in NAMES},
               "dominant_modality_distribution": dict(Counter(r["dominant_modality"] for r in rows)),
               "quality_flag_counts": {"low_margin": sum(r["low_modality_margin"] for r in rows),
                                       "visual_all_zero": sum(r["visual_all_zero"] for r in rows),
                                       "neutral_prediction": sum(r["neutral_prediction_flag"] for r in rows),
                                       "extreme_intensity": sum(r["extreme_intensity_flag"] for r in rows)},
               "global_peak_modality_distribution": dict(Counter(r["global_peak_modality"] for r in rows)),
               "mean_key_position_count": {name: float(np.mean([len(json.loads(r[f"{name.lower()}_key_positions"])) for r in rows])) for name in NAMES},
               "mean_key_interval_count": {name: float(np.mean([len(json.loads(r[f"{name.lower()}_key_intervals"])) for r in rows])) for name in NAMES},
               "mean_dominance_margin": float(np.mean([r["dominance_margin"] for r in rows])),
               "example_samples": examples, "submission_validation": validation}
    with TEST14.open(encoding="utf-8-sig", newline="") as stream:
        test14 = list(csv.DictReader(stream))
    with TEST15.open(encoding="utf-8-sig", newline="") as stream:
        test15 = list(csv.DictReader(stream))
    summary["attachment2_test_descriptive_comparison"] = {
        "sample_count": len(test14),
        "prediction_distribution": dict(Counter(r["pred_polarity"] for r in test14)),
        "mean_modality_contribution": {name: float(np.mean([float(r[f"{name.lower() if name != 'Visual' else 'visual'}_joint_contribution"]) for r in test14])) for name in NAMES},
        "dominant_modality_distribution": dict(Counter(r["dominant_modality"] for r in test14)),
        "mean_dominance_margin": float(np.mean([float(r["dominance_margin"]) for r in test14])),
        "mean_key_position_count": {name: float(np.mean([sum(end - start for start, end in json.loads(r[f"{name.lower()}_key_intervals"])) for r in test15])) for name in NAMES},
        "mean_key_interval_count": {name: float(np.mean([len(json.loads(r[f"{name.lower()}_key_intervals"])) for r in test15])) for name in NAMES},
    }
    return summary


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for path in (PROTOCOL, TEST14, TEST15, *(MODEL_OUT / name for name in
                  ("q3_final_model.pt", "q3_model_config.json", "q3_preprocessing.npz"))):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads((MODEL_OUT / "q3_model_config.json").read_text(encoding="utf-8"))
    if config["name"] != "C_gated_weighted" or config["best_epoch"] != 4:
        raise ValueError("Unexpected frozen model")
    with np.load(MODEL_OUT / "q3_preprocessing.npz", allow_pickle=False) as saved:
        stats = {key: saved[key] for key in saved.files}
    model = Q3Model(config["fusion"], config["hidden"], config["dropout"])
    model.load_state_dict(torch.load(MODEL_OUT / "q3_final_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    block = load_inputs()
    data = preprocess(block, stats)
    p = frozen_predictions(model, data)
    cls, reg, _ = local_perturbations(model, block, data, stats, p["logits"], p["score"])
    within, *_ = normalize_local(cls, reg, data["mask"])
    selected = select_positions(smoothed(within, data["mask"], 3), data["mask"], 0.6,
                                coverage_values=within)
    global_values = p["joint"][:, :, None] * within
    assert np.allclose(p["joint"].sum(axis=1), 1, atol=1e-6)
    assert np.allclose(within.sum(axis=2), 1, atol=2e-6)
    assert np.allclose(global_values.sum(axis=(1, 2)), 1, atol=2e-6)
    assert np.min((selected * within).sum(axis=2)) >= 0.6 - 2e-6
    assert np.isfinite(global_values).all() and np.min(global_values) >= 0
    assert np.all(within * ~data["mask"][:, None, :] == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, details, records = build_outputs(block, data, p, cls, reg, within, global_values, selected)
    if len(details) != 3000 or len(records) != 20:
        raise ValueError("Incomplete explanation output")
    write_csv(OUT / "q3_attachment4_predictions_and_explanations.csv", rows)
    write_csv(OUT / "q3_attachment4_local_importance.csv", details)
    (OUT / "q3_attachment4_explanations.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    submission = [{"sample_id": r["sample_id"], "polarity": r["pred_polarity"],
                   "intensity": r["pred_intensity"], "explanation": r["explanation_text"]} for r in rows]
    write_csv(OUT / "attachment4_predictions.csv", submission)
    validation = validate_submission(submission, data["ids"])
    if not validation["validation_passed"]:
        raise ValueError(f"Submission validation failed: {validation}")
    (OUT / "q3_attachment4_submission_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = select_examples(rows)
    write_example_cards(rows, examples)
    summary = summarize(rows, validation, examples)
    summary["numerical_checks"] = {"max_contribution_sum_error": float(np.max(np.abs(p["joint"].sum(axis=1) - 1))),
                                   "max_global_local_sum_error": float(np.max(np.abs(global_values.sum(axis=(1, 2)) - 1))),
                                   "minimum_key_raw_mass": float(np.min((selected * within).sum(axis=2))),
                                   "local_row_count": len(details)}
    (OUT / "q3_attachment4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(rows, global_values, selected, examples)
    write_prediction_markdown(rows)
    print(json.dumps({"prediction_distribution": summary["prediction_distribution"],
                      "mean_contribution": summary["mean_modality_contribution"],
                      "dominant": summary["dominant_modality_distribution"],
                      "quality_flags": summary["quality_flag_counts"],
                      "examples": examples, "validation_passed": validation["validation_passed"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    if "--modality-figure-only" in sys.argv:
        with (OUT / "q3_attachment4_predictions_and_explanations.csv").open(newline="", encoding="utf-8-sig") as fh:
            plot_modality_contribution(list(csv.DictReader(fh)))
    elif "--prediction-figure-only" in sys.argv:
        with (OUT / "q3_attachment4_predictions_and_explanations.csv").open(newline="", encoding="utf-8-sig") as fh:
            plot_prediction_distribution(list(csv.DictReader(fh)))
    elif "--prediction-markdown-only" in sys.argv:
        with (OUT / "q3_attachment4_predictions_and_explanations.csv").open(newline="", encoding="utf-8-sig") as fh:
            write_prediction_markdown(list(csv.DictReader(fh)))
    else:
        main()
