# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task14: frozen Task13 predictor, whole-modality mean-baseline perturbations."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.q3_gmf import MODS, OUT as PRIOR_OUT, PROTOCOL, Q3Model, pooled_features


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/contribution"
OUT = TASK / "outputs"
FIG = TASK / "figures"
NAMES = ("Text", "Audio", "Visual")
CLASSES = ("Negative", "Neutral", "Positive")
SEED = 2026
BOOTSTRAPS = 1000


def forward(model, data, zero_mod=None, batch_size=256):
    """Whole-sequence standardized zero equals a zero pooled input under masked mean."""
    all_logits, all_scores = [], []
    with torch.no_grad():
        for start in range(0, len(data["class"]), batch_size):
            stop = start + batch_size
            x = {m: torch.from_numpy(data["x"][m][start:stop]) for m in MODS}
            if zero_mod is not None:
                x[zero_mod] = torch.zeros_like(x[zero_mod])
            logits, scores = model(x)
            all_logits.append(logits.numpy())
            all_scores.append(scores.numpy())
    logits = np.concatenate(all_logits)
    scores = np.clip(np.concatenate(all_scores), -3, 3)
    if not np.isfinite(logits).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite frozen-model output")
    return logits, scores


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def normalize_contributions(values):
    sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, sums, out=np.full_like(values, 1 / 3), where=sums > 1e-12), int((sums <= 1e-12).sum())


def explain(model, data):
    logits, scores = forward(model, data)
    probs = softmax(logits)
    pred = logits.argmax(axis=1)
    full_prob = probs[np.arange(len(pred)), pred]
    cls_delta, reg_delta, flips = [], [], []
    for mod in MODS:
        altered_logits, altered_score = forward(model, data, mod)
        altered_prob = softmax(altered_logits)[np.arange(len(pred)), pred]
        cls_delta.append(full_prob - altered_prob)
        reg_delta.append(scores - altered_score)
        flips.append(pred != altered_logits.argmax(axis=1))
    cls_delta = np.stack(cls_delta, axis=1)
    reg_delta = np.stack(reg_delta, axis=1)
    flips = np.stack(flips, axis=1)
    cls_support = np.maximum(cls_delta, 0)
    fallback = cls_support.sum(axis=1) <= 1e-12
    cls_support[fallback] = np.abs(cls_delta[fallback])
    cls_norm, cls_uniform = normalize_contributions(cls_support)
    reg_magnitude = np.abs(reg_delta)
    reg_norm, reg_uniform = normalize_contributions(reg_magnitude)
    joint = 0.5 * (cls_norm + reg_norm)
    if not np.allclose(joint.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Joint contributions do not sum to one")
    rank = np.argsort(joint, axis=1)
    dominant = rank[:, -1]
    margin = joint[np.arange(len(pred)), rank[:, -1]] - joint[np.arange(len(pred)), rank[:, -2]]
    return {
        "logits": logits, "probabilities": probs, "scores": scores, "pred": pred, "full_prob": full_prob,
        "cls_delta": cls_delta, "reg_delta": reg_delta, "reg_magnitude": reg_magnitude,
        "cls_norm": cls_norm, "reg_norm": reg_norm, "joint": joint, "flips": flips,
        "dominant": dominant, "margin": margin,
        "classification_correct": pred == data["class"],
        "absolute_error": np.abs(scores - data["reg"]),
        "classification_abs_fallback_count": int(fallback.sum()),
        "classification_uniform_count": cls_uniform,
        "regression_uniform_count": reg_uniform,
    }


def write_rows(path, data, result):
    rows = []
    n = len(data["class"])
    for i in range(n):
        row = {
            "sample_id": data["ids"][i], "raw_text": str(data["raw_text"][i]),
            "true_polarity": CLASSES[int(data["class"][i])],
            "pred_polarity": CLASSES[int(result["pred"][i])],
            "classification_correct": bool(result["classification_correct"][i]),
            "true_intensity": float(data["reg"][i]), "pred_intensity": float(result["scores"][i]),
            "absolute_error": float(result["absolute_error"][i]),
            "full_pred_prob": float(result["full_prob"][i]),
        }
        for j, label in enumerate(("negative", "neutral", "positive")):
            row[f"full_logit_{label}"] = float(result["logits"][i, j])
            row[f"full_probability_{label}"] = float(result["probabilities"][i, j])
        for j, mod in enumerate(MODS):
            row[f"{mod}_cls_delta"] = float(result["cls_delta"][i, j])
            row[f"{mod}_reg_signed_delta"] = float(result["reg_delta"][i, j])
            row[f"{mod}_reg_magnitude"] = float(result["reg_magnitude"][i, j])
            row[f"{mod}_cls_normalized"] = float(result["cls_norm"][i, j])
            row[f"{mod}_reg_normalized"] = float(result["reg_norm"][i, j])
            row[f"{mod}_joint_contribution"] = float(result["joint"][i, j])
            row[f"{mod}_class_flip"] = bool(result["flips"][i, j])
        row["visual_cls_delta"] = row.pop("vision_cls_delta")
        row["visual_reg_signed_delta"] = row.pop("vision_reg_signed_delta")
        row["visual_reg_magnitude"] = row.pop("vision_reg_magnitude")
        row["visual_cls_normalized"] = row.pop("vision_cls_normalized")
        row["visual_reg_normalized"] = row.pop("vision_reg_normalized")
        row["visual_joint_contribution"] = row.pop("vision_joint_contribution")
        row["visual_class_flip"] = row.pop("vision_class_flip")
        row["dominant_modality"] = NAMES[int(result["dominant"][i])]
        row["dominant_contribution"] = float(result["joint"][i, result["dominant"][i]])
        row["dominance_margin"] = float(result["margin"][i])
        row["visual_all_zero"] = bool(data["visual_all_zero"][i])
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def modality_dict(array):
    return {name: float(array[j]) for j, name in enumerate(NAMES)}


def group_stats(result, select):
    count = int(select.sum())
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "mean_cls_contribution": modality_dict(result["cls_delta"][select].mean(axis=0)),
        "mean_reg_contribution": modality_dict(result["reg_magnitude"][select].mean(axis=0)),
        "mean_reg_signed_delta": modality_dict(result["reg_delta"][select].mean(axis=0)),
        "mean_joint_contribution": modality_dict(result["joint"][select].mean(axis=0)),
        "dominant_modality_distribution": modality_dict(np.bincount(result["dominant"][select], minlength=3) / count),
        "class_flip_rates": modality_dict(result["flips"][select].mean(axis=0)),
        "mean_dominance_margin": float(result["margin"][select].mean()),
        "median_dominance_margin": float(np.median(result["margin"][select])),
        "accuracy": float(result["classification_correct"][select].mean()),
        "mae": float(result["absolute_error"][select].mean()),
    }


def bootstrap_ci(result, seed):
    rng = np.random.default_rng(seed)
    n = len(result["joint"])
    ix = rng.integers(0, n, size=(BOOTSTRAPS, n))
    means = result["joint"][ix].mean(axis=1)
    return {name: [float(v) for v in np.quantile(means[:, j], [0.025, 0.975])]
            for j, name in enumerate(NAMES)}


def summarize(data, result, error_cut, seed):
    n = len(data["class"])
    all_rows = np.ones(n, dtype=bool)
    summary = group_stats(result, all_rows)
    summary["mean_cls_normalized"] = modality_dict(result["cls_norm"].mean(axis=0))
    summary["mean_reg_normalized"] = modality_dict(result["reg_norm"].mean(axis=0))
    summary["cls_reg_dominant_agreement_rate"] = float(np.mean(
        result["cls_norm"].argmax(axis=1) == result["reg_norm"].argmax(axis=1)))
    summary["dominance_margin_at_most_0_1_rate"] = float(np.mean(result["margin"] <= 0.1))
    summary["bootstrap_95pct_ci_mean_joint"] = bootstrap_ci(result, seed)
    summary["correct_vs_wrong"] = {
        "correct": group_stats(result, result["classification_correct"]),
        "wrong": group_stats(result, ~result["classification_correct"]),
    }
    summary["by_true_class"] = {name: group_stats(result, data["class"] == j)
                                for j, name in enumerate(CLASSES)}
    summary["by_pred_class"] = {name: group_stats(result, result["pred"] == j)
                                for j, name in enumerate(CLASSES)}
    summary["visual_zero_vs_nonzero"] = {
        "visual_all_zero": group_stats(result, data["visual_all_zero"]),
        "visual_nonzero": group_stats(result, ~data["visual_all_zero"]),
    }
    summary["regression_error_groups"] = {
        "low_error": group_stats(result, result["absolute_error"] <= error_cut),
        "high_error": group_stats(result, result["absolute_error"] > error_cut),
        "true_abs_intensity_at_least_2": group_stats(result, np.abs(data["reg"]) >= 2),
    }
    summary["fallback_counts"] = {key: result[key] for key in
                                  ("classification_abs_fallback_count", "classification_uniform_count",
                                   "regression_uniform_count")}
    summary["joint_sum_max_abs_error"] = float(np.max(np.abs(result["joint"].sum(axis=1) - 1)))
    summary["dominance_margin_quantiles"] = {
        str(q): float(np.quantile(result["margin"], q)) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
    }
    return summary


def plot_figures(summary):
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(3)
    colors = ("#3C6E71", "#D79857", "#835F91")
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, split in enumerate(("valid", "test")):
        means = [summary[split]["mean_joint_contribution"][name] for name in NAMES]
        ax.bar(x + (k - 0.5) * 0.34, means, width=0.32, label=split)
    ax.set(xticks=x, xticklabels=NAMES, ylabel="Mean joint contribution", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "q3_modality_contribution_overall.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, split in enumerate(("valid", "test")):
        rates = [summary[split]["dominant_modality_distribution"][name] for name in NAMES]
        ax.bar(x + (k - 0.5) * 0.34, rates, width=0.32, label=split)
    ax.set(xticks=x, xticklabels=NAMES, ylabel="Dominant modality share", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "q3_dominant_modality_distribution.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, name in enumerate(NAMES):
        vals = [summary["test"]["by_true_class"][cls]["mean_joint_contribution"][name] for cls in CLASSES]
        ax.bar(x + (j - 1) * 0.25, vals, width=0.24, label=name, color=colors[j])
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Mean joint contribution", xlabel="True class", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "q3_modality_contribution_by_class.png", dpi=170)
    plt.close(fig)


def check_prior_predictions(data, result):
    previous = PRIOR_OUT / "q3_test_predictions.csv"
    with previous.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(result["pred"]) or [r["sample_id"] for r in rows] != data["ids"]:
        raise ValueError("Task13 test sample order differs")
    previous_cls = np.array([int(r["pred_polarity"]) for r in rows])
    previous_reg = np.array([float(r["pred_intensity"]) for r in rows])
    if not np.array_equal(previous_cls, result["pred"]):
        raise ValueError("Frozen classification differs from Task13 test")
    difference = float(np.max(np.abs(previous_reg - result["scores"])))
    if difference > 1e-5:
        raise ValueError(f"Frozen regression differs from Task13 test: {difference}")
    return difference


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for name in ("q3_final_model.pt", "q3_model_config.json", "q3_preprocessing.npz"):
        if not (PRIOR_OUT / name).is_file():
            raise FileNotFoundError(PRIOR_OUT / name)
    config = json.loads((PRIOR_OUT / "q3_model_config.json").read_text(encoding="utf-8"))
    if config["name"] != "C_gated_weighted" or config["best_epoch"] != 4:
        raise ValueError("Unexpected Task13 frozen model")
    with np.load(PRIOR_OUT / "q3_preprocessing.npz", allow_pickle=False) as saved:
        stats = {name: saved[name] for name in saved.files}
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = resolve_recorded_path(protocol["source"]["attachment2"])
    if not source.is_file():
        raise FileNotFoundError(source)
    model = Q3Model(config["fusion"], config["hidden"], config["dropout"])
    model.load_state_dict(torch.load(PRIOR_OUT / "q3_final_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with source.open("rb") as fh:
        raw = pickle.load(fh)
    data = {}
    results = {}
    for split in ("valid", "test"):
        data[split] = pooled_features(raw[split], stats)
        data[split]["raw_text"] = raw[split]["raw_text"]
        results[split] = explain(model, data[split])
        print(f"{split}: n={len(data[split]['class'])}, accuracy={results[split]['classification_correct'].mean():.4f}, "
              f"MAE={results[split]['absolute_error'].mean():.4f}", flush=True)
    match_difference = check_prior_predictions(data["test"], results["test"])
    error_cut = float(np.median(results["valid"]["absolute_error"]))
    summary = {
        "method": {
            "frozen_model": str((PRIOR_OUT / "q3_final_model.pt").relative_to(ROOT)),
            "preprocessing": str((PRIOR_OUT / "q3_preprocessing.npz").relative_to(ROOT)),
            "zero_baseline": "standardized zero = train content-position feature mean",
            "regression_output": "clipped to [-3,3], as in Task13",
            "joint_weights": {"classification": 0.5, "regression": 0.5},
            "zero_channel_fallback": "uniform 1/3 when all three magnitudes are <=1e-12",
            "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed_valid": SEED,
            "bootstrap_seed_test": SEED + 1,
            "regression_error_threshold_valid_median": error_cut,
            "dominance_confident_threshold": None,
            "task13_test_max_abs_regression_difference": match_difference,
        },
        "valid": summarize(data["valid"], results["valid"], error_cut, SEED),
        "test": summarize(data["test"], results["test"], error_cut, SEED + 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("valid", "test"):
        write_rows(OUT / f"q3_modality_contributions_{split}.csv", data[split], results[split])
    (OUT / "q3_modality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_figures(summary)
    print(json.dumps({split: {key: summary[split][key] for key in
                              ("mean_joint_contribution", "dominant_modality_distribution", "class_flip_rates")}
                      for split in ("valid", "test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
