# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task15: frozen-model, structural-position perturbation explanations."""

from __future__ import annotations

import csv
import itertools
import json
import math
import pickle
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

from src.models.q3_gmf import MODS, OUT as MODEL_OUT, PROTOCOL, Q3Model, pooled_features


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/local"
OUT = TASK / "outputs"
FIG = TASK / "figures"
PRIOR = ROOT / "results/q3/contribution/outputs"
NAMES = ("Text", "Audio", "Visual")
CLASSES = ("Negative", "Neutral", "Positive")
SEED = 2026
EPS = 1e-12


def read_prior(split, ids):
    with (PRIOR / f"q3_modality_contributions_{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["sample_id"] for r in rows] != ids:
        raise ValueError(f"Task14 {split} sample IDs/order do not match")
    return rows, np.array([[float(r[f"{m}_joint_contribution"]) for m in ("text", "audio", "visual")]
                           for r in rows], dtype=np.float64)


def predict(model, x, batch_size=1024):
    logits, scores = [], []
    with torch.no_grad():
        for start in range(0, len(x["text"]), batch_size):
            stop = start + batch_size
            a, b = model({m: torch.from_numpy(np.ascontiguousarray(x[m][start:stop])) for m in MODS})
            logits.append(a.numpy())
            scores.append(b.numpy())
    return np.concatenate(logits), np.clip(np.concatenate(scores), -3, 3)


def probabilities(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def local_perturbations(model, raw, data, stats, full_logits, full_scores, baseline="zero", subset=None):
    """A single standardized position replacement changes pooled x by (baseline-z_t)/L."""
    n = len(data["ids"])
    mask = data["mask"]
    selected = np.arange(n) if subset is None else np.asarray(subset)
    pred = full_logits.argmax(axis=1)
    full_prob = probabilities(full_logits)[np.arange(n), pred]
    cls = np.zeros((n, 3, 50), dtype=np.float32)
    reg = np.zeros_like(cls)
    flips = np.zeros((n, 3, 50), dtype=bool)
    for j, mod in enumerate(MODS):
        print(f"  {baseline}: {mod}, {len(selected)} samples", flush=True)
        # Keep only one raw modality sequence at a time.
        sequence = np.asarray(raw[mod])
        for begin in range(0, len(selected), 32):
            sample_ids = selected[begin:begin + 32]
            valid_i, pos = np.where(mask[sample_ids])
            real_i = sample_ids[valid_i]
            length = mask[real_i].sum(axis=1).astype(np.float32)
            z = (sequence[real_i, pos].astype(np.float32) - stats[f"{mod}_mean"]) / stats[f"{mod}_std"]
            if baseline == "zero":
                replacement = np.zeros_like(z)
            else:
                # Mean over this sample's structural content positions, in standardized space.
                replacement = data["x"][mod][real_i]
            changed = {m: np.array(data["x"][m][real_i], copy=True) for m in MODS}
            changed[mod] += (replacement - z) / length[:, None]
            altered_logits, altered_scores = predict(model, changed)
            altered_prob = probabilities(altered_logits)[np.arange(len(real_i)), pred[real_i]]
            cls[real_i, j, pos] = full_prob[real_i] - altered_prob
            reg[real_i, j, pos] = full_scores[real_i] - altered_scores
            flips[real_i, j, pos] = altered_logits.argmax(axis=1) != pred[real_i]
    if not np.isfinite(cls).all() or not np.isfinite(reg).all():
        raise ValueError("Nonfinite local perturbation result")
    return cls, reg, flips


def normalize_local(cls, reg, mask):
    n = len(mask)
    cls_norm = np.zeros_like(cls)
    reg_norm = np.zeros_like(reg)
    cls_fallback = np.zeros((n, 3), dtype=bool)
    cls_uniform = np.zeros((n, 3), dtype=bool)
    reg_uniform = np.zeros((n, 3), dtype=bool)
    for j in range(3):
        positive = np.maximum(cls[:, j], 0)
        positive *= mask
        fallback = positive.sum(axis=1) <= EPS
        cls_fallback[:, j] = fallback
        positive[fallback] = np.abs(cls[fallback, j]) * mask[fallback]
        totals = positive.sum(axis=1)
        uniform = totals <= EPS
        cls_uniform[:, j] = uniform
        np.divide(positive, totals[:, None], out=cls_norm[:, j], where=totals[:, None] > EPS)
        cls_norm[uniform, j] = mask[uniform] / mask[uniform].sum(axis=1, keepdims=True)
        magnitude = np.abs(reg[:, j]) * mask
        totals = magnitude.sum(axis=1)
        uniform = totals <= EPS
        reg_uniform[:, j] = uniform
        np.divide(magnitude, totals[:, None], out=reg_norm[:, j], where=totals[:, None] > EPS)
        reg_norm[uniform, j] = mask[uniform] / mask[uniform].sum(axis=1, keepdims=True)
    within = 0.5 * (cls_norm + reg_norm)
    within *= mask[:, None, :]
    if not np.allclose(within.sum(axis=2), 1, atol=2e-6):
        raise ValueError("Within-modality importance fails normalization")
    return within, cls_norm, reg_norm, cls_fallback, cls_uniform, reg_uniform


def smoothed(values, mask, width):
    if width == 1:
        return values.copy()
    result = np.zeros_like(values)
    for t in range(50):
        result[:, :, t] = (values[:, :, max(0, t - 1):min(50, t + 2)] *
                            mask[:, None, max(0, t - 1):min(50, t + 2)]).sum(axis=2) / \
                           mask[:, max(0, t - 1):min(50, t + 2)].sum(axis=1)[:, None].clip(min=1)
    result *= mask[:, None, :]
    result /= result.sum(axis=2, keepdims=True)
    return result


def select_positions(values, mask, threshold=0.6, top_fraction=None, coverage_values=None):
    selected = np.zeros_like(values, dtype=bool)
    coverage = values if coverage_values is None else coverage_values
    for i in range(len(mask)):
        valid = np.flatnonzero(mask[i])
        for j in range(3):
            order = valid[np.argsort(-values[i, j, valid], kind="stable")]
            if top_fraction is None:
                k = min(len(order), int(np.searchsorted(np.cumsum(coverage[i, j, order]),
                                                        threshold - 1e-7, side="left") + 1))
            else:
                k = max(1, math.ceil(len(order) * top_fraction))
            selected[i, j, order[:k]] = True
    return selected


def intervals(selected_row):
    positions = np.flatnonzero(selected_row)
    if not len(positions):
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(positions)]
    return [(int(positions[a]), int(positions[b - 1] + 1)) for a, b in zip(starts, ends)]


def rule_diagnostics(within, mask):
    result = {}
    for width in (1, 3):
        signal = smoothed(within, mask, width)
        for label, threshold, fraction in (("mass_0.5", 0.5, None), ("mass_0.6", 0.6, None),
                                           ("mass_0.7", 0.7, None), ("top_20pct", None, 0.2)):
            chosen = select_positions(signal, mask, threshold, fraction, coverage_values=within)
            counts = chosen.sum(axis=2)
            interval_count = np.array([[len(intervals(chosen[i, j])) for j in range(3)]
                                       for i in range(len(mask))])
            result[f"width_{width}_{label}"] = {
                "mean_key_positions": float(counts.mean()),
                "mean_key_intervals": float(interval_count.mean()),
                "mean_raw_importance_covered": float((within * chosen).sum(axis=2).mean()),
            }
    return result


def distribution_metrics(within, global_values, selected, mask):
    n = len(mask)
    top1 = within.max(axis=2)
    top3 = np.sort(within, axis=2)[:, :, -3:].sum(axis=2)
    entropy = -(within * np.log(within + EPS)).sum(axis=2)
    entropy /= np.log(mask.sum(axis=1).clip(min=2))[:, None]
    key_count = selected.sum(axis=2)
    interval_count = np.zeros((n, 3), dtype=int)
    interval_length = np.zeros((n, 3), dtype=float)
    single_count = np.zeros((n, 3), dtype=int)
    for i in range(n):
        for j in range(3):
            spans = intervals(selected[i, j])
            interval_count[i, j] = len(spans)
            interval_length[i, j] = np.mean([b - a for a, b in spans])
            single_count[i, j] = sum(b - a == 1 for a, b in spans)
    peak_flat = global_values.reshape(n, -1).argmax(axis=1)
    return {"top1": top1, "top3": top3, "entropy": entropy, "key_count": key_count,
            "interval_count": interval_count, "interval_length": interval_length,
            "single_count": single_count, "global_peak_mod": peak_flat // 50,
            "global_peak_bin": peak_flat % 50,
            "global_peak": global_values.reshape(n, -1).max(axis=1)}


def mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


def group_summary(metrics, prior_rows, indices):
    ix = np.asarray(indices)
    if len(ix) == 0:
        return {"count": 0}
    out = {"count": len(ix), "mean_global_peak_importance": mean_or_none(metrics["global_peak"][ix]),
           "global_peak_modality_distribution": {name: float(np.mean(metrics["global_peak_mod"][ix] == j))
                                                   for j, name in enumerate(NAMES)}}
    for j, name in enumerate(NAMES):
        out[name] = {"mean_top1_share": mean_or_none(metrics["top1"][ix, j]),
                     "mean_top3_share": mean_or_none(metrics["top3"][ix, j]),
                     "mean_normalized_entropy": mean_or_none(metrics["entropy"][ix, j]),
                     "mean_key_position_count": mean_or_none(metrics["key_count"][ix, j]),
                     "mean_key_interval_count": mean_or_none(metrics["interval_count"][ix, j]),
                     "mean_key_interval_length": mean_or_none(metrics["interval_length"][ix, j]),
                     "single_bin_interval_count": int(metrics["single_count"][ix, j].sum()),
                     "multi_bin_interval_count": int((metrics["interval_count"][ix, j] -
                                                      metrics["single_count"][ix, j]).sum())}
    return out


def summarize(split, data, prior_rows, contribution, cls, reg, flips, within, selected, metrics,
              cls_fallback, cls_uniform, reg_uniform):
    n = len(data["ids"])
    valid = data["mask"][:, None, :]
    global_values = contribution[:, :, None] * within
    mass_error = np.abs(global_values.sum(axis=2) - contribution)
    total_error = np.abs(global_values.sum(axis=(1, 2)) - 1)
    if mass_error.max() > 3e-6 or total_error.max() > 3e-6:
        raise ValueError("Task14 contribution and local importance are inconsistent")
    correct = np.array([r["classification_correct"].lower() == "true" for r in prior_rows])
    visual_zero = data["visual_all_zero"]
    peak_agree = np.array([NAMES[x] == r["dominant_modality"] for x, r in
                           zip(metrics["global_peak_mod"], prior_rows)])
    low_margin = np.array([float(r["dominance_margin"]) <= 0.1 for r in prior_rows])
    meaningful_cls = (np.abs(cls) >= 0.05) & valid
    return {
        "count": n,
        "overall": group_summary(metrics, prior_rows, np.arange(n)),
        "by_true_class": {name: group_summary(metrics, prior_rows, np.flatnonzero(data["class"] == j))
                          for j, name in enumerate(CLASSES)},
        "by_pred_class": {name: group_summary(metrics, prior_rows, np.flatnonzero(
            np.array([r["pred_polarity"] == name for r in prior_rows]))) for name in CLASSES},
        "correct_vs_wrong": {"correct": group_summary(metrics, prior_rows, np.flatnonzero(correct)),
                             "wrong": group_summary(metrics, prior_rows, np.flatnonzero(~correct))},
        "visual_zero_analysis": {"visual_nonzero": group_summary(metrics, prior_rows, np.flatnonzero(~visual_zero)),
                                 "visual_all_zero": group_summary(metrics, prior_rows, np.flatnonzero(visual_zero))},
        "dominant_modality_vs_global_peak_agreement_rate": float(peak_agree.mean()),
        "dominant_modality_vs_global_peak_disagreement_count": int((~peak_agree).sum()),
        "low_modality_margin_count": int(low_margin.sum()),
        "peak_agreement_excluding_low_margin": float(peak_agree[~low_margin].mean()),
        "local_class_flip_rate_by_modality": {name: float(flips[:, j][data["mask"]].mean())
                                              for j, name in enumerate(NAMES)},
        "abs_cls_delta_ge_0_05_position_rate": {name: float(meaningful_cls[:, j][data["mask"]].mean())
                                                 for j, name in enumerate(NAMES)},
        "abs_cls_delta_ge_0_05_sample_rate": {name: float(meaningful_cls[:, j].any(axis=1).mean())
                                               for j, name in enumerate(NAMES)},
        "mean_positive_cls_delta_by_modality": {name: float(np.maximum(cls[:, j], 0)[data["mask"]].mean())
                                                  for j, name in enumerate(NAMES)},
        "fallback_counts_by_modality": {name: {"local_cls_abs_fallback": int(cls_fallback[:, j].sum()),
                                                "cls_uniform": int(cls_uniform[:, j].sum()),
                                                "reg_uniform": int(reg_uniform[:, j].sum())}
                                        for j, name in enumerate(NAMES)},
        "max_abs_modality_mass_error": float(mass_error.max()),
        "max_abs_total_mass_error": float(total_error.max()),
    }


def write_csv(path, fields, row_iter):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row_iter)


def long_rows(data, prior_rows, cls, reg, flips, cls_norm, reg_norm, within, global_values,
              selected, cls_fallback):
    for i, prior in enumerate(prior_rows):
        for j, name in enumerate(NAMES):
            for t in range(50):
                yield {"sample_id": data["ids"][i], "modality": name, "bin_index": t,
                       "structural_valid": bool(data["mask"][i, t]),
                       "cls_delta": float(cls[i, j, t]), "local_class_flip": bool(flips[i, j, t]),
                       "reg_signed_delta": float(reg[i, j, t]), "reg_magnitude": float(abs(reg[i, j, t])),
                       "cls_normalized": float(cls_norm[i, j, t]), "reg_normalized": float(reg_norm[i, j, t]),
                       "within_modality_joint": float(within[i, j, t]),
                       "within_modality_importance": float(within[i, j, t]),
                       "global_local_importance": float(global_values[i, j, t]),
                       "selected_key_position": bool(selected[i, j, t]),
                       "local_cls_abs_fallback": bool(cls_fallback[i, j]),
                       "true_polarity": prior["true_polarity"], "pred_polarity": prior["pred_polarity"],
                       "classification_correct": prior["classification_correct"],
                       "true_intensity": prior["true_intensity"], "pred_intensity": prior["pred_intensity"],
                       "dominant_modality": prior["dominant_modality"],
                       "dominance_margin": prior["dominance_margin"],
                       "low_modality_margin": float(prior["dominance_margin"]) <= 0.1,
                       "visual_all_zero": prior["visual_all_zero"]}


def interval_rows(data, prior_rows, within, global_values, selected):
    for i, prior in enumerate(prior_rows):
        for j, name in enumerate(NAMES):
            spans = intervals(selected[i, j])
            spans.sort(key=lambda ab: -float(global_values[i, j, ab[0]:ab[1]].sum()))
            for rank, (a, b) in enumerate(spans, 1):
                within_sum = float(within[i, j, a:b].sum())
                global_sum = float(global_values[i, j, a:b].sum())
                yield {"sample_id": data["ids"][i], "modality": name,
                       "interval_start": a, "interval_end": b, "interval_length": b - a,
                       "interval_importance": global_sum,
                       "within_modality_importance_sum": within_sum,
                       "global_importance_sum": global_sum, "rank": rank,
                       "dominant_modality": prior["dominant_modality"],
                       "visual_all_zero": prior["visual_all_zero"]}


def json_array(a):
    return json.dumps([float(v) for v in a], separators=(",", ":"))


def wide_rows(data, prior_rows, within, global_values, selected, metrics):
    for i, prior in enumerate(prior_rows):
        row = {"sample_id": data["ids"][i]}
        for j, mod in enumerate(("text", "audio", "visual")):
            row[f"{mod}_importance_50"] = json_array(within[i, j])
            row[f"{mod}_global_importance_50"] = json_array(global_values[i, j])
            row[f"{mod}_key_intervals"] = json.dumps(intervals(selected[i, j]), separators=(",", ":"))
        row.update({"global_peak_modality": NAMES[metrics["global_peak_mod"][i]],
                    "global_peak_bin": int(metrics["global_peak_bin"][i]),
                    "global_peak_importance": float(metrics["global_peak"][i]),
                    "dominant_modality": prior["dominant_modality"],
                    "dominance_margin": prior["dominance_margin"],
                    "low_modality_margin": float(prior["dominance_margin"]) <= 0.1,
                    "visual_all_zero": prior["visual_all_zero"]})
        yield row


def baseline_sensitivity(model, raw, data, stats, full_logits, full_scores, within_a):
    rng = np.random.default_rng(SEED)
    subset = np.sort(rng.choice(len(data["ids"]), size=min(100, len(data["ids"])), replace=False))
    cls_b, reg_b, _ = local_perturbations(model, raw, data, stats, full_logits, full_scores,
                                         baseline="sample_mean", subset=subset)
    within_b, *_ = normalize_local(cls_b[subset], reg_b[subset], data["mask"][subset])
    corrs = [[] for _ in MODS]
    overlaps = [[] for _ in MODS]
    for q, i in enumerate(subset):
        valid = np.flatnonzero(data["mask"][i])
        k = max(1, math.ceil(len(valid) * 0.2))
        for j in range(3):
            a = within_a[i, j, valid]
            b = within_b[q, j, valid]
            ra, rb = rankdata(a), rankdata(b)
            if np.std(ra) < EPS or np.std(rb) < EPS:
                corr = None
            else:
                corr = float(np.corrcoef(ra, rb)[0, 1])
                corrs[j].append(corr)
            sa = set(valid[np.argsort(-a, kind="stable")[:k]])
            sb = set(valid[np.argsort(-b, kind="stable")[:k]])
            overlaps[j].append(len(sa & sb) / len(sa | sb))
    return {"seed": SEED, "sample_count": len(subset), "sampling": "fixed random valid subset",
            "baseline_a": "standardized zero (train content-position mean)",
            "baseline_b": "sample structural-content mean in standardized space",
            "top_k": "ceil(20% of structural content positions)",
            "by_modality": {name: {"mean_spearman": mean_or_none(corrs[j]),
                                   "median_spearman": float(np.median(corrs[j])) if corrs[j] else None,
                                   "undefined_spearman_count": len(subset) - len(corrs[j]),
                                   "mean_top20_jaccard": mean_or_none(overlaps[j])}
                            for j, name in enumerate(NAMES)}}


def check_pooled_perturbation_equivalence(model, block, data, stats):
    """Compare the fast pooled update with explicit sequence replacement on one valid sample."""
    pos = int(np.flatnonzero(data["mask"][0])[0])
    mask = torch.from_numpy(data["mask"][:1].astype(np.float32))
    seq = {m: torch.from_numpy(np.ascontiguousarray(
        (np.asarray(block[m][:1], dtype=np.float32) - stats[f"{m}_mean"]) / stats[f"{m}_std"]))
           for m in MODS}
    pooled = {m: torch.from_numpy(np.ascontiguousarray(data["x"][m][:1])) for m in MODS}
    differences = {}
    with torch.no_grad():
        for m in MODS:
            changed_seq = {name: value.clone() for name, value in seq.items()}
            changed_seq[m][0, pos] = 0
            changed_pooled = {name: value.clone() for name, value in pooled.items()}
            changed_pooled[m][0] -= seq[m][0, pos] / float(data["mask"][0].sum())
            a = model(changed_seq, mask)
            b = model(changed_pooled)
            differences[m] = max(float(torch.max(torch.abs(x - y))) for x, y in zip(a, b))
    if max(differences.values()) > 1e-5:
        raise ValueError(f"Pooled and explicit sequence perturbations differ: {differences}")
    return differences


def choose_examples(data, prior_rows, metrics):
    desired = [("negative", "Negative", True), ("neutral", "Neutral", True),
               ("positive", "Positive", True), ("error", None, False)]
    chosen = {}
    for label, cls, correct in desired:
        candidates = [i for i, row in enumerate(prior_rows)
                      if (cls is None or row["true_polarity"] == cls)
                      and (row["classification_correct"].lower() == "true") == correct
                      and not data["visual_all_zero"][i] and data["mask"][i].sum() >= 10]
        if not candidates:
            continue
        margins = np.array([float(prior_rows[i]["dominance_margin"]) for i in candidates])
        peaks = metrics["global_peak"][candidates]
        # Typical among clear examples: favor margin and visible local peak, avoid extreme outliers.
        score = np.minimum(margins, np.quantile(margins, 0.75)) + \
                np.minimum(peaks, np.quantile(peaks, 0.75))
        chosen[label] = candidates[int(np.argmax(score))]
    return chosen


def plot_examples(data, prior_rows, global_values, selected, examples):
    FIG.mkdir(parents=True, exist_ok=True)
    colors = ("#326C73", "#D19548", "#84619B")
    for label, i in examples.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
        for j, ax in enumerate(axes):
            ax.bar(np.arange(50), global_values[i, j], color=colors[j], width=0.85)
            ax.scatter(np.flatnonzero(selected[i, j]), global_values[i, j, selected[i, j]],
                       facecolors="none", edgecolors="black", s=25, linewidths=0.8)
            ax.set_ylabel(NAMES[j])
            ax.set_xlim(-0.5, 49.5)
        axes[-1].set_xlabel("Aligned feature position / bin index (0–49)")
        fig.supylabel("Global local importance")
        fig.suptitle(f"{label.title()} example | true {prior_rows[i]['true_polarity']} / "
                     f"pred {prior_rows[i]['pred_polarity']}", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"q3_local_importance_example_{label}.png", dpi=170)
        plt.close(fig)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config_path = MODEL_OUT / "q3_model_config.json"
    model_path = MODEL_OUT / "q3_final_model.pt"
    stats_path = MODEL_OUT / "q3_preprocessing.npz"
    for path in (config_path, model_path, stats_path, PRIOR / "q3_modality_summary.json",
                 PRIOR / "q3_modality_contributions_valid.csv", PRIOR / "q3_modality_contributions_test.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["name"] != "C_gated_weighted" or config["best_epoch"] != 4:
        raise ValueError("Unexpected frozen Task13 model")
    with np.load(stats_path, allow_pickle=False) as saved:
        stats = {key: saved[key] for key in saved.files}
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = resolve_recorded_path(protocol["source"]["attachment2"])
    if not source.is_file():
        raise FileNotFoundError(source)
    model = Q3Model(config["fusion"], config["hidden"], config["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with source.open("rb") as f:
        raw = pickle.load(f)
    equivalence_data = pooled_features(raw["valid"], stats)
    equivalence = check_pooled_perturbation_equivalence(model, raw["valid"], equivalence_data, stats)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {"method": {"model": str(model_path.relative_to(ROOT)),
                          "preprocessing": str(stats_path.relative_to(ROOT)),
                          "baseline": "standardized zero = train mean",
                          "mask": "1 <= bin < sum(text_bert[1,:])-1",
                          "joint_weights": {"classification": 0.5, "regression": 0.5},
                          "zero_channel_fallback": "uniform over structural content positions",
                          "regression_output": "clipped to [-3,3] per Task13",
                          "pooled_vs_explicit_sequence_max_abs_difference_valid_sample": equivalence,
                          "selection_uses": "3-position moving average for ranking; raw importance for cumulative mass",
                          "importance_values": "unsmoothed", "gap_merge": "strict contiguous"},
               "threshold": 0.6, "smoothing": "3-position moving average for selection only"}
    for split in ("valid", "test"):
        print(f"Processing {split}", flush=True)
        block = raw[split]
        data = pooled_features(block, stats)
        prior_rows, contribution = read_prior(split, data["ids"])
        logits, scores = predict(model, data["x"])
        pred = logits.argmax(axis=1)
        prior_pred = np.array([CLASSES.index(r["pred_polarity"]) for r in prior_rows])
        prior_scores = np.array([float(r["pred_intensity"]) for r in prior_rows])
        if not np.array_equal(pred, prior_pred) or np.max(np.abs(scores - prior_scores)) > 1e-5:
            raise ValueError(f"Frozen {split} predictions differ from Task14")
        cls, reg, flips = local_perturbations(model, block, data, stats, logits, scores)
        within, cls_norm, reg_norm, cls_fallback, cls_uniform, reg_uniform = normalize_local(cls, reg, data["mask"])
        if split == "valid":
            summary["valid_rule_comparison"] = rule_diagnostics(within, data["mask"])
            # The Task's preferred rule is locked here, before any test explanation is evaluated.
            rules = summary["valid_rule_comparison"]
            if (rules["width_3_mass_0.6"]["mean_raw_importance_covered"] < 0.6 or
                    rules["width_3_mass_0.6"]["mean_key_intervals"] >
                    rules["width_1_mass_0.6"]["mean_key_intervals"]):
                raise ValueError("Preferred smoothed 0.6 rule fails valid acceptance check")
            summary["rule_lock_reason"] = (
                "Valid: width-3 mass-0.6 covers at least 0.6 mean raw importance and reduces "
                "interval fragmentation versus raw; Task15 preferred 0.6, strict contiguous. "
                "Locked before test.")
            summary["baseline_sensitivity"] = baseline_sensitivity(model, block, data, stats, logits, scores, within)
        selection_signal = smoothed(within, data["mask"], 3)
        selected = select_positions(selection_signal, data["mask"], 0.6, coverage_values=within)
        selected_mass = (selected * within).sum(axis=2)
        if np.min(selected_mass) < 0.6 - 2e-6:
            raise ValueError("Key positions fail raw 0.6 importance coverage")
        global_values = contribution[:, :, None] * within
        metrics = distribution_metrics(within, global_values, selected, data["mask"])
        summary[split] = summarize(split, data, prior_rows, contribution, cls, reg, flips, within,
                                   selected, metrics, cls_fallback, cls_uniform, reg_uniform)
        rows = long_rows(data, prior_rows, cls, reg, flips, cls_norm, reg_norm, within,
                         global_values, selected, cls_fallback)
        first = next(rows)
        write_csv(OUT / f"q3_local_importance_{split}.csv", list(first), itertools.chain([first], rows))
        rows = interval_rows(data, prior_rows, within, global_values, selected)
        first = next(rows)
        write_csv(OUT / f"q3_key_intervals_{split}.csv", list(first), itertools.chain([first], rows))
        if split == "test":
            rows = wide_rows(data, prior_rows, within, global_values, selected, metrics)
            first = next(rows)
            write_csv(OUT / "q3_local_explanation_samples_test.csv", list(first), itertools.chain([first], rows))
            examples = choose_examples(data, prior_rows, metrics)
            summary["example_samples"] = {label: {"sample_id": data["ids"][i],
                                                   "true_polarity": prior_rows[i]["true_polarity"],
                                                   "pred_polarity": prior_rows[i]["pred_polarity"]}
                                          for label, i in examples.items()}
            plot_examples(data, prior_rows, global_values, selected, examples)
        print(f"{split}: n={len(data['ids'])}, mass error={summary[split]['max_abs_modality_mass_error']:.2g}",
              flush=True)
    summary["by_true_class"] = {split: summary[split]["by_true_class"] for split in ("valid", "test")}
    summary["correct_vs_wrong"] = {split: summary[split]["correct_vs_wrong"] for split in ("valid", "test")}
    summary["visual_zero_analysis"] = {split: summary[split]["visual_zero_analysis"] for split in ("valid", "test")}
    summary["concentration_metrics"] = {split: summary[split]["overall"] for split in ("valid", "test")}
    (OUT / "q3_local_importance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    print("Task15 numerical outputs and figures complete", flush=True)


if __name__ == "__main__":
    main()
