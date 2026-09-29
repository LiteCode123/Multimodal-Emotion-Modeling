# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task16: frozen Q3 model faithfulness of Task15's fixed key positions."""
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
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr, wilcoxon

from src.models.q3_gmf import MODS, OUT as MODEL_OUT, PROTOCOL, Q3Model, pooled_features
from src.explainability.q3_local import probabilities, predict

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/faithfulness"
OUT, FIG = TASK / "outputs", TASK / "figures"
TASK14 = ROOT / "results/q3/contribution/outputs"
TASK15 = ROOT / "results/q3/local/outputs"
NAMES = ("Text", "Audio", "Visual")
CLASSES = ("Negative", "Neutral", "Positive")
SEED, RANDOM_TRIALS, BOOTSTRAPS = 2026, 20, 1000


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def load_locked(split, data):
    ids = data["ids"]
    n = len(ids)
    prior = read_csv(TASK14 / f"q3_modality_contributions_{split}.csv")
    local = read_csv(TASK15 / f"q3_local_importance_{split}.csv")
    interval = read_csv(TASK15 / f"q3_key_intervals_{split}.csv")
    if prior.sample_id.tolist() != ids or len(local) != n * 3 * 50:
        raise ValueError(f"{split}: upstream IDs or row counts differ")
    if local.sample_id.iloc[::150].tolist() != ids or local.modality.iloc[:150:50].tolist() != list(NAMES):
        raise ValueError(f"{split}: Task15 local row order differs")
    if not np.array_equal(local.bin_index.to_numpy().reshape(n, 3, 50),
                          np.broadcast_to(np.arange(50), (n, 3, 50))):
        raise ValueError("Task15 position order differs")
    structural = local.structural_valid.to_numpy(dtype=bool).reshape(n, 3, 50)
    if not np.array_equal(structural, np.broadcast_to(data["mask"][:, None, :], structural.shape)):
        raise ValueError("Task15 structural mask differs")
    key = local.selected_key_position.to_numpy(dtype=bool).reshape(n, 3, 50)
    importance = local.within_modality_importance.to_numpy(dtype=np.float32).reshape(n, 3, 50)
    if np.any(key & ~structural) or np.any(key.sum(axis=2) == 0):
        raise ValueError("Invalid frozen key positions")
    coverage = (key * importance).sum(axis=2)
    if np.min(coverage) < 0.6 - 2e-6:
        raise ValueError("Task15 frozen key coverage below 0.6")
    id_index = {sample_id: i for i, sample_id in enumerate(ids)}
    spans = [[[] for _ in MODS] for _ in ids]
    for row in interval.itertuples(index=False):
        i, j = id_index[row.sample_id], NAMES.index(row.modality)
        a, b = int(row.interval_start), int(row.interval_end)
        if not (0 <= a < b <= 50) or not np.all(key[i, j, a:b]):
            raise ValueError("Task15 interval conflicts with key set")
        spans[i][j].append((a, b, float(row.within_modality_importance_sum)))
    for i in range(n):
        for j in range(3):
            expanded = np.zeros(50, dtype=bool)
            for a, b, _ in spans[i][j]:
                if expanded[a:b].any():
                    raise ValueError("Overlapping intervals")
                expanded[a:b] = True
            if not np.array_equal(expanded, key[i, j]):
                raise ValueError("Task15 intervals do not reconstruct key positions")
    return prior, key, importance, coverage, spans


def infer_variants(model, data, z, sample_indices, selections, modality, baseline="zero"):
    """Exact masked-mean change for simultaneous position replacements."""
    changed = {m: np.repeat(data["x"][m][sample_indices], len(selections[0]), axis=0)
               for m in MODS}
    target = changed[modality].reshape(len(sample_indices), len(selections[0]), -1)
    for q, i in enumerate(sample_indices):
        replacement = 0 if baseline == "zero" else data["x"][modality][i]
        length = float(data["mask"][i].sum())
        for v, positions in enumerate(selections[q]):
            target[q, v] += np.sum(replacement - z[i, positions], axis=0) / length
    logits, scores = predict(model, changed, batch_size=1024)
    return logits.reshape(len(sample_indices), len(selections[0]), 3), scores.reshape(len(sample_indices), -1)


def evaluate_split(split, block, stats, model):
    data = pooled_features(block, stats)
    n = len(data["ids"])
    prior, key, importance, coverage, spans = load_locked(split, data)
    full_logits, full_scores = predict(model, data["x"])
    full_pred = full_logits.argmax(axis=1)
    full_prob = probabilities(full_logits)[np.arange(n), full_pred]
    if [CLASSES[c] for c in full_pred] != prior.pred_polarity.tolist():
        raise ValueError(f"{split}: frozen class differs from Task14")
    if np.max(np.abs(full_scores - prior.pred_intensity.to_numpy(dtype=float))) > 1e-5:
        raise ValueError(f"{split}: frozen regression differs from Task14")
    rng = np.random.default_rng(SEED if split == "valid" else SEED + 1)
    rows, random_rows, interval_rows = [], [], []
    visual_choices = {}
    for j, (mod, name) in enumerate(zip(MODS, NAMES)):
        print(f"{split}: {name}, {n} samples", flush=True)
        z = ((np.asarray(block[mod], dtype=np.float32) - stats[f"{mod}_mean"])
             / stats[f"{mod}_std"])
        for start in range(0, n, 32):
            indices = np.arange(start, min(start + 32, n))
            choices = []
            for i in indices:
                valid = np.flatnonzero(data["mask"][i])
                k = int(key[i, j].sum())
                low = valid[np.argsort(importance[i, j, valid], kind="stable")[:k]]
                random = [np.sort(rng.choice(valid, size=k, replace=False)) for _ in range(RANDOM_TRIALS)]
                chosen = [np.flatnonzero(key[i, j]), low, *random]
                choices.append(chosen)
                if split == "valid" and name == "Visual":
                    visual_choices[int(i)] = chosen
            logits, scores = infer_variants(model, data, z, indices, choices, mod)
            probs = probabilities(logits.reshape(-1, 3)).reshape(len(indices), -1, 3)
            selected_prob = probs[np.arange(len(indices))[:, None],
                                  np.arange(2 + RANDOM_TRIALS)[None, :], full_pred[indices, None]]
            cls_drop = full_prob[indices, None] - selected_prob
            signed = full_scores[indices, None] - scores
            reg_change = np.abs(signed)
            flips = logits.argmax(axis=2) != full_pred[indices, None]
            for q, i in enumerate(indices):
                prior_row = prior.iloc[i]
                random_overlap = [len(set(positions) & set(choices[q][0])) / len(choices[q][0])
                                  for positions in choices[q][2:]]
                row = {
                    "sample_id": data["ids"][i], "modality": name,
                    "n_structural_positions": int(data["mask"][i].sum()),
                    "n_key_positions": len(choices[q][0]),
                    "key_importance_coverage": float(coverage[i, j]),
                    "key_low_overlap_count": len(set(choices[q][0]) & set(choices[q][1])),
                    "random_key_overlap_mean": float(np.mean(random_overlap)),
                    "full_pred_class": CLASSES[full_pred[i]], "full_pred_prob": float(full_prob[i]),
                    "full_pred_intensity": float(full_scores[i]),
                    "key_pred_class": CLASSES[int(logits[q, 0].argmax())],
                    "key_pred_prob": float(selected_prob[q, 0]),
                    "key_pred_intensity": float(scores[q, 0]),
                    "random_mean_pred_prob": float(selected_prob[q, 2:].mean()),
                    "random_std_pred_prob": float(selected_prob[q, 2:].std(ddof=1)),
                    "random_mean_pred_intensity": float(scores[q, 2:].mean()),
                    "random_std_pred_intensity": float(scores[q, 2:].std(ddof=1)),
                    "low_pred_class": CLASSES[int(logits[q, 1].argmax())],
                    "low_pred_prob": float(selected_prob[q, 1]),
                    "low_pred_intensity": float(scores[q, 1]),
                    "key_cls_drop": float(cls_drop[q, 0]),
                    "random_cls_drop": float(cls_drop[q, 2:].mean()),
                    "random_cls_drop_std": float(cls_drop[q, 2:].std(ddof=1)),
                    "low_cls_drop": float(cls_drop[q, 1]),
                    "key_reg_change": float(reg_change[q, 0]),
                    "random_reg_change": float(reg_change[q, 2:].mean()),
                    "random_reg_change_std": float(reg_change[q, 2:].std(ddof=1)),
                    "low_reg_change": float(reg_change[q, 1]),
                    "key_reg_signed": float(signed[q, 0]),
                    "random_reg_signed": float(signed[q, 2:].mean()),
                    "low_reg_signed": float(signed[q, 1]),
                    "key_class_flip": bool(flips[q, 0]),
                    "random_class_flip_rate": float(flips[q, 2:].mean()),
                    "low_class_flip": bool(flips[q, 1]),
                    "key_gt_random_cls": bool(cls_drop[q, 0] > cls_drop[q, 2:].mean()),
                    "key_gt_low_cls": bool(cls_drop[q, 0] > cls_drop[q, 1]),
                    "key_gt_random_reg": bool(reg_change[q, 0] > reg_change[q, 2:].mean()),
                    "key_gt_low_reg": bool(reg_change[q, 0] > reg_change[q, 1]),
                    "dominant_modality": prior_row.dominant_modality,
                    "dominance_margin": float(prior_row.dominance_margin),
                    "low_margin": float(prior_row.dominance_margin) <= 0.1,
                    "classification_correct": bool(prior_row.classification_correct),
                    "true_polarity": prior_row.true_polarity,
                    "pred_polarity": prior_row.pred_polarity,
                    "true_intensity": float(prior_row.true_intensity),
                    "pred_intensity": float(prior_row.pred_intensity),
                    "visual_all_zero": bool(prior_row.visual_all_zero),
                }
                rows.append(row)
                if split == "test":
                    for r, positions in enumerate(choices[q][2:]):
                        random_rows.append({"sample_id": data["ids"][i], "modality": name,
                                            "trial": r + 1,
                                            "selected_bins": json.dumps(positions.tolist(), separators=(",", ":")),
                                            "random_key_overlap": random_overlap[r],
                                            "cls_drop": float(cls_drop[q, r + 2]),
                                            "reg_change": float(reg_change[q, r + 2]),
                                            "class_flip": bool(flips[q, r + 2]),
                                            "seed": SEED + 1})
            # Interval validation uses the same frozen positions, one interval at a time.
            max_intervals = max(len(spans[i][j]) for i in indices)
            if max_intervals:
                for count in range(1, max_intervals + 1):
                    ix = np.array([i for i in indices if len(spans[i][j]) == count], dtype=int)
                    if not len(ix):
                        continue
                    selections = [[np.arange(a, b) for a, b, _ in spans[i][j]] for i in ix]
                    ilogits, iscores = infer_variants(model, data, z, ix, selections, mod)
                    iprob = probabilities(ilogits.reshape(-1, 3)).reshape(len(ix), count, 3)
                    for q, i in enumerate(ix):
                        for v, (a, b, mass) in enumerate(spans[i][j]):
                            interval_rows.append({"split": split, "sample_id": data["ids"][i],
                                                  "modality": name, "interval_start": a,
                                                  "interval_end": b, "interval_length": b - a,
                                                  "within_modality_importance_sum": mass,
                                                  "cls_drop": float(full_prob[i] - iprob[q, v, full_pred[i]]),
                                                  "reg_change": float(abs(full_scores[i] - iscores[q, v])),
                                                  "visual_all_zero": bool(data["visual_all_zero"][i])})
        del z
    samples = pd.DataFrame(rows)
    if len(samples) != n * 3 or (samples.n_key_positions > samples.n_structural_positions).any():
        raise ValueError("Incomplete or invalid sample output")
    return data, samples, pd.DataFrame(random_rows), pd.DataFrame(interval_rows), visual_choices


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    draw = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return [float(v) for v in np.quantile(values[draw].mean(axis=1), [0.025, 0.975])]


def group_stats(frame):
    if len(frame) == 0:
        return {"n_sample_modality": 0}
    result = {"n_sample_modality": int(len(frame)), "n_unique_samples": int(frame.sample_id.nunique())}
    for channel in ("cls_drop", "reg_change"):
        for kind in ("key", "random", "low"):
            result[f"mean_{kind}_{channel}"] = float(frame[f"{kind}_{channel}"].mean())
        for control in ("random", "low"):
            diff = frame[f"key_{channel}"] - frame[f"{control}_{channel}"]
            result[f"mean_key_minus_{control}_{channel}"] = float(diff.mean())
            result[f"win_rate_key_gt_{control}_{channel}"] = float((diff > 0).mean())
            result[f"tie_rate_key_eq_{control}_{channel}"] = float((diff == 0).mean())
    for kind in ("key", "random", "low"):
        result[f"{kind}_class_flip_rate"] = float(frame[f"{kind}_class_flip" if kind != "random"
                                                        else "random_class_flip_rate"].mean())
    return result


def split_summary(frame):
    primary = frame[(frame.modality != "Visual") | (~frame.visual_all_zero)]
    groups = {
        "overall_primary_descriptive": group_stats(primary),
        "by_modality": {name: group_stats(primary[primary.modality == name]) for name in NAMES},
        "by_true_class": {name: group_stats(primary[primary.true_polarity == name]) for name in CLASSES},
        "by_pred_class": {name: group_stats(primary[primary.pred_polarity == name]) for name in CLASSES},
        "correct_vs_wrong": {"correct": group_stats(primary[primary.classification_correct]),
                             "wrong": group_stats(primary[~primary.classification_correct])},
        "low_margin_vs_normal": {"low_margin": group_stats(primary[primary.low_margin]),
                                 "normal_margin": group_stats(primary[~primary.low_margin])},
        "visual_zero_analysis": {
            "visual_nonzero": group_stats(frame[(frame.modality == "Visual") & (~frame.visual_all_zero)]),
            "visual_all_zero": group_stats(frame[(frame.modality == "Visual") & frame.visual_all_zero])},
        "extreme_intensity": {
            "extreme_abs_true_ge_2": group_stats(primary[primary.true_intensity.abs() >= 2]),
            "non_extreme": group_stats(primary[primary.true_intensity.abs() < 2])},
        "text_by_true_class": {name: group_stats(frame[(frame.modality == "Text") &
                                                      (frame.true_polarity == name)]) for name in CLASSES},
        "text_correct_vs_wrong": {"correct": group_stats(frame[(frame.modality == "Text") & frame.classification_correct]),
                                  "wrong": group_stats(frame[(frame.modality == "Text") & ~frame.classification_correct])},
    }
    correlations = {}
    for name in NAMES:
        subset = primary[primary.modality == name]
        correlations[name] = {}
        for x in ("n_key_positions", "key_importance_coverage"):
            correlations[name][x] = {y: float(spearmanr(subset[x], subset[y]).statistic)
                                     for y in ("key_cls_drop", "key_reg_change")}
    groups["key_count_and_coverage_spearman"] = correlations
    text_rows = frame[frame.modality == "Text"]
    flip_tests = {}
    for control in ("random", "low"):
        a = text_rows.key_class_flip.to_numpy(dtype=float)
        b = text_rows["random_class_flip_rate" if control == "random" else "low_class_flip"].to_numpy(dtype=float)
        difference = a - b
        nonzero = difference[difference != 0]
        flip_tests[f"key_vs_{control}"] = {
            "mean_key_flip_rate": float(a.mean()), "mean_control_flip_rate": float(b.mean()),
            "mean_paired_difference": float(difference.mean()),
            "wilcoxon_two_sided_p": float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero) else 1.0,
        }
    groups["text_flip_comparisons"] = flip_tests
    return groups


def check_multi_position_equivalence(model, block, stats):
    """Check the actual frozen network against explicit simultaneous sequence replacement."""
    indices = np.arange(3)
    data = pooled_features({k: np.asarray(v)[indices] for k, v in block.items()}, stats)
    mask = data["mask"]
    seq = {m: ((np.asarray(block[m])[indices].astype(np.float32) - stats[f"{m}_mean"])
               / stats[f"{m}_std"]) for m in MODS}
    errors = {}
    with torch.no_grad():
        for mod in MODS:
            explicit = {m: value.copy() for m, value in seq.items()}
            pooled = {m: value.copy() for m, value in data["x"].items()}
            for i in indices:
                positions = np.flatnonzero(mask[i])[:3]
                explicit[mod][i, positions] = 0
                pooled[mod][i] += (-seq[mod][i, positions].sum(axis=0) / float(mask[i].sum()))
            explicit_out = model({m: torch.from_numpy(v) for m, v in explicit.items()},
                                 torch.from_numpy(mask.astype(np.float32)))
            pooled_out = model({m: torch.from_numpy(v) for m, v in pooled.items()})
            error = max(float(torch.max(torch.abs(a - b))) for a, b in zip(explicit_out, pooled_out))
            errors[mod] = error
            if error > 1e-5:
                raise ValueError(f"Multi-position pooled perturbation differs for {mod}: {error}")
    return errors


def paired_tests(split, frame):
    rng = np.random.default_rng(SEED + (0 if split == "valid" else 1) + 100)
    records = []
    for name in NAMES:
        subset = frame[(frame.modality == name) & ((~frame.visual_all_zero) if name == "Visual" else True)]
        for task_type, metric in (("classification", "cls_drop"), ("regression", "reg_change")):
            for control in ("random", "low"):
                a, b = subset[f"key_{metric}"].to_numpy(float), subset[f"{control}_{metric}"].to_numpy(float)
                diff = a - b
                nz = diff[diff != 0]
                if len(nz):
                    test = wilcoxon(nz, alternative="two-sided", method="auto")
                    ranks = rankdata(abs(nz))
                    effect = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
                    statistic, p = float(test.statistic), float(test.pvalue)
                else:
                    statistic, p, effect = 0.0, 1.0, 0.0
                gap_ci = bootstrap_mean_ci(diff, rng)
                records.append({"split": split, "modality": name, "task_type": task_type,
                                "comparison": f"key_vs_{control}", "n": len(diff),
                                "nonzero_pairs": len(nz), "mean_key": float(a.mean()),
                                "mean_control": float(b.mean()), "mean_difference": float(diff.mean()),
                                "median_difference": float(np.median(diff)),
                                "win_rate": float((diff > 0).mean()),
                                "tie_rate": float((diff == 0).mean()),
                                "wilcoxon_stat": statistic, "p_raw": p,
                                "effect_size": effect,
                                "bootstrap_ci_low": gap_ci[0],
                                "bootstrap_ci_high": gap_ci[1],
                                "mean_key_ci": bootstrap_mean_ci(a, rng),
                                "mean_control_ci": bootstrap_mean_ci(b, rng)})
    tests = pd.DataFrame(records)
    p = tests.p_raw.to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    tests["p_fdr"] = np.clip(adjusted, 0, 1)
    return tests


def interval_summary(frame):
    result = {}
    for name in NAMES:
        sub = frame[(frame.modality == name) & ((~frame.visual_all_zero) if name == "Visual" else True)]
        result[name] = {}
        for metric in ("cls_drop", "reg_change"):
            # Interval length is a potential confounder; these are auxiliary correlations only.
            overall = spearmanr(sub.within_modality_importance_sum, sub[metric])
            within = []
            for _, group in sub.groupby("sample_id"):
                if len(group) >= 2 and group.within_modality_importance_sum.nunique() > 1 and group[metric].nunique() > 1:
                    within.append(float(spearmanr(group.within_modality_importance_sum, group[metric]).statistic))
            result[name][metric] = {"n_intervals": int(len(sub)), "n_samples_multi_interval_rankable": len(within),
                                    "pooled_spearman": float(overall.statistic),
                                    "mean_within_sample_spearman": float(np.mean(within)) if within else None,
                                    "median_within_sample_spearman": float(np.median(within)) if within else None}
    return result


def visual_sensitivity(block, data, stats, model, samples, choices):
    rng = np.random.default_rng(SEED)
    ix = np.sort(rng.choice(len(data["ids"]), size=100, replace=False))
    z = ((np.asarray(block["vision"], dtype=np.float32) - stats["vision_mean"])
         / stats["vision_std"])
    logits, scores = predict(model, data["x"])
    pred = logits.argmax(axis=1)
    prob = probabilities(logits)[np.arange(len(pred)), pred]
    records = []
    for start in range(0, len(ix), 32):
        chosen_ix = ix[start:start + 32]
        chosen = [[choices[int(i)][0], *choices[int(i)][2:]] for i in chosen_ix]
        logits_b, scores_b = infer_variants(model, data, z, chosen_ix, chosen, "vision", baseline="sample_mean")
        probs_b = probabilities(logits_b.reshape(-1, 3)).reshape(len(chosen_ix), 1 + RANDOM_TRIALS, 3)
        for q, i in enumerate(chosen_ix):
            drop = prob[i] - probs_b[q, :, pred[i]]
            change = abs(scores[i] - scores_b[q])
            original = samples[(samples.sample_id == data["ids"][i]) & (samples.modality == "Visual")].iloc[0]
            records.append({"sample_id": data["ids"][i], "visual_all_zero": bool(data["visual_all_zero"][i]),
                            "zero_key_gt_random_cls": bool(original.key_gt_random_cls),
                            "sample_mean_key_gt_random_cls": bool(drop[0] > drop[1:].mean()),
                            "zero_key_gt_random_reg": bool(original.key_gt_random_reg),
                            "sample_mean_key_gt_random_reg": bool(change[0] > change[1:].mean()),
                            "sample_mean_cls_gap": float(drop[0] - drop[1:].mean()),
                            "sample_mean_reg_gap": float(change[0] - change[1:].mean())})
    result = pd.DataFrame(records)
    summary = {"seed": SEED, "sample_count": len(result), "nonzero_count": int((~result.visual_all_zero).sum()),
               "baseline": "sample structural-content modality mean in standardized space"}
    for metric in ("cls", "reg"):
        for label, part in (("all", result), ("visual_nonzero", result[~result.visual_all_zero])):
            summary[f"{metric}_{label}"] = {
                "zero_baseline_win_rate": float(part[f"zero_key_gt_random_{metric}"].mean()),
                "sample_mean_win_rate": float(part[f"sample_mean_key_gt_random_{metric}"].mean()),
                "direction_agreement_rate": float((part[f"zero_key_gt_random_{metric}"] ==
                                                   part[f"sample_mean_key_gt_random_{metric}"]).mean()),
                "sample_mean_gap": float(part[f"sample_mean_{metric}_gap"].mean())}
    return summary


def plot_figures(summary):
    FIG.mkdir(parents=True, exist_ok=True)
    test = summary["test"]["by_modality"]
    for metric, title, filename in (("cls_drop", "Original-class probability drop", "q3_faithfulness_classification.png"),
                                    ("reg_change", "Absolute intensity change", "q3_faithfulness_regression.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(3)
        for q, (kind, color) in enumerate((("key", "#285C6B"), ("random", "#C79954"), ("low", "#8D829B"))):
            ax.bar(x + (q - 1) * 0.24, [test[name][f"mean_{kind}_{metric}"] for name in NAMES],
                   0.23, label=kind.title(), color=color)
        ax.set(xticks=x, xticklabels=NAMES, ylabel=title, title="Frozen model · test · Visual nonzero")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / filename, dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, metric, title in zip(axes, ("cls_drop", "reg_change"), ("Classification", "Regression")):
        x = np.arange(3)
        for q, (control, color) in enumerate((("random", "#285C6B"), ("low", "#C79954"))):
            ax.bar(x + (q - 0.5) * 0.34,
                   [test[name][f"win_rate_key_gt_{control}_{metric}"] for name in NAMES],
                   0.32, label=f"Key > {control.title()}", color=color)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set(xticks=x, xticklabels=NAMES, ylim=(0, 1), title=title)
    axes[0].set_ylabel("Paired win rate")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "q3_faithfulness_win_rates.png", dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    required = [MODEL_OUT / x for x in ("q3_final_model.pt", "q3_model_config.json", "q3_preprocessing.npz")]
    required += [TASK14 / f"q3_modality_contributions_{s}.csv" for s in ("valid", "test")]
    required += [TASK15 / f"q3_{kind}_{s}.csv" for kind in ("local_importance", "key_intervals")
                 for s in ("valid", "test")]
    required += [TASK15 / "q3_local_explanation_samples_test.csv", TASK15 / "q3_local_importance_summary.json"]
    required += [PROTOCOL]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads((MODEL_OUT / "q3_model_config.json").read_text(encoding="utf-8"))
    if config["name"] != "C_gated_weighted" or config["best_epoch"] != 4:
        raise ValueError("Unexpected frozen Task13 model")
    with np.load(MODEL_OUT / "q3_preprocessing.npz", allow_pickle=False) as saved:
        stats = {key: saved[key] for key in saved.files}
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = resolve_recorded_path(protocol["source"]["attachment2"])
    if not source.is_file():
        raise FileNotFoundError(source)
    model = Q3Model(config["fusion"], config["hidden"], config["dropout"])
    model.load_state_dict(torch.load(MODEL_OUT / "q3_final_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with source.open("rb") as f:
        raw = pickle.load(f)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"method": {"model": str(required[0].relative_to(ROOT)),
                          "baseline": "standardized zero = train content-position mean",
                          "random_trials": RANDOM_TRIALS, "random_seed_valid": SEED,
                          "random_seed_test": SEED + 1,
                          "random_sampling": "without replacement within trial; key overlap allowed",
                          "low_selection": "lowest frozen Task15 within_modality_importance; stable bin-index ties",
                          "primary_unit": "paired sample x modality; Visual excludes all-zero in primary analysis",
                          "classification": "original predicted class probability drop; signed",
                          "regression": "absolute change of clipped [-3,3] output",
                          "test": "two-sided Wilcoxon, zero differences removed; rank-biserial effect",
                          "multiplicity": "BH-FDR over 12 modality x task x comparison tests within split",
                          "bootstrap": "1000 paired sample resamples; percentile 95% CI of mean gap",
                          "interval": "one frozen key interval at a time; auxiliary unadjusted Spearman"}}
    summary["method"]["multi_position_explicit_vs_pooled_max_abs_error_valid_3_samples"] = \
        check_multi_position_equivalence(model, raw["valid"], stats)
    data, valid, _, valid_intervals, choices = evaluate_split("valid", raw["valid"], stats, model)
    valid_tests = paired_tests("valid", valid)
    summary["valid"] = split_summary(valid)
    summary["valid"]["interval_validation"] = interval_summary(valid_intervals)
    summary["visual_baseline_sensitivity"] = visual_sensitivity(raw["valid"], data, stats, model, valid, choices)
    summary["protocol_lock"] = {"locked_after_valid_before_test": True,
                                "random_trials": RANDOM_TRIALS, "seed": SEED,
                                "primary": "position-count matched; Visual nonzero primary",
                                "statistics": "paired two-sided Wilcoxon, rank-biserial, 12-test BH-FDR, 1000 bootstrap",
                                "visual_sensitivity": "fixed 100 valid samples, sample-modality-mean baseline"}
    (OUT / "q3_faithfulness_protocol_lock.json").write_text(
        json.dumps({"method": summary["method"], "protocol_lock": summary["protocol_lock"],
                    "valid_tests": valid_tests.to_dict(orient="records")}, indent=2), encoding="utf-8")
    valid.to_csv(OUT / "q3_faithfulness_samples_valid.csv", index=False, encoding="utf-8-sig")
    print("Valid protocol locked; starting test", flush=True)
    _, test, random_trials, test_intervals, _ = evaluate_split("test", raw["test"], stats, model)
    test_tests = paired_tests("test", test)
    summary["test"] = split_summary(test)
    summary["test"]["interval_validation"] = interval_summary(test_intervals)
    summary["significance_tests"] = {"valid": valid_tests.to_dict(orient="records"),
                                     "test": test_tests.to_dict(orient="records")}
    summary["bootstrap_ci"] = {"definition": "95% percentile intervals in significance_tests for mean paired gap and means"}
    test.to_csv(OUT / "q3_faithfulness_samples_test.csv", index=False, encoding="utf-8-sig")
    random_trials.to_csv(OUT / "q3_faithfulness_random_trials_test.csv", index=False, encoding="utf-8-sig")
    pd.concat([valid_tests, test_tests]).to_csv(OUT / "q3_faithfulness_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat([valid_intervals, test_intervals]).to_csv(OUT / "q3_faithfulness_intervals.csv", index=False, encoding="utf-8-sig")
    (OUT / "q3_faithfulness_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_figures(summary)
    print(test_tests[["modality", "task_type", "comparison", "mean_difference", "win_rate", "p_fdr", "effect_size"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
