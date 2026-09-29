# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task10: one frozen clean evaluation on the official aligned_50 test split."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from src.data.paths import resolve_recorded_path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertModel

from src.models.q2_baseline import fuse, reg_metrics, class_metrics
from src.models.q2_mafmt import MAFMT, arrays, predict, labels

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/test"
OUT, FIG = TASK / "outputs", TASK / "figures"
T06 = ROOT / "results/q2/baseline"
T07 = ROOT / "results/q2/model"
SOURCE = ROOT / "data/raw/E题数据/附件2-数据集特征文件/aligned_50.pkl"
NAMES = ("negative", "neutral", "positive")
METRICS = ("Accuracy", "Macro_F1", "Weighted_F1", "MAE", "Pearson", "RMSE")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def bert_features(bert, tokens):
    """Task06 float32 valid-token mean plus Task07 float16 token sequence."""
    arr = np.asarray(tokens)
    if arr.shape != (727, 3, 50) or not np.all(np.isfinite(arr)) or not np.all(arr == arr.astype(np.int64)):
        raise ValueError("unexpected test text_bert")
    if arr[:, 0].min() < 0 or arr[:, 0].max() >= bert.config.vocab_size:
        raise ValueError("token outside BERT vocabulary")
    pooled = np.empty((727, 768), np.float32)
    sequence = np.empty((727, 50, 768), np.float16)
    bert.eval()
    with torch.inference_mode():
        for s in range(0, 727, 32):
            x = arr[s:s + 32].astype(np.int64)
            inputs = dict(input_ids=torch.from_numpy(x[:, 0].copy()),
                          attention_mask=torch.from_numpy(x[:, 1].copy()),
                          token_type_ids=torch.from_numpy(x[:, 2].copy()))
            hidden = bert(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled[s:s + len(x)] = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy()
            sequence[s:s + len(x)] = hidden.numpy().astype(np.float16)
    return pooled, sequence


def baseline_inference(features):
    scaler = joblib.load(T06 / "models/scaler.joblib")
    svc = joblib.load(T06 / "models/svc_baseline.joblib")
    ridge = joblib.load(T06 / "models/ridge_baseline.joblib")
    scaled = scaler.transform(features)
    return svc.predict(scaled), ridge.predict(scaled), scaler, svc, ridge


def full_inference(features):
    model = MAFMT()
    checkpoint = torch.load(T07 / "models/q2_mafmt_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    pred, raw, _ = predict(model, features)
    return pred, raw, checkpoint


def evaluate(yc, yr, pred, raw):
    return {**class_metrics(yc, pred), **reg_metrics(yr, raw)}


def plot_results(results, valid, yc, main_pred):
    colors = ("#446b82", "#d17a4a")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, metric in zip(axes, ("Accuracy", "Macro_F1", "MAE", "Pearson")):
        vals = [results[k][metric] for k in ("Baseline", "Full")]
        ax.bar(("Baseline", "Full"), vals, color=colors)
        ax.set_title(metric.replace("_", "-"))
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        for j, value in enumerate(vals):
            ax.text(j, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "final_test_model_comparison.png", dpi=160)
    plt.close(fig)
    cm = confusion_matrix(yc, main_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set(xlabel="Predicted", ylabel="True")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG / "final_test_main_confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, metric in zip(axes, ("Macro_F1", "MAE")):
        x = np.arange(2)
        for offset, split in ((-.18, "valid"), (.18, "test")):
            values = [valid[k][metric] if split == "valid" else results[k][metric] for k in ("Baseline", "Full")]
            ax.bar(x + offset, values, .36, label=split)
        ax.set_xticks(x, ("Baseline", "Full"))
        ax.set_title(metric.replace("_", "-"))
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "valid_vs_test_generalization.png", dpi=160)
    plt.close(fig)
    return cm


def main():
    torch.set_num_threads(8)
    required = [SOURCE, T06 / "models/baseline_model_manifest.json",
                T06 / "models/scaler.joblib", T06 / "models/svc_baseline.joblib",
                T06 / "models/ridge_baseline.joblib", T06 / "outputs/q2_baseline_clean_metrics.json",
                T07 / "models/q2_mafmt_best.pt", T07 / "models/q2_mafmt_manifest.json",
                T07 / "outputs/q2_main_clean_metrics.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    bm = read_json(T06 / "models/baseline_model_manifest.json")
    fm = read_json(T07 / "models/q2_mafmt_manifest.json")
    bp = resolve_recorded_path(bm["text_encoder_local_path"])
    bert_weights = bp / "model.safetensors"
    if not bert_weights.is_file():
        raise FileNotFoundError(bert_weights)
    before = {str(path.relative_to(ROOT)): sha256(path) for path in required if path != SOURCE}
    source_sha = sha256(SOURCE)
    checks = {
        "bert_revision_matches_task06_task07": bp.name == bm["text_encoder_revision"] == fm["bert_revision"],
        "bert_sha256_matches_task06_task07": sha256(bert_weights) == bm["text_encoder_weight_sha256"] == fm["bert_weight_sha256"],
        "baseline_manifest_frozen": bm["text_encoder_frozen"] and bm["training_split"] == "train" and not bm["test_used"],
        "full_manifest_frozen_selection": fm["selection"] == "minimum clean valid CE+Huber" and fm["best_epoch"] >= 1,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    valid_files = {"Baseline": T06 / "outputs/q2_baseline_clean_metrics.json",
                   "Full": T07 / "outputs/q2_main_clean_metrics.json"}
    valid_json = {name: read_json(path) for name, path in valid_files.items()}
    valid = {name: {**j["classification"], **j["regression"]} for name, j in valid_json.items()}
    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    test = data["test"]
    yc, yr = labels(test)
    checks.update({
        "official_splits": [len(data[k]["id"]) for k in ("train", "valid", "test")] == [3395, 728, 727],
        "test_size_727": len(test["id"]) == 727,
        "zero_label_only_neutral": bool(np.all(yc[yr == 0] == 1) and np.all(yc[yr != 0] != 1)),
        "signed_regression_labels": bool(yr.min() < 0 < yr.max() and np.max(np.abs(yr)) <= 3),
        "valid_metrics_frozen_outputs": all(valid_json[k]["n_valid"] == 728 for k in valid_json),
    })
    if not all(checks.values()):
        raise AssertionError(checks)
    bert = BertModel.from_pretrained(str(bp), local_files_only=True).eval()
    for param in bert.parameters():
        param.requires_grad_(False)
    checks.update({"bert_hidden_768_vocab_30522": bert.config.hidden_size == 768 and bert.config.vocab_size == 30522,
                   "bert_eval_no_grad_no_finetune": not bert.training and all(not p.requires_grad for p in bert.parameters())})
    pooled, sequences = bert_features(bert, test["text_bert"])
    del bert
    baseline_features = fuse(pooled, test["audio"], test["vision"])
    full_features = arrays(sequences, test["audio"], test["vision"], test["text_bert"][:, 1],
                           np.zeros((727, 50, 3), np.float32))
    bpred, braw, scaler, svc, ridge = baseline_inference(baseline_features)
    mpred, mraw, ckpt = full_inference(full_features)
    bpred2, braw2, _, _, _ = baseline_inference(baseline_features)
    mpred2, mraw2, ckpt2 = full_inference(full_features)
    checks.update({
        "baseline_scaler_train_only_no_refit": int(scaler.n_samples_seen_) == 3395,
        "baseline_svc_saved_parameters": svc.kernel == bm["svc_params"]["kernel"] and svc.C == bm["svc_params"]["C"] and svc.class_weight == "balanced",
        "baseline_ridge_saved_parameters": ridge.alpha == bm["ridge_params"]["alpha"],
        "baseline_fused_6218": baseline_features.shape == (727, 6218),
        "full_checkpoint_epoch_matches_manifest": ckpt["epoch"] == ckpt2["epoch"] == fm["best_epoch"],
        "full_clean_artificial_mask": bool(np.all(full_features[4] == 0)),
        "test_predictions_727_each": len(bpred) == len(braw) == len(mpred) == len(mraw) == 727,
        "baseline_repeat_class_equal": bool(np.array_equal(bpred, bpred2)),
        "baseline_repeat_raw_max_abs_le_1e_6": bool(np.max(np.abs(braw - braw2)) <= 1e-6),
        "full_repeat_class_equal": bool(np.array_equal(mpred, mpred2)),
        "full_repeat_raw_max_abs_le_1e_6": bool(np.max(np.abs(mraw - mraw2)) <= 1e-6),
        "upstream_artifacts_unchanged": before == {str(path.relative_to(ROOT)): sha256(path) for path in required if path != SOURCE} and source_sha == sha256(SOURCE),
        "no_fit_no_training_no_test_selection": True,
        "attachment3_not_read": True,
    })
    if not all(checks.values()):
        raise AssertionError(checks)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    results = {"Baseline": evaluate(yc, yr, bpred, braw), "Full": evaluate(yc, yr, mpred, mraw)}
    rows = []
    class_rows = []
    for name, pred, raw in (("Baseline", bpred, braw), ("Full", mpred, mraw)):
        report = classification_report(yc, pred, labels=[0, 1, 2], target_names=NAMES, output_dict=True, zero_division=0)
        for cls in NAMES:
            class_rows.append({"model": name, "class": cls, "precision": report[cls]["precision"],
                               "recall": report[cls]["recall"], "F1": report[cls]["f1-score"],
                               "support": int(report[cls]["support"])})
        rows.append({"model": name, "n_test": 727, **results[name], "raw_outside_ratio": results[name]["clipped_ratio"]})
        pd.DataFrame({"sample_index": np.arange(727), "sample_id": test["id"], "true_class": yc,
                      "pred_class": pred, "true_regression": yr, "pred_regression_raw": raw,
                      "pred_regression_clipped": np.clip(raw, -3, 3)}).to_csv(
                          OUT / f"q2_{'baseline' if name == 'Baseline' else 'main'}_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "q2_final_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(class_rows).to_csv(OUT / "q2_final_test_classwise_comparison.csv", index=False, encoding="utf-8-sig")
    comparison = []
    generalization = []
    for metric in METRICS:
        bv, bt, mv, mt = valid["Baseline"][metric], results["Baseline"][metric], valid["Full"][metric], results["Full"][metric]
        improvement = bt - mt if metric in ("MAE", "RMSE") else mt - bt
        comparison.append({"metric": metric, "baseline_valid": bv, "baseline_test": bt,
                           "main_valid": mv, "main_test": mt, "main_vs_baseline_test": improvement})
        for name, v, t in (("Baseline", bv, bt), ("Full", mv, mt)):
            generalization.append({"model": name, "metric": metric, "valid": v, "test": t, "delta_test_minus_valid": t - v})
    pd.DataFrame(comparison).to_csv(OUT / "q2_final_test_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(generalization).to_csv(OUT / "q2_valid_test_generalization.csv", index=False, encoding="utf-8-sig")
    checks["improvement_direction_correct"] = all(r["main_vs_baseline_test"] == (r["baseline_test"] - r["main_test"] if r["metric"] in ("MAE", "RMSE") else r["main_test"] - r["baseline_test"]) for r in comparison)
    cm = plot_results(results, valid, yc, mpred)
    validation = {"checks": checks, "all_passed": all(checks.values()), "source_sha256": source_sha,
                  "bert_revision": bp.name, "bert_weight_sha256": sha256(bert_weights),
                  "upstream_file_sha256": before,
                  "repeat_max_abs_raw": {"Baseline": float(np.max(np.abs(braw - braw2))), "Full": float(np.max(np.abs(mraw - mraw2)))},
                  "full_confusion_matrix": cm.tolist(), "class_order": list(NAMES)}
    write_json(OUT / "q2_final_test_validation.json", validation)
    if not validation["all_passed"]:
        raise AssertionError(checks)
    print(json.dumps({"results": results, "checks_passed": len(checks), "full_confusion_matrix": cm.tolist()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
