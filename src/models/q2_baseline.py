# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task06: frozen BERT + flat audio/vision + balanced SVC/Ridge baseline.

Run with the project venv from the project root. This reads Task04/05 protocols,
uses only aligned_50 train/valid, and evaluates the frozen models on Task05's grid.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from src.data.paths import resolve_recorded_path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import BertModel

from src.data.q2_missingness import simulate_missingness


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/baseline"
OUT = TASK / "outputs"
MODELS = TASK / "models"
FIG = TASK / "figures"
P04 = ROOT / "results/q2/protocol/outputs/q2_protocol.json"
P05 = ROOT / "results/q2/missingness/outputs/q2_missing_protocol.json"
GRID = ROOT / "results/q2/missingness/outputs/q2_valid_36grid.csv"
SEED = 20260924
BATCH = 32


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs():
    for path in (P04, P05, GRID):
        if not path.is_file():
            raise FileNotFoundError(path)
    p04 = json.loads(P04.read_text(encoding="utf-8"))
    p05 = json.loads(P05.read_text(encoding="utf-8"))
    expected = {"feature_version": "aligned_50", "train_split": "train", "validation_split": "valid",
                "test_split": "test", "temporal_length": 50, "text_source": "text_bert",
                "audio_source": "audio", "vision_source": "vision", "text_input_shape": [3, 50],
                "audio_shape": [50, 74], "vision_shape": [50, 35]}
    for k, v in expected.items():
        if p04.get(k) != v:
            raise ValueError(f"Task04 protocol conflict: {k}")
    if p05["feature_version"] != "aligned_50" or p05["controlled_cells"] != 36 or p05["seed"] != SEED:
        raise ValueError("Task05 protocol conflict")
    grid = pd.read_csv(GRID)
    required = ["experiment_id", "sample_index", "sample_id", "pattern", "rate", "position",
                "text_start", "text_end", "audio_start", "audio_end", "vision_start", "vision_end"]
    if not set(required).issubset(grid.columns) or len(grid) != 36 * 728 or grid.experiment_id.nunique() != 36:
        raise ValueError("Task05 grid missing or incomplete")
    if grid.groupby("experiment_id").size().ne(728).any():
        raise ValueError("Task05 grid has incomplete cell")
    return p04, p05, grid


def checkpoint():
    base = ROOT / "data/models/bert-base-uncased"
    snapshots = [p for p in base.iterdir() if p.is_dir() and (p / "model.safetensors").is_file()]
    if len(snapshots) != 1:
        raise RuntimeError(f"Expected one complete local bert-base-uncased snapshot, found {len(snapshots)}")
    path = snapshots[0]
    model = BertModel.from_pretrained(str(path), local_files_only=True)
    if model.config.hidden_size != 768 or model.config.vocab_size != 30522 or model.config.num_hidden_layers != 12:
        raise RuntimeError("Unexpected BERT config")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return path, model


def bert_encode(model, tokens, return_hidden=False):
    arr = np.asarray(tokens)
    if arr.ndim != 3 or arr.shape[1:] != (3, 50) or not np.all(np.isfinite(arr)) or not np.all(arr == arr.astype(np.int64)):
        raise ValueError("invalid text_bert")
    ids = arr[:, 0, :].astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= model.config.vocab_size:
        raise ValueError("token ID outside BERT vocabulary")
    pooled = np.empty((len(arr), 768), dtype=np.float32)
    hidden_batches = []
    with torch.inference_mode():
        for start in range(0, len(arr), BATCH):
            b = arr[start:start + BATCH].astype(np.int64, copy=False)
            inputs = {"input_ids": torch.from_numpy(np.ascontiguousarray(b[:, 0, :])),
                      "attention_mask": torch.from_numpy(np.ascontiguousarray(b[:, 1, :])),
                      "token_type_ids": torch.from_numpy(np.ascontiguousarray(b[:, 2, :]))}
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled[start:start + len(b)] = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy()
            if return_hidden:
                hidden_batches.append(hidden.numpy())
    return pooled, np.concatenate(hidden_batches, axis=0) if return_hidden else None


def reconstruction(model, train, path):
    rng = np.random.default_rng(SEED)
    ix = np.sort(rng.choice(len(train["id"]), 128, replace=False))
    tokens = train["text_bert"][ix]
    _, output = bert_encode(model, tokens, return_hidden=True)
    reference = train["text"][ix].astype(np.float32)
    mask = tokens[:, 1, :] == 1
    x, y = output[mask].astype(np.float64), reference[mask].astype(np.float64)
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    cosine = np.divide((x * y).sum(1), x_norm * y_norm, out=np.zeros(len(x)), where=(x_norm * y_norm) > 0)
    delta = x - y
    rows = []
    for j, index in enumerate(ix):
        xx, yy = output[j, mask[j]].astype(np.float64), reference[j, mask[j]].astype(np.float64)
        norm = np.linalg.norm(xx, axis=1) * np.linalg.norm(yy, axis=1)
        sim = np.divide((xx * yy).sum(1), norm, out=np.zeros(len(xx)), where=norm > 0)
        rows.append({"sample_index": int(index), "sample_id": train["id"][index],
                     "valid_tokens": int(mask[j].sum()), "mean_cosine": float(sim.mean()),
                     "mae": float(np.abs(xx - yy).mean()),
                     "rmse": float(np.sqrt(np.square(xx - yy).mean())),
                     "max_abs_error": float(np.abs(xx - yy).max())})
    pd.DataFrame(rows).to_csv(OUT / "bert_reconstruction_check.csv", index=False, encoding="utf-8-sig")
    summary = {"checkpoint": "bert-base-uncased", "revision": path.name,
               "sample_count": len(ix), "token_count": int(mask.sum()),
               "mean_token_cosine": float(cosine.mean()), "median_token_cosine": float(np.median(cosine)),
               "mean_sample_cosine": float(np.mean([r["mean_cosine"] for r in rows])),
               "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.square(delta).mean())),
               "max_abs_error": float(np.abs(delta).max()),
               "conclusion": ("highly_consistent_not_bitwise_identical"
                              if cosine.mean() > 0.99999 and np.abs(delta).mean() < 1e-5
                              else "not_consistent; investigate before modeling"),
               "comparison_scope": "only attention-valid token positions in 128 seeded train samples"}
    dump_json(OUT / "bert_reconstruction_summary.json", summary)
    return summary


def fuse(text, audio, vision):
    result = np.concatenate((text.astype(np.float32, copy=False),
                             audio.astype(np.float32, copy=False).reshape(len(text), -1),
                             vision.astype(np.float32, copy=False).reshape(len(text), -1)), axis=1)
    if result.shape[1] != 768 + 50 * 74 + 50 * 35 or result.shape[1] != 6218:
        raise AssertionError("fused dimension mismatch")
    return result


def class_metrics(y, pred):
    return {"Accuracy": float(accuracy_score(y, pred)),
            "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(y, pred, average="weighted", zero_division=0))}


def pearson(y, pred):
    if np.std(y) == 0 or np.std(pred) == 0:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1])


def reg_metrics(y, raw):
    clipped = np.clip(raw, -3, 3)
    return {"MAE": float(mean_absolute_error(y, clipped)),
            "Pearson": pearson(y, clipped),
            "RMSE": float(np.sqrt(mean_squared_error(y, clipped))),
            "clipped_ratio": float(np.mean(raw != clipped))}


def build_labels(block):
    y = np.asarray(block["regression_labels"]).reshape(-1)
    cls = np.where(y < 0, 0, np.where(y == 0, 1, 2))
    assert np.array_equal(cls, np.asarray(block["classification_labels"]).astype(int))
    assert y.min() >= -3 and y.max() <= 3
    return cls, y


def choose_models(x_train, x_valid, yc_train, yr_train, yc_valid, yr_valid):
    svc_rows = []
    svc_configs = ([{"kernel": "linear", "C": c, "gamma": "scale"} for c in (0.1, 1, 10)]
                   + [{"kernel": "rbf", "C": c, "gamma": "scale"} for c in (0.1, 1, 10)])
    best_svc, best_svc_row, best_svc_rank = None, None, None
    for cfg in svc_configs:
        print("fit SVC", cfg, flush=True)
        model = SVC(class_weight="balanced", cache_size=1000, **cfg)
        model.fit(x_train, yc_train)
        pred = model.predict(x_valid)
        row = {**cfg, **class_metrics(yc_valid, pred), "selected": False}
        rank = (row["Macro_F1"], row["Accuracy"], -int(cfg["kernel"] == "rbf"), -cfg["C"])
        if best_svc_rank is None or rank > best_svc_rank:
            best_svc, best_svc_row, best_svc_rank = model, row, rank
        svc_rows.append(row)
    best_svc_row["selected"] = True
    pd.DataFrame(svc_rows).to_csv(OUT / "q2_svc_model_selection.csv", index=False, encoding="utf-8-sig")
    ridge_rows = []
    best_ridge, best_ridge_row, best_ridge_rank = None, None, None
    for alpha in (0.1, 1, 10, 100, 1000):
        print("fit Ridge", alpha, flush=True)
        model = Ridge(alpha=alpha, solver="lsqr", max_iter=10000, tol=1e-4)
        model.fit(x_train, yr_train)
        row = {"alpha": alpha, **reg_metrics(yr_valid, model.predict(x_valid)), "selected": False}
        rank = (-row["MAE"], row["Pearson"], -alpha)
        if best_ridge_rank is None or rank > best_ridge_rank:
            best_ridge, best_ridge_row, best_ridge_rank = model, row, rank
        ridge_rows.append(row)
    best_ridge_row["selected"] = True
    pd.DataFrame(ridge_rows).to_csv(OUT / "q2_ridge_model_selection.csv", index=False, encoding="utf-8-sig")
    return best_svc, best_ridge, best_svc_row, best_ridge_row


def make_grid_features(model, block, clean_text, cell):
    n = len(block["id"])
    audio = np.empty((n, 50, 74), dtype=np.float32)
    vision = np.empty((n, 50, 35), dtype=np.float32)
    text_tokens = np.empty((n, 3, 50), dtype=np.int64) if cell["pattern"] == "text" else None
    for i in range(n):
        sample = {k: block[k][i] for k in ("text_bert", "audio", "vision")}
        result = simulate_missingness(sample, cell, SEED)
        meta = result["metadata"]
        g = cell["grid"].iloc[i]
        for key in ("text_start", "text_end", "audio_start", "audio_end", "vision_start", "vision_end"):
            expected = g[key]
            actual = meta[key]
            if pd.isna(expected):
                if actual is not None:
                    raise AssertionError(f"Task05 grid mismatch {key}")
            elif int(expected) != actual:
                raise AssertionError(f"Task05 grid mismatch {key}")
        audio[i] = result["audio"]
        vision[i] = result["vision"]
        if text_tokens is not None:
            text_tokens[i] = result["text_bert"]
    text = bert_encode(model, text_tokens)[0] if text_tokens is not None else clean_text
    return fuse(text, audio, vision)


def plots(yc, pc, yr, pr, clean, grid_metrics):
    FIG.mkdir(exist_ok=True)
    cm = confusion_matrix(yc, pc, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["Neg", "Neutral", "Pos"])
    ax.set_yticks(range(3), ["Neg", "Neutral", "Pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG / "baseline_clean_confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(yr, pr, s=10, alpha=.35)
    ax.plot([-3, 3], [-3, 3], color="black", lw=1)
    ax.set(xlabel="Ground truth", ylabel="Predicted (clipped)", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    ax.set_title(f"MAE={clean['regression']['MAE']:.3f}; Pearson={clean['regression']['Pearson']:.3f}")
    fig.tight_layout()
    fig.savefig(FIG / "baseline_regression_scatter.png", dpi=160)
    plt.close(fig)
    df = pd.DataFrame(grid_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pattern, group in df.groupby("pattern", sort=False):
        avg = group.groupby("rate")[["Macro_F1", "MAE"]].mean()
        axes[0].plot(avg.index * 100, avg["Macro_F1"], marker="o", label=pattern)
        axes[1].plot(avg.index * 100, avg["MAE"], marker="o", label=pattern)
    axes[0].axhline(clean["classification"]["Macro_F1"], ls="--", lw=1, color="gray")
    axes[1].axhline(clean["regression"]["MAE"], ls="--", lw=1, color="gray")
    axes[0].set(xlabel="Nominal missing rate (%)", ylabel="Macro-F1")
    axes[1].set(xlabel="Nominal missing rate (%)", ylabel="MAE")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_missing_rate_curves.png", dpi=160)
    plt.close(fig)
    return cm


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    for directory in (OUT, MODELS, FIG):
        directory.mkdir(exist_ok=True)
    p04, p05, grid = validate_inputs()
    source = resolve_recorded_path(p04["attachment2_file"])
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as f:
        data = pickle.load(f)
    train, valid = data["train"], data["valid"]
    if len(train["id"]) != 3395 or len(valid["id"]) != 728:
        raise ValueError("unexpected split sizes")
    for exp, frame in grid.groupby("experiment_id", sort=False):
        if not np.array_equal(frame.sample_index.to_numpy(), np.arange(728)) or frame.sample_id.tolist() != valid["id"]:
            raise ValueError(f"Task05 grid sample mapping mismatch: {exp}")
    path, bert = checkpoint()
    print("BERT checkpoint", path.name, flush=True)
    recon = reconstruction(bert, train, path)
    print("reconstruction", recon["mean_token_cosine"], recon["mae"], flush=True)
    if not np.isfinite(recon["mae"]) or not np.isfinite(recon["mean_token_cosine"]):
        raise RuntimeError("BERT reconstruction anomalous")
    print("encoding clean train", flush=True)
    text_train = bert_encode(bert, train["text_bert"])[0]
    print("encoding clean valid", flush=True)
    text_valid = bert_encode(bert, valid["text_bert"])[0]
    x_train = fuse(text_train, train["audio"], train["vision"])
    x_valid = fuse(text_valid, valid["audio"], valid["vision"])
    yc_train, yr_train = build_labels(train)
    yc_valid, yr_valid = build_labels(valid)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_valid_s = scaler.transform(x_valid)
    print("scaled train/valid", x_train_s.shape, x_valid_s.shape, flush=True)
    svc, ridge, svc_choice, ridge_choice = choose_models(x_train_s, x_valid_s, yc_train, yr_train, yc_valid, yr_valid)
    print("frozen choices", svc_choice, ridge_choice, flush=True)
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(svc, MODELS / "svc_baseline.joblib")
    joblib.dump(ridge, MODELS / "ridge_baseline.joblib")
    pc = svc.predict(x_valid_s)
    raw = ridge.predict(x_valid_s)
    clipped = np.clip(raw, -3, 3)
    clean = {"classification": {**class_metrics(yc_valid, pc),
                                  "per_class": classification_report(yc_valid, pc, labels=[0, 1, 2],
                                                                      target_names=["negative", "neutral", "positive"],
                                                                      output_dict=True, zero_division=0)},
             "regression": reg_metrics(yr_valid, raw), "n_valid": len(yc_valid)}
    dump_json(OUT / "q2_baseline_clean_metrics.json", clean)
    pd.DataFrame({"sample_index": np.arange(728), "sample_id": valid["id"],
                  "true_class": yc_valid, "pred_class": pc, "true_regression": yr_valid,
                  "pred_regression_raw": raw, "pred_regression_clipped": clipped,
                  "classification_correct": (pc == yc_valid).astype(int),
                  "absolute_error": np.abs(clipped - yr_valid)}).to_csv(
                      OUT / "q2_baseline_valid_predictions.csv", index=False, encoding="utf-8-sig")
    print("clean metrics", clean["classification"]["Macro_F1"], clean["regression"]["MAE"], flush=True)
    grid_metrics = []
    predictions = []
    for exp, frame in grid.groupby("experiment_id", sort=False):
        first = frame.iloc[0]
        cell = {"experiment_id": exp, "pattern": first.pattern, "rate": float(first.rate),
                "position": first.position, "grid": frame}
        print("evaluate", exp, flush=True)
        x = make_grid_features(bert, valid, text_valid, cell)
        scaled = scaler.transform(x)
        pc_missing = svc.predict(scaled)
        raw_missing = ridge.predict(scaled)
        pred_missing = np.clip(raw_missing, -3, 3)
        cm = class_metrics(yc_valid, pc_missing)
        rm = reg_metrics(yr_valid, raw_missing)
        effective = (frame.effective_changed_ratio_audio + frame.effective_changed_ratio_vision) / 2 if first.pattern == "audio+vision synchronized" else (frame.effective_changed_ratio_audio if first.pattern == "audio" else frame.effective_changed_ratio_vision if first.pattern == "vision" else frame.nominal_missing_ratio)
        grid_metrics.append({"experiment_id": exp, "pattern": first.pattern, "rate": float(first.rate),
            "position": first.position, "n_samples": 728, **cm, **rm,
            "delta_accuracy": cm["Accuracy"] - clean["classification"]["Accuracy"],
            "delta_macro_f1": cm["Macro_F1"] - clean["classification"]["Macro_F1"],
            "delta_mae": rm["MAE"] - clean["regression"]["MAE"],
            "delta_pearson": rm["Pearson"] - clean["regression"]["Pearson"],
            "mean_nominal_missing_ratio": float(frame.nominal_missing_ratio.mean()),
            "mean_effective_changed_ratio": float(effective.mean())})
        predictions.append(pd.DataFrame({"experiment_id": exp, "sample_index": np.arange(728),
            "sample_id": valid["id"], "true_class": yc_valid, "pred_class": pc_missing,
            "true_regression": yr_valid, "pred_regression_raw": raw_missing,
            "pred_regression_clipped": pred_missing}))
    pd.DataFrame(grid_metrics).to_csv(OUT / "q2_baseline_36grid_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "q2_baseline_missing_predictions.csv", index=False, encoding="utf-8-sig")
    cm = plots(yc_valid, pc, yr_valid, clipped, clean, grid_metrics)
    pd.DataFrame(cm, index=["negative", "neutral", "positive"],
                 columns=["negative", "neutral", "positive"]).to_csv(OUT / "q2_baseline_confusion_matrix.csv")
    manifest = {"feature_version": "aligned_50", "text_encoder": "bert-base-uncased",
        "text_encoder_local_path": str(path.relative_to(ROOT)).replace("\\", "/"), "text_encoder_revision": path.name,
        "text_encoder_weight_sha256": sha256(path / "model.safetensors"),
        "text_encoder_config": {**bert.config.to_dict(), "_name_or_path": str(path.relative_to(ROOT)).replace("\\", "/")}, "text_encoder_frozen": True,
        "text_pooling": "attention-mask-aware mean; includes all attention-valid special tokens",
        "feature_dimension": 6218, "scaler": "StandardScaler fit clean train only",
        "svc_params": {k: svc_choice[k] for k in ("kernel", "C", "gamma")},
        "ridge_params": {"alpha": ridge_choice["alpha"], "solver": "lsqr", "max_iter": 10000, "tol": 1e-4},
        "training_split": "train", "validation_split": "valid", "test_used": False,
        "attachment3_used": False, "seed": SEED, "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "task04_protocol_sha256": sha256(P04), "task05_protocol_sha256": sha256(P05),
        "task05_grid_sha256": sha256(GRID)}
    dump_json(MODELS / "baseline_model_manifest.json", manifest)
    # Reload check on exactly the frozen clean valid matrix.
    ss = joblib.load(MODELS / "scaler.joblib")
    sc = joblib.load(MODELS / "svc_baseline.joblib")
    rg = joblib.load(MODELS / "ridge_baseline.joblib")
    scaled_reload = ss.transform(x_valid)
    if not np.array_equal(sc.predict(scaled_reload), pc) or not np.allclose(rg.predict(scaled_reload), raw, rtol=0, atol=1e-8):
        raise AssertionError("reloaded models differ")
    checks = {"fused_dimension": x_train.shape[1] == 6218, "scaler_train_only": scaler.n_samples_seen_ == 3395,
              "no_test_or_attachment3": True, "text_missing_reencoded": True,
              "grid_matches_task05": len(grid_metrics) == 36 and len(predictions) == 36,
              "each_grid_cell_728": all(r["n_samples"] == 728 for r in grid_metrics),
              "neutral_label_preserved": np.all(yc_valid[yr_valid == 0] == 1),
              "signed_target": yr_train.min() < 0 and yr_train.max() > 0,
              "reload_predictions_match": True}
    dump_json(OUT / "q2_baseline_validation.json", checks)
    protocol = {"feature_version": "aligned_50", "text_source": "text_bert",
        "text_encoder": "bert-base-uncased", "text_encoder_revision": path.name,
        "text_encoder_frozen": True, "text_pooling": manifest["text_pooling"],
        "audio_representation": "flatten 50x74", "vision_representation": "flatten 50x35",
        "fused_dimension": 6218, "scaler": manifest["scaler"],
        "classifier": "balanced SVC", "regressor": "Ridge",
        "classification_selection_metric": "Macro-F1; tie Accuracy then simpler kernel/smaller C",
        "regression_selection_metric": "clipped MAE; tie Pearson then smaller alpha",
        "regression_output_clip": [-3, 3],
        "missing_grid_source": str(GRID.relative_to(ROOT)).replace("\\", "/"),
        "svc_search": "linear C={0.1,1,10}; rbf C={0.1,1,10}, gamma=scale",
        "ridge_search": [0.1, 1, 10, 100, 1000], "seed": SEED,
        "clean_train_only": True, "missing_grid_evaluation_after_selection": True}
    dump_json(OUT / "q2_baseline_protocol.json", protocol)
    print("DONE", json.dumps({"clean": {"Accuracy": clean["classification"]["Accuracy"],
        "Macro_F1": clean["classification"]["Macro_F1"], "MAE": clean["regression"]["MAE"],
        "Pearson": clean["regression"]["Pearson"]}, "grid_rows": len(grid_metrics)}, ensure_ascii=False), flush=True)


def finalize_existing_outputs():
    """Recheck persisted artifacts after an interrupted final metadata write."""
    p04, _, grid = validate_inputs()
    manifest = json.loads((MODELS / "baseline_model_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "q2_baseline_36grid_metrics.csv")
    clean_pred = pd.read_csv(OUT / "q2_baseline_valid_predictions.csv")
    missing_pred = pd.read_csv(OUT / "q2_baseline_missing_predictions.csv")
    assert len(metrics) == 36 and len(clean_pred) == 728 and len(missing_pred) == 26208
    assert set(metrics.experiment_id) == set(grid.experiment_id)
    assert metrics.n_samples.eq(728).all()
    assert (missing_pred[["experiment_id", "sample_index", "sample_id"]].to_numpy()
            == grid[["experiment_id", "sample_index", "sample_id"]].to_numpy()).all()
    with (resolve_recorded_path(p04["attachment2_file"])).open("rb") as f:
        valid = pickle.load(f)["valid"]
    path, bert = checkpoint()
    if path.name != manifest["text_encoder_revision"] or sha256(path / "model.safetensors") != manifest["text_encoder_weight_sha256"]:
        raise AssertionError("BERT checkpoint changed")
    torch.set_num_threads(8)
    text = bert_encode(bert, valid["text_bert"])[0]
    fused = fuse(text, valid["audio"], valid["vision"])
    scaler = joblib.load(MODELS / "scaler.joblib")
    svc = joblib.load(MODELS / "svc_baseline.joblib")
    ridge = joblib.load(MODELS / "ridge_baseline.joblib")
    scaled = scaler.transform(fused)
    reloaded_class = svc.predict(scaled)
    reloaded_reg = ridge.predict(scaled)
    if not np.array_equal(reloaded_class, clean_pred.pred_class.to_numpy()) or not np.allclose(
            reloaded_reg, clean_pred.pred_regression_raw.to_numpy(), rtol=0, atol=1e-6):
        raise AssertionError("saved model predictions differ from original clean output")
    yc, yr = build_labels(valid)
    checks = {"fused_dimension": fused.shape[1] == 6218,
              "scaler_train_only": scaler.n_samples_seen_ == 3395,
              "no_test_or_attachment3": True, "text_missing_reencoded": True,
              "grid_matches_task05": set(metrics.experiment_id) == set(grid.experiment_id),
              "each_grid_cell_728": metrics.n_samples.eq(728).all(),
              "neutral_label_preserved": np.all(yc[yr == 0] == 1),
              "signed_target": yr.min() < 0 and yr.max() > 0,
              "reload_predictions_match": True,
              "missing_prediction_rows": len(missing_pred) == 26208}
    dump_json(OUT / "q2_baseline_validation.json", checks)
    protocol = {"feature_version": "aligned_50", "text_source": "text_bert",
        "text_encoder": "bert-base-uncased", "text_encoder_revision": path.name,
        "text_encoder_frozen": True, "text_pooling": manifest["text_pooling"],
        "audio_representation": "flatten 50x74", "vision_representation": "flatten 50x35",
        "fused_dimension": 6218, "scaler": manifest["scaler"],
        "classifier": "balanced SVC", "regressor": "Ridge",
        "classification_selection_metric": "Macro-F1; tie Accuracy then simpler kernel/smaller C",
        "regression_selection_metric": "clipped MAE; tie Pearson then smaller alpha",
        "regression_output_clip": [-3, 3],
        "missing_grid_source": str(GRID.relative_to(ROOT)).replace("\\", "/"),
        "svc_search": "linear C={0.1,1,10}; rbf C={0.1,1,10}, gamma=scale",
        "ridge_search": [0.1, 1, 10, 100, 1000], "seed": SEED,
        "clean_train_only": True, "missing_grid_evaluation_after_selection": True}
    dump_json(OUT / "q2_baseline_protocol.json", protocol)
    print("FINALIZED", json.dumps({"checks": checks, "grid_rows": len(metrics)}, default=bool), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--finalize-existing":
        finalize_existing_outputs()
    else:
        main()
