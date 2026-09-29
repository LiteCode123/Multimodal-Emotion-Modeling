# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task13: train and freeze a mask-aware, three-modality multitask predictor."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/model"
OUT = TASK / "outputs"
FIG = TASK / "figures"
PROTOCOL = ROOT / "results/q3/protocol/outputs/q3_explainability_protocol.json"
MODS = ("text", "audio", "vision")
SEED = 2026


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def structural_mask(block):
    attention = np.asarray(block["text_bert"])[:, 1, :]
    lengths = attention.sum(axis=1).astype(int)
    indices = np.arange(attention.shape[1])[None, :]
    mask = (indices >= 1) & (indices < lengths[:, None] - 1)
    if not np.all(mask.sum(axis=1) >= 1):
        raise ValueError("Empty structural content mask")
    if not np.all(np.diff(attention, axis=1) <= 0):
        raise ValueError("text_bert attention is not a prefix mask")
    return mask


def check_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["sequence_length"] == 50
    assert protocol["padding_rule"]["token_content_index_rule"] == "1 <= i < sum(text_bert[1,:]) - 1"
    assert tuple(protocol["modalities"]) == MODS
    return resolve_recorded_path(protocol["source"]["attachment2"])


def fit_preprocessing(train):
    mask = structural_mask(train)
    count = int(mask.sum())
    stats = {}
    for mod in MODS:
        x = np.asarray(train[mod])
        sums = np.zeros(x.shape[-1], dtype=np.float64)
        squares = np.zeros_like(sums)
        for start in range(0, len(x), 128):
            rows = x[start:start + 128][mask[start:start + 128]].astype(np.float64)
            sums += rows.sum(axis=0)
            squares += np.square(rows).sum(axis=0)
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
        std = np.maximum(std, 1e-6)
        stats[f"{mod}_mean"] = mean.astype(np.float32)
        stats[f"{mod}_std"] = std.astype(np.float32)
    stats["content_position_count"] = np.array(count, dtype=np.int64)
    return stats


def pooled_features(block, stats):
    mask = structural_mask(block)
    count = mask.sum(axis=1).astype(np.float32)[:, None]
    result = {}
    for mod in MODS:
        x = np.asarray(block[mod])
        pooled = np.empty((len(x), x.shape[-1]), dtype=np.float32)
        for start in range(0, len(x), 128):
            end = min(start + 128, len(x))
            pooled[start:end] = (x[start:end] * mask[start:end, :, None]).sum(axis=1) / count[start:end]
        result[mod] = (pooled - stats[f"{mod}_mean"]) / stats[f"{mod}_std"]
        if not np.isfinite(result[mod]).all():
            raise ValueError(f"Nonfinite {mod} standardized features")
    y_cls = np.asarray(block["classification_labels"], dtype=np.int64).reshape(-1)
    y_reg = np.asarray(block["regression_labels"], dtype=np.float32).reshape(-1)
    assert np.array_equal(y_cls, np.where(y_reg < 0, 0, np.where(y_reg > 0, 2, 1)))
    return {
        "x": result,
        "class": y_cls,
        "reg": y_reg,
        "ids": [str(i) for i in block["id"]],
        "visual_all_zero": np.all(np.asarray(block["vision"]) == 0, axis=(1, 2)),
        "mask": mask,
    }


class Q3Model(nn.Module):
    def __init__(self, fusion="concat", hidden=128, dropout=0.2):
        super().__init__()
        self.fusion = fusion
        self.encoders = nn.ModuleDict({
            mod: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
            for mod, dim in (("text", 768), ("audio", 74), ("vision", 35))
        })
        if fusion == "gated":
            self.gate = nn.Linear(hidden * 3, 3)
            fusion_dim = hidden
        else:
            fusion_dim = hidden * 3
        self.shared = nn.Sequential(nn.Linear(fusion_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, 3)
        self.reg = nn.Linear(hidden, 1)

    def forward(self, inputs, mask=None):
        h = []
        for mod in MODS:
            x = inputs[mod]
            if x.ndim == 3:
                if mask is None:
                    raise ValueError("A structural mask is required for sequences")
                x = (x * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            h.append(self.encoders[mod](x))
        combined = torch.cat(h, dim=-1)
        if self.fusion == "gated":
            weights = torch.softmax(self.gate(combined), dim=-1)
            combined = sum(weights[:, i:i + 1] * value for i, value in enumerate(h))
        z = self.shared(combined)
        return self.cls(z), self.reg(z).squeeze(-1)


def metrics(y_cls, pred_cls, y_reg, pred_reg):
    f1 = f1_score(y_cls, pred_cls, labels=[0, 1, 2], average=None, zero_division=0)
    pearson = float(np.corrcoef(y_reg, pred_reg)[0, 1]) if np.std(pred_reg) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_cls, pred_cls)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": dict(zip(("Negative", "Neutral", "Positive"), map(float, f1))),
        "mae": float(mean_absolute_error(y_reg, pred_reg)),
        "pearson": pearson,
        "confusion_matrix": confusion_matrix(y_cls, pred_cls, labels=[0, 1, 2]).tolist(),
    }


def predict(model, data, batch_size=512):
    model.eval()
    logits, scores = [], []
    with torch.no_grad():
        for start in range(0, len(data["class"]), batch_size):
            x = {m: torch.from_numpy(data["x"][m][start:start + batch_size]) for m in MODS}
            a, b = model(x)
            logits.append(a.numpy())
            scores.append(b.numpy())
    logit = np.concatenate(logits)
    score = np.clip(np.concatenate(scores), -3, 3)
    return logit.argmax(axis=1), score


def train_one(config, train, valid, class_weights):
    seed_all()
    model = Q3Model(config["fusion"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    weights = class_weights if config["weighted"] else None
    ce = nn.CrossEntropyLoss(weight=weights)
    reg_loss = nn.SmoothL1Loss()
    arrays = {m: torch.from_numpy(train["x"][m]) for m in MODS}
    y_cls = torch.from_numpy(train["class"])
    y_reg = torch.from_numpy(train["reg"])
    best = None
    best_score = -np.inf
    patience = 0
    history = []
    for epoch in range(1, 61):
        model.train()
        losses = []
        for ix in torch.randperm(len(y_cls)).split(128):
            x = {m: arrays[m][ix] for m in MODS}
            logits, score = model(x)
            loss = ce(logits, y_cls[ix]) + config["lambda_reg"] * reg_loss(score, y_reg[ix])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()) * len(ix))
        pred_cls, pred_reg = predict(model, valid)
        met = metrics(valid["class"], pred_cls, valid["reg"], pred_reg)
        with torch.no_grad():
            vx = {m: torch.from_numpy(valid["x"][m]) for m in MODS}
            a, b = model(vx)
            valid_loss = float((ce(a, torch.from_numpy(valid["class"])) + config["lambda_reg"] * reg_loss(b, torch.from_numpy(valid["reg"]))).item())
        composite = met["macro_f1"] + 0.2 * met["accuracy"] - 0.1 * met["mae"] + 0.1 * met["pearson"]
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(y_cls), "valid_loss": valid_loss,
                        "valid_accuracy": met["accuracy"], "valid_macro_f1": met["macro_f1"],
                        "valid_mae": met["mae"], "valid_pearson": met["pearson"]})
        if composite > best_score + 1e-5:
            best_score = composite
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = met
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == 1:
            print(config["name"], epoch, f"F1={met['macro_f1']:.4f} MAE={met['mae']:.4f}", flush=True)
        if patience >= 8:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_metrics, best_score, history


def error_summary(data, pred_cls, pred_reg):
    errors = np.abs(data["reg"] - pred_reg)
    examples = sorted(range(len(errors)), key=lambda i: -errors[i])[:10]
    groups = {}
    for name, select in (("visual_all_zero", data["visual_all_zero"]),
                         ("visual_nonzero", ~data["visual_all_zero"]),
                         ("true_near_zero", np.abs(data["reg"]) <= 0.25),
                         ("pred_near_zero", np.abs(pred_reg) <= 0.25),
                         ("true_extreme", np.abs(data["reg"]) >= 2.0)):
        groups[name] = {"count": int(select.sum()), "accuracy": float(accuracy_score(data["class"][select], pred_cls[select])) if select.any() else None,
                        "mae": float(np.mean(errors[select])) if select.any() else None}
    return {"groups": groups, "largest_regression_errors": [
        {"sample_id": data["ids"][i], "true_polarity": int(data["class"][i]), "pred_polarity": int(pred_cls[i]),
         "true_intensity": float(data["reg"][i]), "pred_intensity": float(pred_reg[i]), "absolute_error": float(errors[i]),
         "visual_all_zero": bool(data["visual_all_zero"][i])} for i in examples]}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_training_curve(history):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["valid_loss"] for r in history], label="valid")
    axes[0].set(xlabel="Epoch", ylabel="Multitask loss")
    axes[0].set_ylim(top=2.45)
    axes[0].legend(loc="upper right")
    axes[1].plot(epochs, [r["valid_macro_f1"] for r in history], label="Macro-F1")
    axes[1].plot(epochs, [r["valid_mae"] for r in history], label="MAE")
    axes[1].set(xlabel="Epoch", ylabel="Valid metric")
    axes[1].set_ylim(top=.68)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "q3_training_curve.png", dpi=160)
    plt.close(fig)


def figures(history, test, pred_cls, pred_reg):
    plot_training_curve(history)
    cm = confusion_matrix(test["class"], pred_cls, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=["Neg", "Neutral", "Pos"],
           yticklabels=["Neg", "Neutral", "Pos"], xlabel="Predicted", ylabel="True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG / "q3_confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(test["reg"], pred_reg, s=8, alpha=0.35)
    ax.plot([-3, 3], [-3, 3], color="black", linewidth=1)
    ax.set(xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    fig.tight_layout()
    fig.savefig(FIG / "q3_regression_scatter.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--allow-overwrite", action="store_true", help="Explicitly replace a frozen Task13 run")
    parser.add_argument("--training-curve-only", action="store_true", help="Redraw the training curve from saved history")
    args = parser.parse_args()
    if args.training_curve_only:
        with (OUT / "q3_training_history.csv").open(newline="", encoding="utf-8-sig") as fh:
            history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(fh)]
        plot_training_curve(history)
        return
    if (OUT / "q3_final_model.pt").exists() and not args.allow_overwrite:
        raise FileExistsError("Task13 model is frozen; use a clean checkout or --allow-overwrite to rerun")
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    seed_all()
    source = check_protocol()
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("rb") as fh:
        raw = pickle.load(fh)
    assert all(k in raw for k in ("train", "valid", "test"))
    stats = fit_preprocessing(raw["train"])
    np.savez_compressed(OUT / "q3_preprocessing.npz", **stats)
    train = pooled_features(raw["train"], stats)
    valid = pooled_features(raw["valid"], stats)
    counts = np.bincount(train["class"], minlength=3)
    class_weights = torch.from_numpy((len(train["class"]) / (3 * counts)).astype(np.float32))
    configs = [
        {"name": "A_concat_plain", "fusion": "concat", "weighted": False, "lambda_reg": 1.0},
        {"name": "A_concat_weighted", "fusion": "concat", "weighted": True, "lambda_reg": 1.0},
        {"name": "C_gated_weighted", "fusion": "gated", "weighted": True, "lambda_reg": 1.0},
        {"name": "A_concat_lambda05", "fusion": "concat", "weighted": False, "lambda_reg": 0.5},
    ][:args.max_candidates]
    results = []
    for config in configs:
        model, epoch, met, score, history = train_one(config, train, valid, class_weights)
        results.append((config, model, epoch, met, score, history))
    winner = max(results, key=lambda r: r[4])
    config, model, epoch, valid_met, score, history = winner
    print("LOCKED", config["name"], "epoch", epoch, "valid", valid_met, flush=True)
    rows = []
    for candidate, _, best_epoch, met, comp, _ in results:
        rows.append({"model": candidate["name"], "pooling": "structural_masked_mean", "fusion": candidate["fusion"],
                     "loss_setting": f"CE_weighted={candidate['weighted']};SmoothL1_lambda={candidate['lambda_reg']}",
                     "accuracy_valid": met["accuracy"], "macro_f1_valid": met["macro_f1"],
                     "mae_valid": met["mae"], "pearson_valid": met["pearson"],
                     "selection_score": comp, "best_epoch": best_epoch, "selected": candidate["name"] == config["name"]})
    write_csv(OUT / "q3_model_comparison.csv", rows)
    write_csv(OUT / "q3_model_selection.csv", [{"selected_model": config["name"],
        "best_epoch": epoch, "selection_score": score,
        "selection_rule": "valid_macro_f1+0.2*accuracy-0.1*MAE+0.1*Pearson"}])
    write_csv(OUT / "q3_training_history.csv", history)
    model_config = {**config, "hidden": 128, "dropout": 0.2, "seed": SEED, "batch_size": 128,
                    "learning_rate": 8e-4, "weight_decay": 1e-3, "max_epochs": 60, "patience": 8,
                    "best_epoch": epoch, "selection_rule": "max(valid_macro_f1 + 0.2*accuracy - 0.1*MAE + 0.1*Pearson)",
                    "preprocessing": "train content-position z-score; mean pooling after normalization",
                    "model_input": "preprocessed [B,50,D] and structural mask [B,50], or equivalent pooled [B,D]"}
    write_json(OUT / "q3_model_config.json", model_config)
    torch.save(model.state_dict(), OUT / "q3_final_model.pt")
    # The test split is first transformed and scored only after the model has been selected and saved.
    test = pooled_features(raw["test"], stats)
    valid_pred_cls, valid_pred_reg = predict(model, valid)
    test_pred_cls, test_pred_reg = predict(model, test)
    valid_met = metrics(valid["class"], valid_pred_cls, valid["reg"], valid_pred_reg)
    test_met = metrics(test["class"], test_pred_cls, test["reg"], test_pred_reg)
    final = {"valid": valid_met, "test": test_met,
             "visual_zero_counts": {"train": int(train["visual_all_zero"].sum()), "valid": int(valid["visual_all_zero"].sum()), "test": int(test["visual_all_zero"].sum())},
             "valid_error_analysis": error_summary(valid, valid_pred_cls, valid_pred_reg),
             "test_error_analysis": error_summary(test, test_pred_cls, test_pred_reg)}
    write_json(OUT / "q3_final_metrics.json", final)
    write_json(OUT / "q3_test_metrics.json", test_met)
    write_csv(OUT / "q3_test_predictions.csv", [
        {"sample_id": test["ids"][i], "true_polarity": int(test["class"][i]), "pred_polarity": int(test_pred_cls[i]),
         "true_intensity": float(test["reg"][i]), "pred_intensity": float(test_pred_reg[i]),
         "visual_all_zero": bool(test["visual_all_zero"][i])} for i in range(len(test["ids"]))])
    figures(history, test, test_pred_cls, test_pred_reg)
    print("TEST", json.dumps(test_met), flush=True)


if __name__ == "__main__":
    main()
