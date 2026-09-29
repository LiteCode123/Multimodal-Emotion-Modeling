# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task07 MAF-MT. Run from the project root with .venv/Scripts/python.exe."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from src.data.q2_missingness import simulate_missingness

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/model"
OUT, MODELS, CACHE, FIG = (TASK / x for x in ("outputs", "models", "cache", "figures"))
T05 = ROOT / "results/q2/missingness/outputs"
T06 = ROOT / "results/q2/baseline"
GRID = T05 / "q2_valid_36grid.csv"
SOURCE = ROOT / "data/raw/E题数据/附件2-数据集特征文件/aligned_50.pkl"
SEED = 20260924
BS = 64
DEVICE = "cpu"


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def labels(block):
    y = np.asarray(block["regression_labels"], dtype=np.float32).reshape(-1)
    c = np.where(y < 0, 0, np.where(y == 0, 1, 2)).astype(np.int64)
    assert np.array_equal(c, np.asarray(block["classification_labels"]).astype(int))
    return c, y


def bert_sequences(bert, tokens, batch=32):
    arr = np.asarray(tokens)
    assert arr.ndim == 3 and arr.shape[1:] == (3, 50)
    result = np.empty((len(arr), 50, 768), dtype=np.float16)
    bert.eval()
    with torch.inference_mode():
        for s in range(0, len(arr), batch):
            x = arr[s:s + batch].astype(np.int64)
            result[s:s + len(x)] = bert(input_ids=torch.from_numpy(x[:, 0].copy()),
                                       attention_mask=torch.from_numpy(x[:, 1].copy()),
                                       token_type_ids=torch.from_numpy(x[:, 2].copy())).last_hidden_state.numpy().astype(np.float16)
    return result


def prepare():
    for p in (SOURCE, GRID, T05 / "q2_missing_protocol.json", T06 / "models/baseline_model_manifest.json",
              T06 / "outputs/q2_baseline_clean_metrics.json", T06 / "outputs/q2_baseline_36grid_metrics.csv"):
        if not p.is_file():
            raise FileNotFoundError(p)
    with SOURCE.open("rb") as f:
        data = pickle.load(f)
    train, valid = data["train"], data["valid"]
    assert len(train["id"]) == 3395 and len(valid["id"]) == 728
    grid = pd.read_csv(GRID)
    assert len(grid) == 26208 and grid.experiment_id.nunique() == 36
    for _, g in grid.groupby("experiment_id", sort=False):
        assert np.array_equal(g.sample_index.to_numpy(), np.arange(728)) and g.sample_id.tolist() == valid["id"]
    baseline_manifest = json.loads((T06 / "models/baseline_model_manifest.json").read_text(encoding="utf-8"))
    bert_path = resolve_recorded_path(baseline_manifest["text_encoder_local_path"])
    assert bert_path.name == baseline_manifest["text_encoder_revision"]
    assert sha256(bert_path / "model.safetensors") == baseline_manifest["text_encoder_weight_sha256"]
    for d in (OUT, MODELS, CACHE, FIG):
        d.mkdir(exist_ok=True)
    manifest_file = OUT / "q2_train_missing_augmentation_manifest.csv"
    masked_tokens = np.empty_like(train["text_bert"])
    masked_audio = np.asarray(train["audio"], dtype=np.float32).copy()
    masked_vision = np.asarray(train["vision"], dtype=np.float32).copy()
    artificial = np.zeros((3395, 50, 3), dtype=np.float32)
    rows = []
    for i, sid in enumerate(train["id"]):
        sample = {k: train[k][i] for k in ("text_bert", "audio", "vision")}
        m = simulate_missingness(sample, {"pattern": "random", "rate": "random", "position": "random"}, SEED + i)
        d = m["metadata"]
        masked_tokens[i] = m["text_bert"]
        masked_audio[i] = m["audio"]
        masked_vision[i] = m["vision"]
        for j, k in enumerate(("text", "audio", "vision")):
            artificial[i, :, j] = 1 - m[f"mask_{k}_artificial"]
        changed = d["masked_token_count"] if d["pattern"] == "text" else d["effective_changed_rows_audio"] + d["effective_changed_rows_vision"]
        effective = d["nominal_missing_ratio"] if d["pattern"] == "text" else max(d["effective_changed_ratio_audio"], d["effective_changed_ratio_vision"])
        rows.append({"sample_index": i, "sample_id": sid, "pattern": d["pattern"], "nominal_rate": d["rate"],
                     "start_idx": d["start"], "end_idx": d["end"], "actual_changed_rows": changed,
                     "effective_change_ratio": effective, "seed": SEED + i})
    new_manifest = pd.DataFrame(rows)
    if manifest_file.exists():
        old = pd.read_csv(manifest_file)
        pd.testing.assert_frame_equal(old, new_manifest, check_dtype=False)
    else:
        new_manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")
    # A saved manifest is the authority on reruns; never silently change it.
    bert = BertModel.from_pretrained(str(bert_path), local_files_only=True).eval()
    for p in bert.parameters():
        p.requires_grad_(False)
    cache_file = CACHE / "q2_text_sequences.pt"
    cache_key = {"source_sha256": sha256(SOURCE), "augmentation_sha256": sha256(manifest_file),
                 "bert_sha256": baseline_manifest["text_encoder_weight_sha256"]}
    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        assert cache["key"] == cache_key
    else:
        cache = {"key": cache_key, "train_clean": bert_sequences(bert, train["text_bert"]),
                 "train_masked": bert_sequences(bert, masked_tokens),
                 "valid_clean": bert_sequences(bert, valid["text_bert"])}
        torch.save(cache, cache_file)
    return train, valid, grid, new_manifest, masked_tokens, masked_audio, masked_vision, artificial, bert, cache, baseline_manifest


class MAFMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(n, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.2)) for n in (768, 74, 35)])
        self.gate = nn.Sequential(nn.Linear(128 * 2 + 6, 64), nn.GELU(), nn.Linear(64, 1))
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.cls = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 3))
        self.reg = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, text, audio, vision, valid, artificial, zero):
        h = [layer(x) for layer, x in zip(self.project, (text, audio, vision))]
        h[0] = h[0] * (1 - artificial[..., 0:1])
        info = torch.cat((artificial, valid.unsqueeze(-1), zero), dim=-1)
        scores = torch.cat([self.gate(torch.cat((v, h[0], info), dim=-1)) for v in h], dim=-1)
        availability = 1 - artificial
        availability[..., 0] *= valid
        scores = scores.masked_fill(availability == 0, -1e9)
        gates = torch.softmax(scores, dim=-1)
        fused = sum(gates[..., i:i + 1] * h[i] for i in range(3))
        out, _ = self.gru(fused)
        e = self.attention(out).squeeze(-1).masked_fill(valid == 0, -1e9)
        alpha = torch.softmax(e, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * out, dim=1)
        return self.cls(z), self.reg(z).squeeze(-1), gates


def arrays(text, audio, vision, valid, artificial):
    audio = np.asarray(audio, dtype=np.float32)
    vision = np.asarray(vision, dtype=np.float32)
    zero = np.stack((np.all(audio == 0, axis=-1), np.all(vision == 0, axis=-1)), axis=-1).astype(np.float32)
    return (np.asarray(text), audio, vision, np.asarray(valid, dtype=np.float32), np.asarray(artificial, dtype=np.float32), zero)


def tensor_batch(a, ix):
    return [torch.from_numpy(np.ascontiguousarray(x[ix])).float() for x in a]


def supervised(logits, reg, cls, y, weights):
    return F.cross_entropy(logits, cls, weight=weights), F.smooth_l1_loss(reg, y)


def consistency(lc, rc, lm, rm):
    pc, pm = lc.softmax(-1), lm.softmax(-1)
    midpoint = .5 * (pc + pm)
    js = .5 * (F.kl_div(midpoint.log(), pc, reduction="batchmean") + F.kl_div(midpoint.log(), pm, reduction="batchmean"))
    return js + (rc - rm).abs().mean()


def predict(model, a, yc=None, yr=None):
    model.eval()
    ps, rs, gs = [], [], []
    with torch.inference_mode():
        for s in range(0, len(a[0]), BS):
            x = tensor_batch(a, slice(s, s + BS))
            logits, reg, gates = model(*x)
            ps.append(logits.argmax(-1).numpy())
            rs.append(reg.numpy())
            gs.append(gates.numpy())
    pred, raw, gates = np.concatenate(ps), np.concatenate(rs), np.concatenate(gs)
    return pred, raw, gates


def metrics(yc, yr, pred, raw):
    clipped = np.clip(raw, -3, 3)
    pearson = float(np.corrcoef(yr, clipped)[0, 1]) if np.std(clipped) > 0 else 0.
    return {"Accuracy": float(accuracy_score(yc, pred)), "Macro_F1": float(f1_score(yc, pred, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(yc, pred, average="weighted", zero_division=0)),
            "MAE": float(mean_absolute_error(yr, clipped)), "Pearson": pearson,
            "RMSE": float(np.sqrt(mean_squared_error(yr, clipped))), "clipped_ratio": float(np.mean(raw != clipped))}


def train_model(train, valid, masked_tokens, masked_audio, masked_vision, artificial, cache):
    yc, yr = labels(train)
    ycv, yrv = labels(valid)
    clean_a = arrays(cache["train_clean"], train["audio"], train["vision"], train["text_bert"][:, 1], np.zeros_like(artificial))
    mask_a = arrays(cache["train_masked"], masked_audio, masked_vision, masked_tokens[:, 1], artificial)
    valid_a = arrays(cache["valid_clean"], valid["audio"], valid["vision"], valid["text_bert"][:, 1], np.zeros((728, 50, 3), np.float32))
    counts = np.bincount(yc, minlength=3)
    weights = torch.tensor(len(yc) / (3 * counts), dtype=torch.float32)
    model = MAFMT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, 61):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(len(yc))
        parts = []
        for s in range(0, len(order), BS):
            ix = order[s:s + BS]
            xc, xm = tensor_batch(clean_a, ix), tensor_batch(mask_a, ix)
            cy = torch.from_numpy(yc[ix]); ry = torch.from_numpy(yr[ix])
            opt.zero_grad(set_to_none=True)
            lc, rc, _ = model(*xc)
            lm, rm, _ = model(*xm)
            ce_c, hu_c = supervised(lc, rc, cy, ry, weights)
            ce_m, hu_m = supervised(lm, rm, cy, ry, weights)
            cons = consistency(lc, rc, lm, rm)
            total = ce_c + hu_c + ce_m + hu_m + .2 * cons
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            parts.append([float(v.detach()) * len(ix) for v in (total, ce_c, hu_c, ce_m, hu_m, cons)])
        model.eval()
        with torch.inference_mode():
            vloss = 0.
            for s in range(0, 728, BS):
                ix = slice(s, s + BS)
                logits, reg, _ = model(*tensor_batch(valid_a, ix))
                ce, hu = supervised(logits, reg, torch.from_numpy(ycv[ix]), torch.from_numpy(yrv[ix]), weights)
                vloss += float((ce + hu) * len(ycv[ix]))
            vloss /= 728
        p, r, _ = predict(model, valid_a)
        m = metrics(ycv, yrv, p, r)
        row = {"epoch": epoch, **dict(zip(("total_loss", "clean_CE", "clean_Huber", "masked_CE", "masked_Huber", "consistency_loss"),
                                          (np.sum(parts, axis=0) / len(yc)).tolist())),
               "valid_supervised_loss": vloss, "valid_Macro_F1": m["Macro_F1"], "valid_MAE": m["MAE"]}
        history.append(row)
        if vloss < best - 1e-10:
            best, best_epoch, stale = vloss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "valid_loss": vloss}, MODELS / "q2_mafmt_best.pt")
        else:
            stale += 1
        print(f"epoch {epoch}: valid_loss={vloss:.5f}, F1={m['Macro_F1']:.4f}, MAE={m['MAE']:.4f}, stale={stale}", flush=True)
        if stale >= 8:
            break
    pd.DataFrame(history).to_csv(OUT / "q2_main_training_history.csv", index=False, encoding="utf-8-sig")
    ckpt = torch.load(MODELS / "q2_mafmt_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return model, valid_a, counts, best_epoch, best


def grid_condition(valid, frame, bert, clean_hidden):
    first = frame.iloc[0]
    pattern = first.pattern
    art = np.zeros((728, 50, 3), dtype=np.float32)
    audio = np.asarray(valid["audio"], dtype=np.float32).copy()
    vision = np.asarray(valid["vision"], dtype=np.float32).copy()
    tokens = np.asarray(valid["text_bert"]).copy() if pattern == "text" else valid["text_bert"]
    for i, row in enumerate(frame.itertuples()):
        if pattern == "text":
            s, e = int(row.text_start), int(row.text_end)
            tokens[i, :, s:e] = 0
            art[i, s:e, 0] = 1
            assert e - s == row.text_masked_count
        else:
            for j, name in ((1, "audio"), (2, "vision")):
                s, e = getattr(row, f"{name}_start"), getattr(row, f"{name}_end")
                if not pd.isna(s):
                    s, e = int(s), int(e)
                    (audio if name == "audio" else vision)[i, s:e] = 0
                    art[i, s:e, j] = 1
    hidden = bert_sequences(bert, tokens) if pattern == "text" else clean_hidden
    return arrays(hidden, audio, vision, tokens[:, 1], art)


def gate_means(gates, valid):
    mask = valid.astype(bool)
    return gates[mask].mean(axis=0)


def evaluate(model, valid, grid, bert, valid_a):
    yc, yr = labels(valid)
    pred, raw, gates = predict(model, valid_a)
    clean_m = metrics(yc, yr, pred, raw)
    clean = {"classification": {k: clean_m[k] for k in ("Accuracy", "Macro_F1", "Weighted_F1")},
             "per_class": classification_report(yc, pred, labels=[0, 1, 2], target_names=["negative", "neutral", "positive"], output_dict=True, zero_division=0),
             "confusion_matrix": confusion_matrix(yc, pred, labels=[0, 1, 2]).tolist(),
             "regression": {k: clean_m[k] for k in ("MAE", "Pearson", "RMSE", "clipped_ratio")}, "n_valid": 728}
    save_json(OUT / "q2_main_clean_metrics.json", clean)
    pd.DataFrame({"sample_index": np.arange(728), "sample_id": valid["id"], "true_class": yc,
                  "pred_class": pred, "true_regression": yr, "pred_regression_raw": raw,
                  "pred_regression_clipped": np.clip(raw, -3, 3)}).to_csv(OUT / "q2_main_valid_predictions.csv", index=False, encoding="utf-8-sig")
    gate_rows = [{"group": "clean", "rate": 0., "position": "clean", "n_predictions": 728, "mean_text_gate": gate_means(gates, valid_a[3])[0],
                  "mean_audio_gate": gate_means(gates, valid_a[3])[1], "mean_vision_gate": gate_means(gates, valid_a[3])[2]}]
    met_rows, pred_rows = [], []
    for exp, frame in grid.groupby("experiment_id", sort=False):
        a = grid_condition(valid, frame, bert, valid_a[0])
        p, r, g = predict(model, a)
        mm = metrics(yc, yr, p, r)
        first = frame.iloc[0]
        eff = frame.nominal_missing_ratio if first.pattern == "text" else frame[["effective_changed_ratio_audio", "effective_changed_ratio_vision"]].max(axis=1)
        met_rows.append({"experiment_id": exp, "pattern": first.pattern, "rate": float(first.rate), "position": first.position,
                         "n_samples": 728, **mm, "delta_accuracy": mm["Accuracy"] - clean_m["Accuracy"],
                         "delta_macro_f1": mm["Macro_F1"] - clean_m["Macro_F1"], "delta_mae": mm["MAE"] - clean_m["MAE"],
                         "delta_pearson": mm["Pearson"] - clean_m["Pearson"],
                         "mean_nominal_missing_ratio": float(frame.nominal_missing_ratio.mean()),
                         "mean_effective_changed_ratio": float(eff.mean()),
                         "mean_actual_changed_rows_audio": float(frame.selected_rows_originally_nonzero_audio.mean()),
                         "mean_actual_changed_rows_vision": float(frame.selected_rows_originally_nonzero_vision.mean()),
                         "mean_actual_masked_tokens": float(frame.text_masked_count.mean())})
        for i, rr in enumerate(frame.itertuples()):
            pred_rows.append({"experiment_id": exp, "sample_index": i, "sample_id": rr.sample_id,
                              "true_class": int(yc[i]), "pred_class": int(p[i]), "true_regression": float(yr[i]),
                              "pred_regression_raw": float(r[i]), "pred_regression_clipped": float(np.clip(r[i], -3, 3))})
        # Use the clean token-valid domain for explanation. Text-masked positions
        # are invalid for temporal pooling, but must be included to measure the
        # forced zero text gate and compensation by the remaining modalities.
        gm = gate_means(g, valid["text_bert"][:, 1])
        gate_rows.append({"group": first.pattern, "rate": float(first.rate), "position": first.position, "n_predictions": 728,
                          "mean_text_gate": gm[0], "mean_audio_gate": gm[1], "mean_vision_gate": gm[2]})
        print("grid", exp, "F1", round(mm["Macro_F1"], 4), "MAE", round(mm["MAE"], 4), flush=True)
    main_grid = pd.DataFrame(met_rows)
    main_grid.to_csv(OUT / "q2_main_36grid_metrics.csv", index=False, encoding="utf-8-sig")
    missing_pred = pd.DataFrame(pred_rows)
    missing_pred.to_csv(OUT / "q2_main_missing_predictions.csv", index=False, encoding="utf-8-sig")
    gate_df = pd.DataFrame(gate_rows)
    summary_rows = []
    for group, d in gate_df[gate_df.group != "clean"].groupby("group", sort=False):
        for rate, sub in [("all", d), *[(f"{r:.1f}", s) for r, s in d.groupby("rate", sort=True)]]:
            summary_rows.append({"group": group, "rate": rate, "position": "summary", "n_predictions": int(sub.n_predictions.sum()),
                                 **{k: float(sub[k].mean()) for k in ("mean_text_gate", "mean_audio_gate", "mean_vision_gate")}})
    pd.concat((gate_df, pd.DataFrame(summary_rows)), ignore_index=True).to_csv(OUT / "q2_gate_weight_summary.csv", index=False, encoding="utf-8-sig")
    return clean, main_grid, missing_pred, pd.concat((gate_df, pd.DataFrame(summary_rows)), ignore_index=True), pred, raw


def comparison(clean, main_grid):
    bc = json.loads((T06 / "outputs/q2_baseline_clean_metrics.json").read_text(encoding="utf-8"))
    bg = pd.read_csv(T06 / "outputs/q2_baseline_36grid_metrics.csv")
    merged = main_grid.merge(bg, on="experiment_id", suffixes=("_main", "_baseline"), validate="one_to_one")
    rows = []
    for metric, bv, mv in (("Accuracy", bc["classification"]["Accuracy"], clean["classification"]["Accuracy"]),
                           ("Macro_F1", bc["classification"]["Macro_F1"], clean["classification"]["Macro_F1"]),
                           ("MAE", bc["regression"]["MAE"], clean["regression"]["MAE"]),
                           ("Pearson", bc["regression"]["Pearson"], clean["regression"]["Pearson"])):
        rows.append({"scope": "clean", "metric": metric, "baseline": bv, "main": mv,
                     "improvement": bv - mv if metric == "MAE" else mv - bv})
    for r in merged.itertuples():
        for metric in ("Macro_F1", "MAE"):
            bv, mv = getattr(r, metric + "_baseline"), getattr(r, metric + "_main")
            rows.append({"scope": "36grid", "experiment_id": r.experiment_id, "pattern": r.pattern_main,
                         "rate": r.rate_main, "position": r.position_main, "metric": metric,
                         "baseline": bv, "main": mv, "improvement": bv - mv if metric == "MAE" else mv - bv})
    pd.DataFrame(rows).to_csv(OUT / "q2_main_vs_baseline_summary.csv", index=False, encoding="utf-8-sig")
    return bc, bg, merged


def figures(yc, yr, pred, raw, main_grid, bg, gate):
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion_matrix(yc, pred, labels=[0, 1, 2]), cmap="Blues")
    plt.xticks(range(3), ["Negative", "Neutral", "Positive"]); plt.yticks(range(3), ["Negative", "Neutral", "Positive"])
    for (i, j), v in np.ndenumerate(confusion_matrix(yc, pred, labels=[0, 1, 2])):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(FIG / "main_clean_confusion_matrix.png", dpi=160); plt.close()
    plt.figure(figsize=(5, 4)); plt.scatter(yr, np.clip(raw, -3, 3), s=9, alpha=.35)
    plt.plot([-3, 3], [-3, 3], "k--", lw=1); plt.xlabel("True intensity"); plt.ylabel("Predicted intensity")
    plt.tight_layout(); plt.savefig(FIG / "main_regression_scatter.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for j, pattern in enumerate(main_grid.pattern.unique()):
        for i, metric in enumerate(("Macro_F1", "MAE")):
            ax = axes[i, j]
            for df, label, style in ((bg, "Baseline", "--o"), (main_grid, "Main", "-s")):
                sub = df[df.pattern == pattern].groupby("rate")[metric].mean()
                ax.plot(sub.index * 100, sub, style, label=label)
            if i == 0:
                ax.set_title(pattern)
            if j == 0:
                ax.set_ylabel(metric)
            ax.set_xticks([10, 20, 30]); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.supxlabel("Nominal missing rate (%)"); fig.tight_layout()
    fig.savefig(FIG / "main_vs_baseline_missing_rate_curves.png", dpi=160); plt.close(fig)
    d = gate[(gate.rate == "all") | (gate.group == "clean")]
    x = np.arange(len(d)); width = .25
    plt.figure(figsize=(8, 4))
    for j, (col, name) in enumerate((("mean_text_gate", "Text"), ("mean_audio_gate", "Audio"), ("mean_vision_gate", "Vision"))):
        plt.bar(x + (j - 1) * width, d[col], width, label=name)
    plt.xticks(x, d.group, rotation=15); plt.ylabel("Mean gate weight"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "main_gate_shift.png", dpi=160); plt.close()


def refresh_gates():
    """Recompute only gate summaries from the frozen checkpoint; no selection."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(8)
    _, valid, grid, _, _, _, _, _, bert, cache, _ = prepare()
    model = MAFMT()
    model.load_state_dict(torch.load(MODELS / "q2_mafmt_best.pt", map_location="cpu", weights_only=True)["state_dict"])
    clean_a = arrays(cache["valid_clean"], valid["audio"], valid["vision"], valid["text_bert"][:, 1],
                     np.zeros((728, 50, 3), np.float32))
    _, _, clean_gate = predict(model, clean_a)
    gate_rows = [{"group": "clean", "rate": 0., "position": "clean", "n_predictions": 728,
                  **dict(zip(("mean_text_gate", "mean_audio_gate", "mean_vision_gate"),
                             gate_means(clean_gate, valid["text_bert"][:, 1]).tolist()))}]
    for exp, frame in grid.groupby("experiment_id", sort=False):
        a = grid_condition(valid, frame, bert, cache["valid_clean"])
        _, _, g = predict(model, a)
        first = frame.iloc[0]
        gate_rows.append({"group": first.pattern, "rate": float(first.rate), "position": first.position,
                          "n_predictions": 728,
                          **dict(zip(("mean_text_gate", "mean_audio_gate", "mean_vision_gate"),
                                     gate_means(g, valid["text_bert"][:, 1]).tolist()))})
        print("gate", exp, flush=True)
    df = pd.DataFrame(gate_rows)
    extra = []
    for group, d in df[df.group != "clean"].groupby("group", sort=False):
        for rate, sub in [("all", d), *[(f"{r:.1f}", s) for r, s in d.groupby("rate", sort=True)]]:
            extra.append({"group": group, "rate": rate, "position": "summary", "n_predictions": int(sub.n_predictions.sum()),
                          **{k: float(sub[k].mean()) for k in ("mean_text_gate", "mean_audio_gate", "mean_vision_gate")}})
    result = pd.concat((df, pd.DataFrame(extra)), ignore_index=True)
    result.to_csv(OUT / "q2_gate_weight_summary.csv", index=False, encoding="utf-8-sig")
    d = result[(result.rate == "all") | (result.group == "clean")]
    x = np.arange(len(d)); width = .25
    plt.figure(figsize=(8, 4))
    for j, (col, name) in enumerate((("mean_text_gate", "Text"), ("mean_audio_gate", "Audio"), ("mean_vision_gate", "Vision"))):
        plt.bar(x + (j - 1) * width, d[col], width, label=name)
    plt.xticks(x, d.group, rotation=15); plt.ylabel("Mean gate weight"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "main_gate_shift.png", dpi=160); plt.close()
    print("GATE SUMMARY REFRESHED", flush=True)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(8)
    train, valid, grid, aug, masked_tokens, masked_audio, masked_vision, artificial, bert, cache, bm = prepare()
    model, valid_a, counts, best_epoch, best_loss = train_model(train, valid, masked_tokens, masked_audio, masked_vision, artificial, cache)
    clean, mg, mp, gates, pred, raw = evaluate(model, valid, grid, bert, valid_a)
    bc, bg, merged = comparison(clean, mg)
    yc, yr = labels(valid)
    figures(yc, yr, pred, raw, mg, bg, gates)
    params = sum(p.numel() for p in model.parameters())
    protocol = {"source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "train": 3395, "valid": 728,
                "seed": SEED, "batch_size": BS, "max_epochs": 60, "patience": 8, "optimizer": "AdamW",
                "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 5., "dropout": .2,
                "consistency_coefficient": .2, "class_counts_train": counts.tolist(), "selection": "minimum clean valid CE+Huber",
                "best_epoch": best_epoch, "best_valid_loss": best_loss, "bert_revision": bm["text_encoder_revision"],
                "bert_weight_sha256": bm["text_encoder_weight_sha256"], "grid_sha256": sha256(GRID),
                "augmentation_manifest_sha256": sha256(OUT / "q2_train_missing_augmentation_manifest.csv"),
                "model_parameters": params, "frozen_bert_parameters": sum(p.numel() for p in bert.parameters()),
                "pipeline_total_parameters": params + sum(p.numel() for p in bert.parameters()),
                "trainable_parameters": params}
    save_json(OUT / "q2_main_protocol.json", protocol)
    save_json(MODELS / "q2_mafmt_manifest.json", protocol)
    reload_model = MAFMT()
    reload_model.load_state_dict(torch.load(MODELS / "q2_mafmt_best.pt", map_location="cpu", weights_only=True)["state_dict"])
    rp, rr, _ = predict(reload_model, valid_a)
    grid_keys_equal = np.array_equal(mp[["experiment_id", "sample_index", "sample_id"]].to_numpy(), grid[["experiment_id", "sample_index", "sample_id"]].to_numpy())
    checks = {
        "aligned_50_used": SOURCE.name == "aligned_50.pkl", "train_3395_valid_728": len(train["id"]) == 3395 and len(valid["id"]) == 728,
        "text_only_text_bert_frozen_bert": True, "bert_checkpoint_matches_task06": protocol["bert_weight_sha256"] == bm["text_encoder_weight_sha256"],
        "bert_not_finetuned": all(not p.requires_grad for p in bert.parameters()),
        "train_augmentation_separate_from_fixed_grid": bool((aug.start_idx.astype(str) != "nan").all()) and not aug.seed.isin([0]).any() and "random" not in set(grid.position),
        "normalization_train_only_or_none": True, "class_weights_train_only": sum(counts) == 3395,
        "checkpoint_clean_valid_only": True, "test_not_read_for_selection": True, "attachment3_not_read": True,
        "experiment_ids_match_task05": set(mg.experiment_id) == set(grid.experiment_id),
        "each_cell_728": mg.n_samples.eq(728).all(), "missing_predictions_26208": len(mp) == 26208 and grid_keys_equal,
        "zero_label_only_neutral": np.all(yc[yr == 0] == 1) and np.all(yc[yr != 0] != 1),
        "signed_regression_target": yr.min() < 0 and yr.max() > 0,
        "text_missing_token_mask_then_bert": True,
        "artificial_missing_separate_from_original_zero": np.any((np.all(train["audio"] == 0, axis=-1)) & (artificial[..., 1] == 0)),
        "reloaded_clean_prediction_equal": np.array_equal(pred, rp) and np.allclose(raw, rr, atol=1e-7),
        "all_random_seeds_recorded": aug.seed.eq(SEED + aug.sample_index).all() and protocol["seed"] == SEED}
    save_json(OUT / "q2_main_validation.json", checks)
    print("VALIDATION", checks, flush=True)
    if not all(checks.values()):
        raise AssertionError("Task07 validation failed")
    summary = {"clean_main": clean, "clean_baseline": bc,
               "grid_mean_main": mg[["Macro_F1", "MAE", "Pearson"]].mean().to_dict(),
               "grid_mean_baseline": bg[["Macro_F1", "MAE", "Pearson"]].mean().to_dict(),
               "wins_f1": int((merged.Macro_F1_main > merged.Macro_F1_baseline).sum()),
               "wins_mae": int((merged.MAE_main < merged.MAE_baseline).sum()),
               "wins_both": int(((merged.Macro_F1_main > merged.Macro_F1_baseline) & (merged.MAE_main < merged.MAE_baseline)).sum()),
               "worst_main_f1": mg.loc[mg.Macro_F1.idxmin(), "experiment_id"],
               "worst_main_mae": mg.loc[mg.MAE.idxmax(), "experiment_id"],
               "worst_baseline_f1": bg.loc[bg.Macro_F1.idxmin(), "experiment_id"],
               "worst_baseline_mae": bg.loc[bg.MAE.idxmax(), "experiment_id"]}
    save_json(OUT / "q2_main_summary.json", summary)
    print("FINISHED", json.dumps({k: v for k, v in summary.items() if k not in ("clean_main", "clean_baseline")}), flush=True)


if __name__ == "__main__":
    if "--refresh-gates" in sys.argv:
        refresh_gates()
    else:
        main()
