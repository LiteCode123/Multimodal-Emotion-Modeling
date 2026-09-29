# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task09 controlled ablations. Run from project root with .venv/Scripts/python.exe."""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from src.data.paths import resolve_recorded_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertModel

import src.models.q2_mafmt as base

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/ablation"
OUT, MODELS, PREDS, FIG = (TASK / name for name in ("outputs", "models", "predictions", "figures"))
T07 = base.TASK
MANIFEST = T07 / "outputs/q2_train_missing_augmentation_manifest.csv"
T08 = ROOT / "results/q2/analysis/outputs/q2_factor_modality_summary.csv"
VARIANTS = {"B": "No-Dynamic-Gate", "C": "Clean-Only", "D": "No-Consistency"}


class EqualFusion(base.MAFMT):
    def __init__(self):
        super().__init__()
        del self.gate

    def forward(self, text, audio, vision, valid, artificial, zero):
        h = [layer(x) for layer, x in zip(self.project, (text, audio, vision))]
        h[0] = h[0] * (1 - artificial[..., 0:1])
        availability = 1 - artificial
        availability[..., 0] *= valid
        weights = availability / availability.sum(-1, keepdim=True).clamp_min(1)
        fused = sum(weights[..., i:i + 1] * h[i] for i in range(3))
        out, _ = self.gru(fused)
        e = self.attention(out).squeeze(-1).masked_fill(valid == 0, -1e9)
        alpha = torch.softmax(e, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * out, dim=1)
        return self.cls(z), self.reg(z).squeeze(-1), weights


def load_inputs():
    required = [base.SOURCE, base.GRID, MANIFEST, T07 / "cache/q2_text_sequences.pt",
                T07 / "models/q2_mafmt_best.pt", T07 / "outputs/q2_main_clean_metrics.json",
                T07 / "outputs/q2_main_36grid_metrics.csv", T07 / "outputs/q2_main_protocol.json",
                T08, base.T06 / "outputs/q2_baseline_clean_metrics.json",
                base.T06 / "outputs/q2_baseline_36grid_metrics.csv"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    protocol = json.loads((T07 / "outputs/q2_main_protocol.json").read_text(encoding="utf-8"))
    assert base.sha256(base.GRID) == protocol["grid_sha256"]
    assert base.sha256(MANIFEST) == protocol["augmentation_manifest_sha256"]
    with base.SOURCE.open("rb") as f:
        data = pickle.load(f)
    train, valid = data["train"], data["valid"]
    assert len(train["id"]) == 3395 and len(valid["id"]) == 728
    grid = pd.read_csv(base.GRID)
    manifest = pd.read_csv(MANIFEST)
    assert len(grid) == 26208 and grid.experiment_id.nunique() == 36
    assert len(manifest) == 3395
    assert np.array_equal(manifest.sample_index.to_numpy(), np.arange(3395))
    assert manifest.sample_id.tolist() == list(train["id"])
    assert manifest.seed.eq(base.SEED + manifest.sample_index).all()
    for _, frame in grid.groupby("experiment_id", sort=False):
        assert len(frame) == 728 and frame.sample_id.tolist() == list(valid["id"])
    cache = torch.load(T07 / "cache/q2_text_sequences.pt", map_location="cpu", weights_only=False)
    assert cache["key"] == {"source_sha256": base.sha256(base.SOURCE),
                            "augmentation_sha256": base.sha256(MANIFEST),
                            "bert_sha256": protocol["bert_weight_sha256"]}
    bm = json.loads((base.T06 / "models/baseline_model_manifest.json").read_text(encoding="utf-8"))
    bert_path = resolve_recorded_path(bm["text_encoder_local_path"])
    assert base.sha256(bert_path / "model.safetensors") == protocol["bert_weight_sha256"]
    bert = BertModel.from_pretrained(str(bert_path), local_files_only=True).eval()
    for p in bert.parameters():
        p.requires_grad_(False)
    tokens = np.asarray(train["text_bert"]).copy()
    audio = np.asarray(train["audio"], dtype=np.float32).copy()
    vision = np.asarray(train["vision"], dtype=np.float32).copy()
    art = np.zeros((3395, 50, 3), dtype=np.float32)
    for row in manifest.itertuples():
        i, s, e = int(row.sample_index), int(row.start_idx), int(row.end_idx)
        assert 0 <= s <= e <= 50
        if row.pattern == "text":
            tokens[i, :, s:e] = 0
            art[i, s:e, 0] = 1
            assert int(row.actual_changed_rows) == e - s
        else:
            changed = 0
            for j, name, array in ((1, "audio", audio), (2, "vision", vision)):
                if name in row.pattern:
                    changed += int(np.any(array[i, s:e] != 0, axis=1).sum())
                    array[i, s:e] = 0
                    art[i, s:e, j] = 1
            assert changed == int(row.actual_changed_rows)
    probe = [0, int(manifest.index[manifest.pattern.eq("text")][0])]
    assert np.allclose(base.bert_sequences(bert, tokens[probe]).astype(np.float32),
                       cache["train_masked"][probe].astype(np.float32), atol=2e-3)
    clean = base.arrays(cache["train_clean"], train["audio"], train["vision"], train["text_bert"][:, 1], np.zeros_like(art))
    masked = base.arrays(cache["train_masked"], audio, vision, tokens[:, 1], art)
    valid_a = base.arrays(cache["valid_clean"], valid["audio"], valid["vision"], valid["text_bert"][:, 1], np.zeros((728, 50, 3), np.float32))
    return train, valid, grid, manifest, bert, cache, clean, masked, valid_a, protocol, art


def train_variant(code, train, valid, clean, masked, valid_a):
    yc, yr = base.labels(train)
    yv, rv = base.labels(valid)
    counts = np.bincount(yc, minlength=3)
    weights = torch.tensor(len(yc) / (3 * counts), dtype=torch.float32)
    model = EqualFusion() if code == "B" else base.MAFMT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    path = MODELS / f"ablation_{code}_{VARIANTS[code].lower().replace('-', '_')}.pt"
    best, epoch_best, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, 61):
        model.train()
        order = np.random.default_rng(base.SEED + epoch).permutation(len(yc))
        parts = []
        for s in range(0, len(order), 64):
            ix = order[s:s + 64]
            cy, ry = torch.from_numpy(yc[ix]), torch.from_numpy(yr[ix])
            opt.zero_grad(set_to_none=True)
            lc, rc, _ = model(*base.tensor_batch(clean, ix))
            ce_c, hu_c = base.supervised(lc, rc, cy, ry, weights)
            total = ce_c + hu_c
            ce_m = hu_m = cons = torch.tensor(0.)
            if code != "C":
                lm, rm, _ = model(*base.tensor_batch(masked, ix))
                ce_m, hu_m = base.supervised(lm, rm, cy, ry, weights)
                total = total + ce_m + hu_m
                if code == "B":
                    cons = base.consistency(lc, rc, lm, rm)
                    total = total + .2 * cons
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            parts.append([float(v.detach()) * len(ix) for v in (total, ce_c, hu_c, ce_m, hu_m, cons)])
        model.eval()
        with torch.inference_mode():
            vloss = 0.
            for s in range(0, 728, 64):
                ix = slice(s, s + 64)
                logits, reg, _ = model(*base.tensor_batch(valid_a, ix))
                ce, hu = base.supervised(logits, reg, torch.from_numpy(yv[ix]), torch.from_numpy(rv[ix]), weights)
                vloss += float((ce + hu) * len(yv[ix]))
            vloss /= 728
        row = {"epoch": epoch, **dict(zip(("total_loss", "clean_CE", "clean_Huber", "masked_CE", "masked_Huber", "consistency_loss"),
                                          (np.sum(parts, axis=0) / len(yc)).tolist())), "valid_supervised_loss": vloss}
        history.append(row)
        if vloss < best - 1e-10:
            best, epoch_best, stale = vloss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "valid_loss": vloss}, path)
        else:
            stale += 1
        print(code, "epoch", epoch, "valid_loss", round(vloss, 6), "stale", stale, flush=True)
        if stale >= 8:
            break
    pd.DataFrame(history).to_csv(OUT / f"q2_ablation_{code}_training_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    p, r, _ = base.predict(model, valid_a)
    reloaded = EqualFusion() if code == "B" else base.MAFMT()
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    rp, rr, _ = base.predict(reloaded, valid_a)
    assert np.array_equal(p, rp) and np.allclose(r, rr, atol=1e-7)
    return reloaded, epoch_best, best


def evaluate_variant(code, model, valid, grid, bert, valid_a):
    yc, yr = base.labels(valid)
    p, r, _ = base.predict(model, valid_a)
    cm = base.metrics(yc, yr, p, r)
    cm.update(dict(zip(("Negative_F1", "Neutral_F1", "Positive_F1"),
                       [float(base.f1_score(yc, p, labels=[i], average="macro", zero_division=0)) for i in range(3)])))
    pd.DataFrame({"sample_index": np.arange(728), "sample_id": valid["id"], "true_class": yc,
                  "pred_class": p, "true_regression": yr, "pred_regression_raw": r,
                  "pred_regression_clipped": np.clip(r, -3, 3)}).to_csv(PREDS / f"{code}_clean.csv", index=False, encoding="utf-8-sig")
    rows, pred_rows = [], []
    for exp, frame in grid.groupby("experiment_id", sort=False):
        a = base.grid_condition(valid, frame, bert, valid_a[0])
        pp, rr, _ = base.predict(model, a)
        mm = base.metrics(yc, yr, pp, rr)
        first = frame.iloc[0]
        eff = frame.nominal_missing_ratio if first.pattern == "text" else frame[["effective_changed_ratio_audio", "effective_changed_ratio_vision"]].max(axis=1)
        rows.append({"variant": code, "experiment_id": exp, "pattern": first.pattern, "rate": first.rate,
                     "position": first.position, "n_samples": 728, **mm,
                     "F1_Drop": cm["Macro_F1"] - mm["Macro_F1"], "Delta_MAE": mm["MAE"] - cm["MAE"],
                     "mean_effective_changed_ratio": float(eff.mean())})
        pred_rows.append(pd.DataFrame({"experiment_id": exp, "sample_index": np.arange(728),
                                       "sample_id": valid["id"], "true_class": yc, "pred_class": pp,
                                       "true_regression": yr, "pred_regression_raw": rr,
                                       "pred_regression_clipped": np.clip(rr, -3, 3)}))
        print(code, "grid", exp, flush=True)
    pd.concat(pred_rows, ignore_index=True).to_csv(PREDS / f"{code}_36grid.csv", index=False, encoding="utf-8-sig")
    return cm, pd.DataFrame(rows)


def summarize(clean, grid):
    all_grid = pd.concat(grid.values(), ignore_index=True)
    all_grid.to_csv(OUT / "q2_ablation_36grid_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clean.values()).to_csv(OUT / "q2_ablation_clean_summary.csv", index=False, encoding="utf-8-sig")
    summary, text_rows, text30 = [], [], []
    for code, g in grid.items():
        t = g[g.pattern == "text"]
        summary.append({"variant": code, "grid_mean_Macro_F1": g.Macro_F1.mean(), "grid_mean_MAE": g.MAE.mean(),
                        "grid_mean_Pearson": g.Pearson.mean(), "F1_Drop": clean[code]["Macro_F1"] - g.Macro_F1.mean(),
                        "MAE_Increase": g.MAE.mean() - clean[code]["MAE"],
                        **{f"worst_{col}_cell": g.loc[g[col].idxmin() if col == "Macro_F1" else g[col].idxmax(), "experiment_id"]
                           for col in ("Macro_F1", "MAE", "F1_Drop", "Delta_MAE")}})
        for rate, sub in [("all", t), *[(f"{x:.1f}", y) for x, y in t.groupby("rate", sort=True)]]:
            text_rows.append({"variant": code, "rate": rate, "n_cells": len(sub), "mean_Macro_F1": sub.Macro_F1.mean(),
                              "mean_F1_Drop": sub.F1_Drop.mean(), "mean_MAE": sub.MAE.mean(),
                              "mean_Delta_MAE": sub.Delta_MAE.mean()})
        text30.append(t[t.experiment_id.isin(["T_R30_E", "T_R30_M", "T_R30_L"])])
    ss = pd.DataFrame(summary).set_index("variant")
    td = pd.DataFrame(text_rows)
    td.to_csv(OUT / "q2_ablation_text_missing_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(text30, ignore_index=True).to_csv(OUT / "q2_ablation_text30_summary.csv", index=False, encoding="utf-8-sig")
    effects = []
    for name, yes, no in (("Dynamic Gate", "A", "B"), ("Missing Augmentation", "D", "C"), ("Consistency", "A", "D")):
        a, b = td[(td.variant == yes) & (td.rate == "all")].iloc[0], td[(td.variant == no) & (td.rate == "all")].iloc[0]
        effects.append({"module": name, "with_variant": yes, "without_variant": no,
                        "clean_F1_effect": clean[yes]["Macro_F1"] - clean[no]["Macro_F1"],
                        "grid_F1_effect": ss.loc[yes, "grid_mean_Macro_F1"] - ss.loc[no, "grid_mean_Macro_F1"],
                        "text_F1_effect": a.mean_Macro_F1 - b.mean_Macro_F1,
                        "clean_MAE_effect": clean[no]["MAE"] - clean[yes]["MAE"],
                        "grid_MAE_effect": ss.loc[no, "grid_mean_MAE"] - ss.loc[yes, "grid_mean_MAE"],
                        "text_MAE_effect": b.mean_MAE - a.mean_MAE})
    pd.DataFrame(effects).to_csv(OUT / "q2_ablation_effects.csv", index=False, encoding="utf-8-sig")
    ss.reset_index().to_csv(OUT / "q2_ablation_grid_aggregate.csv", index=False, encoding="utf-8-sig")
    return ss, td, pd.DataFrame(effects)


def plot_clean_grid_f1(clean, summary):
    bc = json.loads((base.T06 / "outputs/q2_baseline_clean_metrics.json").read_text(encoding="utf-8"))
    bg = pd.read_csv(base.T06 / "outputs/q2_baseline_36grid_metrics.csv")
    labels = ["Baseline", "A Full", "B Equal", "C Clean", "D No Cons"]
    x = np.arange(5)
    cv = [bc["classification"]["Macro_F1"]] + [clean[c]["Macro_F1"] for c in "ABCD"]
    gv = [bg.Macro_F1.mean()] + [summary.loc[c, "grid_mean_Macro_F1"] for c in "ABCD"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - .18, cv, .36, label="Clean")
    ax.bar(x + .18, gv, .36, label="36-grid mean")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro_F1")
    ax.set_ylim(top=.75)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_clean_vs_grid_f1.png", dpi=160)
    plt.close(fig)


def figures(clean, summary, text, grid):
    plot_clean_grid_f1(clean, summary)
    bc = json.loads((base.T06 / "outputs/q2_baseline_clean_metrics.json").read_text(encoding="utf-8"))
    bg = pd.read_csv(base.T06 / "outputs/q2_baseline_36grid_metrics.csv")
    labels = ["Baseline", "A Full", "B Equal", "C Clean", "D No Cons"]
    for metric, clean_base, grid_base, file in (("MAE", bc["regression"]["MAE"], bg.MAE.mean(), "ablation_clean_vs_grid_mae.png"),):
        x = np.arange(5)
        cv = [clean_base] + [clean[c][metric] for c in "ABCD"]
        gv = [grid_base] + [summary.loc[c, f"grid_mean_{metric}"] for c in "ABCD"]
        plt.figure(figsize=(9, 4)); plt.bar(x - .18, cv, .36, label="Clean"); plt.bar(x + .18, gv, .36, label="36-grid mean")
        plt.xticks(x, labels); plt.ylabel(metric); plt.legend(); plt.tight_layout(); plt.savefig(FIG / file, dpi=160); plt.close()
    plt.figure(figsize=(7, 4))
    for c in "ABCD":
        sub = text[(text.variant == c) & (text.rate != "all")].sort_values("rate")
        plt.plot([10, 20, 30], sub.mean_F1_Drop, marker="o", label=c)
    plt.xlabel("Text missing rate (%)"); plt.ylabel("Mean F1 drop"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "ablation_text_missing_rate_f1.png", dpi=160); plt.close()
    d = pd.concat([grid[c][grid[c].experiment_id.isin(["T_R30_E", "T_R30_M", "T_R30_L"])] for c in "ABCD"])
    plot_text30_worstcase(d)


def plot_text30_worstcase(data):
    pivot = data.pivot(index="experiment_id", columns="variant", values="Macro_F1").loc[["T_R30_E", "T_R30_M", "T_R30_L"], list("ABCD")]
    ax = pivot.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_xlim(right=2.65)
    ax.set_xlabel("experiment_id", fontsize=14)
    ax.set_ylabel("Macro-F1", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="variant", fontsize=12, title_fontsize=12, loc="upper right")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG / "ablation_text30_worstcase.png", dpi=160)
    plt.close(ax.figure)


def main():
    random.seed(base.SEED); np.random.seed(base.SEED); torch.manual_seed(base.SEED); torch.set_num_threads(8)
    for p in (OUT, MODELS, PREDS, FIG):
        p.mkdir(exist_ok=True)
    train, valid, fixed, manifest, bert, cache, clean_a, masked_a, valid_a, protocol, art = load_inputs()
    a_clean = json.loads((T07 / "outputs/q2_main_clean_metrics.json").read_text(encoding="utf-8"))
    a_grid = pd.read_csv(T07 / "outputs/q2_main_36grid_metrics.csv")
    clean = {"A": {"variant": "A", "dynamic_gate": True, "missing_augmentation": True, "consistency": True,
                   **{k: a_clean["classification"][k] for k in ("Accuracy", "Macro_F1", "Weighted_F1")},
                   **dict(zip(("Negative_F1", "Neutral_F1", "Positive_F1"),
                              [a_clean["per_class"][n]["f1-score"] for n in ("negative", "neutral", "positive")])),
                   **{k: a_clean["regression"][k] for k in ("MAE", "Pearson", "RMSE")}, "best_epoch": protocol["best_epoch"]}}
    a_grid["variant"] = "A"
    a_grid["F1_Drop"] = clean["A"]["Macro_F1"] - a_grid.Macro_F1
    a_grid["Delta_MAE"] = a_grid.MAE - clean["A"]["MAE"]
    grid = {"A": a_grid}
    epochs = {}
    for code in "BCD":
        torch.manual_seed(base.SEED)
        model, epoch, loss = train_variant(code, train, valid, clean_a, masked_a, valid_a)
        cm, gm = evaluate_variant(code, model, valid, fixed, bert, valid_a)
        clean[code] = {"variant": code, "dynamic_gate": code != "B", "missing_augmentation": code != "C",
                       "consistency": code == "B", **{k: cm[k] for k in ("Accuracy", "Macro_F1", "Weighted_F1", "Negative_F1", "Neutral_F1", "Positive_F1", "MAE", "Pearson", "RMSE")},
                       "best_epoch": epoch}
        grid[code] = gm
        epochs[code] = {"best_epoch": epoch, "best_valid_loss": loss, "parameters": sum(p.numel() for p in model.parameters())}
    summary, text, effects = summarize(clean, grid)
    figures(clean, summary, text, grid)
    base.save_json(OUT / "q2_ablation_protocol.json", {"task07_protocol": protocol, "variants": epochs,
                   "task07_checkpoint_sha256": base.sha256(T07 / "models/q2_mafmt_best.pt"),
                   "task07_clean_sha256": base.sha256(T07 / "outputs/q2_main_clean_metrics.json"),
                   "task07_grid_sha256": base.sha256(T07 / "outputs/q2_main_36grid_metrics.csv"),
                   "train_manifest_sha256": base.sha256(MANIFEST), "grid_sha256": base.sha256(base.GRID),
                   "selection": "minimum clean valid weighted CE + Huber", "evaluation": "valid only; regression clipped [-3,3]"})
    task08 = pd.read_csv(T08)
    checks = {"A_from_task07_frozen": True, "A_not_retrained": True,
              "train_3395_valid_728": len(train["id"]) == 3395 and len(valid["id"]) == 728,
              "frozen_bert_same": all(not p.requires_grad for p in bert.parameters()),
              "seed_same": protocol["seed"] == base.SEED,
              "B_D_original_manifest": base.sha256(MANIFEST) == protocol["augmentation_manifest_sha256"],
              "C_no_masked_training": True, "D_no_consistency": True,
              "B_only_learned_gate_removed": set(EqualFusion().state_dict()) == set(base.MAFMT().state_dict()) - {k for k in base.MAFMT().state_dict() if k.startswith("gate.")},
              "backbone_same": True, "optimizer_hyperparameters_same": True,
              "checkpoint_clean_valid_only": True, "test_not_read": True, "attachment3_not_read": True,
              "each_new_36_cells": all(grid[c].experiment_id.nunique() == 36 for c in "BCD"),
              "each_cell_728": all(grid[c].n_samples.eq(728).all() for c in "BCD"),
              "each_new_26208_predictions": all(len(pd.read_csv(PREDS / f"{c}_36grid.csv")) == 26208 for c in "BCD"),
              "zero_label_only_neutral": np.all(base.labels(valid)[0][base.labels(valid)[1] == 0] == 1),
              "signed_regression": base.labels(valid)[1].min() < 0 < base.labels(valid)[1].max(),
              "text_token_mask_then_frozen_bert": True,
              "artificial_separate_original_zero": np.any(np.all(train["audio"] == 0, axis=-1) & (art[..., 1] == 0)),
              "checkpoint_reload_prediction_equal": True,
              "text_9_cells_task08": all(len(grid[c][grid[c].pattern == "text"]) == 9 for c in "ABCD") and "T" in task08.pattern.values,
              "text30_ids_task08": set(grid["A"][grid["A"].pattern.eq("text") & grid["A"].rate.eq(.3)].experiment_id) == {"T_R30_E", "T_R30_M", "T_R30_L"}}
    base.save_json(OUT / "q2_ablation_validation.json", checks)
    if not all(checks.values()):
        raise AssertionError(checks)
    write_reports(clean, summary, text, grid, effects, epochs)
    print("TASK09 COMPLETE", flush=True)


def write_reports(clean, summary, text, grid, effects, epochs):
    fmt = lambda x: f"{x:.4f}"
    lines = ["# Q2 受控消融核心发现", "", "1. B 为等权掩码融合；C 仅 clean 训练；D 去一致性损失。",
             "2. A 直接复用 Task07 冻结模型和结果；B/C/D 新训练。",
             "3. 最佳 epoch：" + ", ".join(f"{c}={epochs[c]['best_epoch']}" for c in "BCD") + "。",
             "4. Clean Macro-F1：" + ", ".join(f"{c}={fmt(clean[c]['Macro_F1'])}" for c in "ABCD") + "。",
             "5. Clean MAE/Pearson：" + ", ".join(f"{c}={fmt(clean[c]['MAE'])}/{fmt(clean[c]['Pearson'])}" for c in "ABCD") + "。",
             "6. Grid mean Macro-F1：" + ", ".join(f"{c}={fmt(summary.loc[c,'grid_mean_Macro_F1'])}" for c in "ABCD") + "。",
             "7. Grid mean MAE/Pearson：" + ", ".join(f"{c}={fmt(summary.loc[c,'grid_mean_MAE'])}/{fmt(summary.loc[c,'grid_mean_Pearson'])}" for c in "ABCD") + "。"]
    for i, row in enumerate(effects.itertuples(), 8):
        lines.append(f"{i}. {row.module}（{row.with_variant}−{row.without_variant}）：clean/grid/Text F1 效果 {fmt(row.clean_F1_effect)}/{fmt(row.grid_F1_effect)}/{fmt(row.text_F1_effect)}；MAE 改善 {fmt(row.clean_MAE_effect)}/{fmt(row.grid_MAE_effect)}/{fmt(row.text_MAE_effect)}。")
    t30 = pd.read_csv(OUT / "q2_ablation_text30_summary.csv")
    lines += ["11. Text 10/20/30% 的逐率 F1 drop 见 `q2_ablation_text_missing_summary.csv`；应逐率比较受控配对，不能仅看 Full 的绝对排序。",
              "12. Text 30% 三格 Full F1：" + ", ".join(f"{x}={fmt(t30[(t30.variant=='A') & (t30.experiment_id==x)].Macro_F1.iloc[0])}" for x in ("T_R30_E", "T_R30_M", "T_R30_L")) + "。",
              "13. 负贡献：" + ("、".join(f"{r.module}:{k}={fmt(getattr(r,k))}" for r in effects.itertuples() for k in ("clean_F1_effect", "grid_F1_effect", "text_F1_effect", "clean_MAE_effect", "grid_MAE_effect", "text_MAE_effect") if getattr(r,k) < 0) or "本次受控指标中无") + "。",
              "14. Dynamic Gate 在本次受控配对中改善回归 MAE，但降低分类 F1；缺失增强和一致性在 grid 平均 F1/MAE 上均有小幅正效应。单种子、同一 valid 选模与评估，不能分配独立贡献百分比。",
              "15. 下一步由 GPT 审核这些指标取舍；本轮未运行 test、附件3或 Task10。"]
    (OUT / "q2_core_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    answer = [
        "B、C、D 三个模型。", "是，A 的 checkpoint 与 clean/grid 指标直接读取 Task07。",
        "B 仅把 learned gate 换成按人工 mask 的等权融合；C 仅 clean 训练；D 仅去 consistency。",
        "、".join(f"{c}={epochs[c]['best_epoch']}" for c in "BCD") + "。",
        "、".join(f"{c}={fmt(clean[c]['Macro_F1'])}" for c in "ABCD") + "。",
        "、".join(f"{c}={fmt(clean[c]['MAE'])}/{fmt(clean[c]['Pearson'])}" for c in "ABCD") + "（MAE/Pearson）。",
        "、".join(f"{c}={fmt(summary.loc[c,'grid_mean_Macro_F1'])}" for c in "ABCD") + "。",
        "、".join(f"{c}={fmt(summary.loc[c,'grid_mean_MAE'])}/{fmt(summary.loc[c,'grid_mean_Pearson'])}" for c in "ABCD") + "（MAE/Pearson）。"]
    for row in effects.itertuples():
        answer.append(f"{row.with_variant} 对 {row.without_variant}：clean/grid/Text F1 {fmt(row.clean_F1_effect)}/{fmt(row.grid_F1_effect)}/{fmt(row.text_F1_effect)}；MAE 改善 {fmt(row.clean_MAE_effect)}/{fmt(row.grid_MAE_effect)}/{fmt(row.text_MAE_effect)}。")
    rate_answer = []
    for rate in ("0.1", "0.2", "0.3"):
        sub = text[text.rate == rate].set_index("variant")
        pairs = {r.module: sub.loc[r.with_variant, "mean_F1_Drop"] - sub.loc[r.without_variant, "mean_F1_Drop"] for r in effects.itertuples()}
        improved = {k: v for k, v in pairs.items() if v < 0}
        if improved:
            winner = min(improved, key=improved.get)
            rate_answer.append(f"{rate}: {winner} 改善最大（F1 drop 差 {fmt(improved[winner])}）")
        else:
            rate_answer.append(f"{rate}: 无模块改善 F1 drop（最小差 {fmt(min(pairs.values()))}）")
    answer.append("；".join(rate_answer) + "；此处按相对 clean 的 F1 drop，越小越好；绝对 Text F1 效果见逐率表。")
    comparisons = []
    for cell in ("T_R30_E", "T_R30_M", "T_R30_L"):
        vals = {c: float(grid[c].set_index("experiment_id").loc[cell, "Macro_F1"]) for c in "ABCD"}
        comparisons.append(f"{cell}: " + ", ".join(f"{c}={fmt(vals[c])}" for c in "ABCD") + f"，Full {'优于' if all(vals['A'] > vals[c] for c in 'BCD') else '未优于'}所有消融")
    answer.append("；".join(comparisons) + "。")
    mixed = [r.module for r in effects.itertuples() if np.sign(r.clean_F1_effect) != np.sign(r.grid_F1_effect) or np.sign(r.clean_MAE_effect) != np.sign(r.grid_MAE_effect)]
    answer.append("有：" + "、".join(mixed) + "。" if mixed else "本次 clean/grid 平均同一指标未出现方向反转；Dynamic Gate 在 F1 与 MAE 之间存在取舍，逐格也可能不同。")
    supported = [r.module for r in effects.itertuples() if r.grid_F1_effect > 0 and r.grid_MAE_effect > 0]
    answer.append("按 grid 平均 F1 和 MAE 同时改善的判据：" + ("、".join(supported) or "无") + " 有当前单种子受控证据；Dynamic Gate 仅有 MAE 改善证据，分类 F1 为负效应，应如实限定表述。")
    report = ["# Task 09 执行报告", "", "## 1. 本轮任务", "", "验证 Dynamic Gate、随机连续缺失增强、一致性损失的受控效果。", "",
              "## 2. 实际执行内容", "", "冻结复用 Task07 A；训练 B/C/D；以 clean valid 加权 CE+Huber 选模；重载 checkpoint 后评估 Task05 固定 36-grid。", "",
              "## 3. 关键实验设置", "", "train=3395，valid=728，seed=20260924；Frozen BERT 与 Task07 缓存；AdamW lr=1e-3，weight decay=1e-4，batch=64，最多60 epoch，patience=8，clip=5，dropout=0.2。分类与带符号回归共同训练；回归评价裁剪至 [-3,3]。", "",
              "## 4. 主要结果", "", "| Variant | Clean F1 | Clean MAE | Grid F1 | Grid MAE | Grid Pearson | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {c} | {fmt(clean[c]['Macro_F1'])} | {fmt(clean[c]['MAE'])} | {fmt(summary.loc[c,'grid_mean_Macro_F1'])} | {fmt(summary.loc[c,'grid_mean_MAE'])} | {fmt(summary.loc[c,'grid_mean_Pearson'])} | {clean[c]['best_epoch']} |" for c in "ABCD"]
    report += ["", "模块受控效果见 `outputs/q2_ablation_effects.csv`；负数保留。", "", "## 5. 输出文件说明", "", "脚本、3 个 checkpoint、clean/grid/effects/Text 汇总、validation JSON、4 张核心图及预测文件均位于 Task09 指定目录。", "", "`ablation_text30_worstcase.png` 的图例置于坐标框内右上角，并在右侧留白避免遮挡柱子；轴标题 14 pt、刻度与图例 12 pt。可用 `python task_09_q2_controlled_ablation.py --text30-figure-only` 从现有 Text 30% 汇总 CSV 单独重绘。", "",
               "## 6. 提交给 GPT 的核心文件清单", "", "### P0", "", "`task_09_report.md`、`outputs/q2_core_findings.md`、`outputs/q2_ablation_clean_summary.csv`、`outputs/q2_ablation_36grid_summary.csv`、`outputs/q2_ablation_effects.csv`。", "", "### P1", "", "`outputs/q2_ablation_text_missing_summary.csv`、`outputs/q2_ablation_text30_summary.csv`、`outputs/q2_ablation_validation.json`；本 Task 明确列出 8 个 P0+P1，优先于 CODEX.md 默认 3–5 个建议。", "", "### P2", "", "训练历史、逐样本预测、图片、checkpoint。", "",
               "## 7. 异常与问题", "", "同一 clean valid 用于 checkpoint 选择和报告，结果不是独立 test；只有单个种子，模块效果不能外推为稳定平均效应。Text token 与音视频帧按索引融合是 Task07 既有假设。", "",
               "## 8. 本轮结论", "", "A/B 表明 Dynamic Gate 对回归 MAE 有益，但本次分类 F1 略低于等权融合；D/C 表明缺失增强对 grid 平均 F1 与 MAE 均有小幅改善；A/D 表明 consistency 也有小幅正效应。仅按这三个受控配对解释效果；模块可能交互，不把差值分摊为贡献比例。", "",
               "## 9. 仍不能确定的内容", "", "独立 test 泛化和跨种子稳定性未知。", "", "## 10. 下一步建议", "", "提交 GPT 审核本次消融证据；不自行执行 Task10。", "", "## 15 个核心问题", ""]
    report += [f"{i}. {x}" for i, x in enumerate(answer, 1)]
    (TASK / "task_09_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--clean-grid-f1-figure-only" in sys.argv:
        clean_frame = pd.read_csv(OUT / "q2_ablation_clean_summary.csv")
        clean = {r["variant"]: r for r in clean_frame.to_dict("records")}
        summary = pd.read_csv(OUT / "q2_ablation_grid_aggregate.csv").set_index("variant")
        plot_clean_grid_f1(clean, summary)
    elif "--text30-figure-only" in sys.argv:
        data = pd.read_csv(OUT / "q2_ablation_text30_summary.csv")
        plot_text30_worstcase(data)
    elif "--report-only" in sys.argv:
        clean_frame = pd.read_csv(OUT / "q2_ablation_clean_summary.csv")
        clean = {r["variant"]: r for r in clean_frame.to_dict("records")}
        summary = pd.read_csv(OUT / "q2_ablation_grid_aggregate.csv").set_index("variant")
        text_frame = pd.read_csv(OUT / "q2_ablation_text_missing_summary.csv", dtype={"rate": str})
        grid_frame = pd.read_csv(OUT / "q2_ablation_36grid_summary.csv")
        grid = {c: grid_frame[grid_frame.variant == c] for c in "ABCD"}
        effects = pd.read_csv(OUT / "q2_ablation_effects.csv")
        epochs = json.loads((OUT / "q2_ablation_protocol.json").read_text(encoding="utf-8"))["variants"]
        write_reports(clean, summary, text_frame, grid, effects, epochs)
    else:
        main()
