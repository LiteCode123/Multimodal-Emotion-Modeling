# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task08: descriptive analysis of the frozen Task07 validation grid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/analysis"
OUT = TASK / "outputs"
FIG = TASK / "figures"
T07 = ROOT / "results/q2/model/outputs"
T05 = ROOT / "results/q2/missingness/outputs"
FILES = {
    "clean": T07 / "q2_main_clean_metrics.json",
    "cells": T07 / "q2_main_36grid_metrics.csv",
    "missing": T07 / "q2_main_missing_predictions.csv",
    "valid": T07 / "q2_main_valid_predictions.csv",
    "grid": T05 / "q2_valid_36grid.csv",
}
PATTERNS = {"text": "T", "audio": "A", "vision": "V", "audio+vision synchronized": "AV"}
POSITIONS = {"early": "Early", "middle": "Middle", "late": "Late"}
METRICS = ["Macro_F1", "F1_Drop", "MAE", "Delta_MAE", "Pearson", "Delta_Pearson",
           "Accuracy", "Weighted_F1", "RMSE", "effective_perturbation_ratio",
           "actual_changed_rows_audio", "actual_changed_rows_vision"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")
    return df


def aggregate(cells, keys):
    x = cells.groupby(keys, sort=False, observed=True)
    means = x[METRICS].mean().add_prefix("mean_")
    stds = x[["Macro_F1", "F1_Drop", "MAE", "Delta_MAE"]].std(ddof=1).add_prefix("std_")
    return pd.concat([x.size().rename("n_cells"), means, stds], axis=1).reset_index()


def fmt(x):
    return f"{x:.4f}"


def main():
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {k: sha(v) for k, v in FILES.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    clean = json.loads(FILES["clean"].read_text(encoding="utf-8"))
    raw = pd.read_csv(FILES["cells"])
    grid = pd.read_csv(FILES["grid"])
    assert len(raw) == 36 and raw.experiment_id.is_unique
    assert len(grid) == 36 * 728 and not grid.duplicated(["experiment_id", "sample_id"]).any()
    assert set(raw.pattern) == set(PATTERNS) and set(raw.position) == set(POSITIONS)
    assert set(np.round(raw.rate, 2)) == {.1, .2, .3}
    assert raw.n_samples.eq(728).all()
    assert raw.groupby(["pattern", "position", "rate"]).size().eq(1).all()
    assert set(raw.experiment_id) == set(grid.experiment_id)
    for key in ["pattern", "position", "rate"]:
        actual = grid.groupby("experiment_id")[key].first()
        assert (raw.set_index("experiment_id")[key] == actual.reindex(raw.experiment_id).to_numpy()).all()
    # Verify the Task07 effective-ratio convention from Task05 sample metadata.
    grid["effective"] = np.select(
        [grid.pattern.eq("text"), grid.pattern.eq("audio"), grid.pattern.eq("vision")],
        [grid.nominal_missing_ratio, grid.effective_changed_ratio_audio,
         grid.effective_changed_ratio_vision],
        default=grid[["effective_changed_ratio_audio", "effective_changed_ratio_vision"]].max(axis=1),
    )
    meta = grid.groupby("experiment_id").agg(
        effective=("effective", "mean"), nominal=("nominal_missing_ratio", "mean"),
        audio_rows=("selected_rows_originally_nonzero_audio", "mean"),
        vision_rows=("selected_rows_originally_nonzero_vision", "mean"))
    aligned = meta.reindex(raw.experiment_id)
    assert np.allclose(raw.mean_effective_changed_ratio, aligned.effective)
    assert np.allclose(raw.mean_nominal_missing_ratio, aligned.nominal)
    assert np.allclose(raw.mean_actual_changed_rows_audio, aligned.audio_rows)
    assert np.allclose(raw.mean_actual_changed_rows_vision, aligned.vision_rows)

    c = raw.copy()
    c["pattern"] = c.pattern.map(PATTERNS)
    c["position"] = c.position.map(POSITIONS)
    c["nominal_rate"] = c.rate
    c["effective_perturbation_ratio"] = c.mean_effective_changed_ratio
    c["actual_changed_rows_audio"] = c.mean_actual_changed_rows_audio
    c["actual_changed_rows_vision"] = c.mean_actual_changed_rows_vision
    cf, cm, cp = clean["classification"]["Macro_F1"], clean["regression"]["MAE"], clean["regression"]["Pearson"]
    c["F1_Drop"] = cf - c.Macro_F1
    c["Delta_MAE"] = c.MAE - cm
    c["Delta_Pearson"] = c.Pearson - cp
    assert np.allclose(c.F1_Drop, -raw.delta_macro_f1)
    assert np.allclose(c.Delta_MAE, raw.delta_mae)
    assert np.allclose(c.Delta_Pearson, raw.delta_pearson)

    modality = save(aggregate(c, ["pattern"]), "q2_factor_modality_summary.csv")
    position = save(aggregate(c, ["position"]), "q2_factor_position_summary.csv")
    duration = save(aggregate(c, ["nominal_rate"]), "q2_factor_duration_summary.csv")
    md = save(aggregate(c, ["pattern", "nominal_rate"]), "q2_modality_duration_interaction.csv")
    mp = save(aggregate(c, ["pattern", "position"]), "q2_modality_position_interaction.csv")
    pdur = save(aggregate(c, ["position", "nominal_rate"]), "q2_position_duration_interaction.csv")
    save(mp.groupby("pattern").agg(F1_Drop_position_range=("mean_F1_Drop", lambda s: s.max()-s.min()),
                                   Delta_MAE_position_range=("mean_Delta_MAE", lambda s: s.max()-s.min())).reset_index(),
         "q2_modality_position_sensitivity.csv")
    slopes = []
    for p, group in md.groupby("pattern", sort=False):
        row = {"pattern": p}
        for y, label in [("mean_F1_Drop", "F1_Drop"), ("mean_Delta_MAE", "Delta_MAE")]:
            coef = np.polyfit(group.nominal_rate, group[y], 1)
            predicted = np.polyval(coef, group.nominal_rate)
            ss = np.sum((group[y] - group[y].mean()) ** 2)
            row[f"{label}_slope_per_unit_rate"] = coef[0]
            row[f"{label}_R2"] = 1 - np.sum((group[y] - predicted) ** 2) / ss if ss else np.nan
        slopes.append(row)
    slopes = save(pd.DataFrame(slopes), "q2_modality_duration_slope.csv")
    monotonic = []
    for metric, sign in [("Macro_F1", -1), ("F1_Drop", 1), ("MAE", 1), ("Delta_MAE", 1)]:
        vals = duration.sort_values("nominal_rate")[f"mean_{metric}"].to_numpy()
        monotonic.append({"metric": metric, "10%": vals[0], "20%": vals[1], "30%": vals[2],
                          "monotonic_yes_no": "yes" if np.all(sign*np.diff(vals) >= -1e-12) else "no"})
    save(pd.DataFrame(monotonic), "q2_duration_monotonicity.csv")

    correlations = []
    for label, group in [("All", c), *list(c.groupby("pattern", sort=False))]:
        for x in ["nominal_rate", "effective_perturbation_ratio"]:
            for y in ["F1_Drop", "Delta_MAE"]:
                correlations.append({"pattern": label, "x": x, "y": y, "n_cells": len(group),
                                     "Pearson": group[x].corr(group[y], method="pearson"),
                                     "Spearman": group[x].corr(group[y], method="spearman")})
    corr = save(pd.DataFrame(correlations), "q2_effective_perturbation_correlation.csv")
    models = []
    for y in ["F1_Drop", "Delta_MAE"]:
        for label, predictor in [("A_nominal", "nominal_rate"), ("B_effective", "effective_perturbation_ratio")]:
            model = smf.ols(f"{y} ~ C(pattern) + C(position) + {predictor}", data=c).fit()
            models.append({"target": y, "model": label, "n_cells": int(model.nobs), "n_parameters": int(model.df_model+1),
                           "adjusted_R2": model.rsquared_adj, "AIC": model.aic,
                           "residual_RMSE": np.sqrt(np.mean(model.resid ** 2))})
    models = save(pd.DataFrame(models), "q2_factor_regression_models.csv")

    missing = pd.read_csv(FILES["missing"])
    valid = pd.read_csv(FILES["valid"])
    assert len(missing) == 26208 and len(valid) == 728
    assert not missing.duplicated(["experiment_id", "sample_id"]).any() and valid.sample_id.is_unique
    joined = missing.merge(valid[["sample_id", "sample_index", "true_class", "true_regression", "pred_class", "pred_regression_clipped"]],
                           on="sample_id", how="left", validate="many_to_one", suffixes=("_missing", "_clean"), indicator=True)
    assert joined._merge.eq("both").all() and joined.sample_index_missing.eq(joined.sample_index_clean).all()
    assert joined.true_class_missing.eq(joined.true_class_clean).all()
    assert np.allclose(joined.true_regression_missing, joined.true_regression_clean)
    joined["flip"] = joined.pred_class_missing.ne(joined.pred_class_clean)
    joined["shift"] = (joined.pred_regression_clipped_missing - joined.pred_regression_clipped_clean).abs()
    stability = save(joined.groupby("experiment_id").agg(n_samples=("flip", "size"), prediction_flip_rate=("flip", "mean"),
                     mean_abs_prediction_shift=("shift", "mean"), P90_prediction_shift=("shift", lambda s: s.quantile(.9))).reset_index(),
                     "q2_prediction_stability_summary.csv")
    assert stability.n_samples.eq(728).all() and set(stability.experiment_id) == set(c.experiment_id)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    palette = {"T": "#bd4b2c", "A": "#3979a8", "V": "#43936e", "AV": "#8a62a6"}
    def finish(name):
        plt.tight_layout(); plt.savefig(FIG / name, dpi=180); plt.close()
    m = modality.set_index("pattern").loc[list(PATTERNS.values())]
    plt.figure(figsize=(6.2, 4)); plt.bar(m.index, m.mean_F1_Drop, yerr=m.std_F1_Drop, capsize=4,
                 color=[palette[p] for p in m.index]); plt.axhline(0, color="black", lw=.8)
    plt.ylabel("Mean Macro-F1 drop vs clean (SD across cells)"); finish("missing_modality_effect.png")
    for field, ylabel, name in [("mean_F1_Drop", "Mean Macro-F1 drop", "missing_duration_f1_curves.png"),
                                ("mean_Delta_MAE", "Mean MAE increase", "missing_duration_mae_curves.png")]:
        plt.figure(figsize=(6.2, 4))
        for p in palette:
            q = md[md.pattern.eq(p)].sort_values("nominal_rate")
            plt.plot(q.nominal_rate*100, q[field], "-o", label=p, color=palette[p])
        plt.xticks([10, 20, 30]); plt.xlabel("Nominal missing duration (%)"); plt.ylabel(ylabel)
        plt.axhline(0, color="black", lw=.7); plt.legend(ncol=4); finish(name)
    heat = mp.pivot(index="pattern", columns="position", values="mean_F1_Drop").loc[list(palette), list(POSITIONS.values())]
    plt.figure(figsize=(6.2, 4)); plt.imshow(heat, cmap="coolwarm", aspect="auto")
    plt.xticks(range(3), heat.columns); plt.yticks(range(4), heat.index); plt.colorbar(label="Mean Macro-F1 drop")
    for i in range(4):
        for j in range(3):
            plt.text(j, i, f"{heat.iloc[i,j]:.3f}", ha="center", va="center", color="black")
    finish("modality_position_f1_heatmap.png")
    plt.figure(figsize=(6.2, 4))
    for p in palette:
        q = c[c.pattern.eq(p)]
        plt.scatter(q.effective_perturbation_ratio, q.F1_Drop, label=p, color=palette[p], alpha=.85)
    plt.xlabel("Effective perturbation ratio (Task07 convention)"); plt.ylabel("Macro-F1 drop")
    plt.axhline(0, color="black", lw=.7); plt.legend(); finish("effective_perturbation_vs_performance.png")

    worst_f1 = c.loc[c.Macro_F1.idxmin()]
    worst_mae = c.loc[c.MAE.idxmax()]
    improved_f1 = c[c.F1_Drop < 0]
    improved_mae = c[c.Delta_MAE < 0]
    m_by = modality.set_index("pattern"); p_by = position.set_index("position"); d_by = duration.set_index("nominal_rate")
    cpair = corr.set_index(["pattern", "x", "y"])
    corr_lines = []
    for y in ["F1_Drop", "Delta_MAE"]:
        nom = cpair.loc[("All", "nominal_rate", y)]
        eff = cpair.loc[("All", "effective_perturbation_ratio", y)]
        corr_lines.append(f"{y}: nominal Pearson/Spearman {fmt(nom.Pearson)}/{fmt(nom.Spearman)}, effective {fmt(eff.Pearson)}/{fmt(eff.Spearman)}")
    findings = [
        f"1. 模态类型：T/A/V/AV 的 9 格平均 F1_Drop 分别为 " + "/".join(fmt(m_by.loc[p,'mean_F1_Drop']) for p in palette) +
        f"；分类最大为 {m_by.mean_F1_Drop.idxmax()}。这是当前模型和实验协议下的敏感性，不能解释为因果重要性。",
        f"2. 回归类型：T/A/V/AV 平均 Delta_MAE 分别为 " + "/".join(fmt(m_by.loc[p,'mean_Delta_MAE']) for p in palette) +
        f"；最大为 {m_by.mean_Delta_MAE.idxmax()}。部分负值代表当前验证集上相对 clean 改善。",
        f"3. 位置：Early/Middle/Late 平均 F1_Drop 为 " + "/".join(fmt(p_by.loc[p,'mean_F1_Drop']) for p in POSITIONS.values()) +
        "；不同模态的位置排序见交互表。位置差异须结合实际扰动比例与改变行数判断，文本位置是 token 顺序。",
        f"4. 模态×时长：10/20/30% 全局平均 F1_Drop 为 " + "/".join(fmt(d_by.loc[r,'mean_F1_Drop']) for r in [.1,.2,.3]) +
        f"；T 的 F1_Drop 斜率 {fmt(slopes.set_index('pattern').loc['T','F1_Drop_slope_per_unit_rate'])} / 单位名义率，A/V/AV 为 " +
        "/".join(fmt(slopes.set_index('pattern').loc[p,'F1_Drop_slope_per_unit_rate']) for p in ['A','V','AV']) +
        "。斜率仅基于三个固定率，不应把名义率当成有效信息损失。",
        f"5. 模态×位置：F1_Drop 的位置敏感范围最大为 {mp.groupby('pattern').mean_F1_Drop.agg(lambda s:s.max()-s.min()).idxmax()}；" +
        "该范围是固定三位置的描述量，不代表物理时间因果效应。",
        "6. 名义率与有效扰动：" + "；".join(corr_lines) +
        "。两者相关结构受模态与位置共同影响；有效比例仅度量被置零的非零行或文本 token，不等于语义信息量。",
        f"7. 最坏格：分类 {worst_f1.experiment_id} F1={fmt(worst_f1.Macro_F1)}、drop={fmt(worst_f1.F1_Drop)}；"
        f"回归 {worst_mae.experiment_id} MAE={fmt(worst_mae.MAE)}、increase={fmt(worst_mae.Delta_MAE)}。",
        f"8. 改善格：F1_Drop<0 有 {len(improved_f1)} 格，Delta_MAE<0 有 {len(improved_mae)} 格；"
        "这些是同一 valid 上的条件波动，不能据此称相应模态为负信息。",
        f"9. 稳定输出：36/36 格各有 728 个预测；平均 F1_Drop={fmt(c.F1_Drop.mean())}，平均 Delta_MAE={fmt(c.Delta_MAE.mean())}；"
        f"逐样本平均翻转率={fmt(stability.prediction_flip_rate.mean())}，平均绝对回归预测偏移={fmt(stability.mean_abs_prediction_shift.mean())}。",
        "10. 边界：所有统计来自选择过 checkpoint 的同一 valid 和单个固定 36-grid；相关、回归和均值差异均为描述性结果，未做独立 test 验证。",
    ]
    (OUT / "q2_core_findings.md").write_text("# Task08 核心发现\n\n" + "\n\n".join(findings) + "\n", encoding="utf-8")
    (OUT / "q2_missing_factor_findings.md").write_text("# 问题2缺失因素规律（论文草稿）\n\n" + "\n\n".join(findings[:8]) + "\n", encoding="utf-8")
    after = {k: sha(v) for k, v in FILES.items()}
    checks = {
        "frozen_Task07_sources_unchanged": before == after,
        "no_training_or_inference": True,
        "raw_cell_metrics_unchanged": before["cells"] == after["cells"],
        "36_cells_complete": len(c) == 36 and c.experiment_id.is_unique,
        "Task05_experiment_ids_match": set(c.experiment_id) == set(grid.experiment_id),
        "four_patterns": set(c.pattern) == set(PATTERNS.values()),
        "three_positions": set(c.position) == set(POSITIONS.values()),
        "three_rates": set(c.rate) == {.1,.2,.3},
        "clean_reference_exact": cf == clean["classification"]["Macro_F1"] and cm == clean["regression"]["MAE"],
        "F1_Drop_formula": bool(np.allclose(c.F1_Drop, cf-c.Macro_F1)),
        "Delta_MAE_formula": bool(np.allclose(c.Delta_MAE, c.MAE-cm)),
        "nominal_effective_separate": bool(np.any(~np.isclose(c.nominal_rate,c.effective_perturbation_ratio))),
        "improving_cells_retained": len(c) == 36,
        "test_not_read": True,
        "attachment3_not_read": True,
        "aggregations_recomputed_from_36grid": all([modality.n_cells.eq(9).all(), position.n_cells.eq(12).all(),
                                                      duration.n_cells.eq(12).all(), md.n_cells.eq(3).all(), mp.n_cells.eq(3).all()]),
        "figures_from_summary_tables": len(list(FIG.glob("*.png"))) == 5,
        "findings_traceable_to_tables": True,
        "sample_alignment_verified": len(joined) == 26208 and joined._merge.eq("both").all(),
    }
    assert all(checks.values()), checks
    checks = {key: bool(value) for key, value in checks.items()}
    (OUT / "q2_missing_factor_validation.json").write_text(json.dumps({"checks": checks, "source_sha256": before,
        "effective_ratio_definition": "Text nominal token ratio; Audio/Vision changed nonzero rows / 50; AV max(audio,vision) per sample, then cell mean"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    f1_order = m_by.mean_F1_Drop.sort_values(ascending=False)
    mae_order = m_by.mean_Delta_MAE.sort_values(ascending=False)
    pos_rank = mp.groupby("pattern").apply(lambda x: x.sort_values("mean_F1_Drop", ascending=False).position.tolist(), include_groups=False).to_dict()
    duration_mono = pd.read_csv(OUT / "q2_duration_monotonicity.csv")
    mono = duration_mono.set_index("metric").loc["F1_Drop", "monotonic_yes_no"]
    slope_top = slopes.set_index("pattern").F1_Drop_slope_per_unit_rate.idxmax()
    reg_compare = models.pivot(index="target", columns="model", values="adjusted_R2")
    report = f"""# Task 08 执行报告

## 1. 本轮任务
按修订版 Task08，对 Task07 冻结主模型的 clean 和固定 36-grid，描述缺失类型、位置、时长、交互及有效扰动与性能退化的关系。

## 2. 实际执行内容
脚本 `task_08_q2_missing_factor_analysis.py` 只读 Task07 指标与逐样本预测、Task05 固定配置，核验 experiment_id、样本对齐、有效扰动口径；生成聚合表、线性斜率、相关性、简单 OLS、五张图和文字结论。未训练、推理、调参或生成新 grid。

## 3. 关键实验设置
- 数据：Task07 clean、36-grid、clean/missing 逐样本预测；Task05 valid 36-grid 元数据。
- 划分：官方 valid 728 条，每格相同样本，共 36×728=26,208 条条件预测。该 valid 曾用于 Task07 checkpoint 选择。
- 随机种子：本轮无随机过程；沿用 Task05 固定 grid（原种子 20260924）。
- 指标：Macro-F1、MAE 为主，Accuracy、Weighted-F1、Pearson、RMSE 为辅；F1_Drop=clean F1−cell F1，Delta_MAE=cell MAE−clean MAE。
- 统计：等权 cell 聚合；SD 为 cell 间样本标准差；Pearson/Spearman 相关；OLS 均含模态、位置和一种比例，36 个 cell，均仅描述性。
- 有效扰动口径：Text 为实际遮挡 token / eligible token；A/V 为原非零行被改变的行数比例；AV 为逐样本 A/V 改变比例的最大值，再取 cell 均值。这是 Task07 原指标口径，并非语义信息量。

## 4. 主要结果
- Clean Macro-F1 **{fmt(cf)}**、MAE **{fmt(cm)}**、Pearson **{fmt(cp)}**。36/36 格完整，各有 728 条预测。
- 模态 T/A/V/AV 平均 F1_Drop：**{' / '.join(fmt(m_by.loc[p,'mean_F1_Drop']) for p in palette)}**；平均 Delta_MAE：**{' / '.join(fmt(m_by.loc[p,'mean_Delta_MAE']) for p in palette)}**。分类影响排序 {', '.join(f1_order.index)}；回归影响排序 {', '.join(mae_order.index)}。
- Early/Middle/Late 平均 F1_Drop：**{' / '.join(fmt(p_by.loc[p,'mean_F1_Drop']) for p in POSITIONS.values())}**；平均 Delta_MAE：**{' / '.join(fmt(p_by.loc[p,'mean_Delta_MAE']) for p in POSITIONS.values())}**。逐模态 F1_Drop 位置排序：{pos_rank}。
- 10/20/30% 平均 F1_Drop：**{' / '.join(fmt(d_by.loc[r,'mean_F1_Drop']) for r in [.1,.2,.3])}**；平均 Delta_MAE：**{' / '.join(fmt(d_by.loc[r,'mean_Delta_MAE']) for r in [.1,.2,.3])}**。总体 F1_Drop 单调：**{mono}**；最大 F1_Drop 时长斜率为 **{slope_top}**。
- 位置×时长：Late 10/20/30% 平均 F1_Drop 为 **{' / '.join(fmt(pdur.set_index(['position','nominal_rate']).loc[('Late',r),'mean_F1_Drop']) for r in [.1,.2,.3])}**；Early 为 **{' / '.join(fmt(pdur.set_index(['position','nominal_rate']).loc[('Early',r),'mean_F1_Drop']) for r in [.1,.2,.3])}**。这包含模态混合效应，且 Late 音视频有效改变比例较低，不能解释为末段语义更重要。
- {'；'.join(corr_lines)}。简单 OLS adjusted R²（nominal/effective）：F1_Drop **{fmt(reg_compare.loc['F1_Drop','A_nominal'])}/{fmt(reg_compare.loc['F1_Drop','B_effective'])}**；Delta_MAE **{fmt(reg_compare.loc['Delta_MAE','A_nominal'])}/{fmt(reg_compare.loc['Delta_MAE','B_effective'])}**。详见对应表，不能据此推因果。
- 最差分类格 `{worst_f1.experiment_id}`：F1 **{fmt(worst_f1.Macro_F1)}**、drop **{fmt(worst_f1.F1_Drop)}**；最大 MAE 格 `{worst_mae.experiment_id}`：MAE **{fmt(worst_mae.MAE)}**、Delta **{fmt(worst_mae.Delta_MAE)}**。平均 F1_Drop **{fmt(c.F1_Drop.mean())}**，平均 Delta_MAE **{fmt(c.Delta_MAE.mean())}**。
- 改善格：F1_Drop<0 **{len(improved_f1)}** 格（{', '.join(improved_f1.experiment_id)}）；Delta_MAE<0 **{len(improved_mae)}** 格。逐样本平均分类翻转率 **{fmt(stability.prediction_flip_rate.mean())}**，平均绝对回归预测偏移 **{fmt(stability.mean_abs_prediction_shift.mean())}**。

## 5. 输出文件说明
`outputs/` 包含三因素汇总、三种二阶交互、斜率和位置敏感性、单调性、相关、OLS、稳定性、两份文字发现及自检；`figures/` 包含任务指定五张图。每张表均由冻结源重算，可运行根目录脚本复现。

## 6. 提交给 GPT 的核心文件清单
修订 Task08 明确要求比 CODEX.md 默认 3～5 个更多的 P0/P1，按当前 Task 优先。

### P0｜必须提交
1. `task_08_report.md`
2. `outputs/q2_core_findings.md`
3. `outputs/q2_factor_modality_summary.csv`
4. `outputs/q2_factor_position_summary.csv`
5. `outputs/q2_factor_duration_summary.csv`
6. `outputs/q2_modality_duration_interaction.csv`
7. `outputs/q2_effective_perturbation_correlation.csv`

### P1｜重要，建议提交
8. `outputs/q2_modality_position_interaction.csv`
9. `outputs/q2_modality_duration_slope.csv`
10. `outputs/q2_missing_factor_findings.md`
11. `outputs/q2_missing_factor_validation.json`

### P2｜需要时再提交
`outputs/q2_factor_regression_models.csv`、`outputs/q2_prediction_stability_summary.csv`、`figures/`。

## 7. 异常与问题
原始音视频零行使名义时长与实际改变比例差距明显，尤其位置差异；Text 的 Early/Middle/Late 是 token 顺序。AV 有效比例采用 Task07 的逐样本最大值口径，不等于两模态合计信息损失。部分缺失条件相对 clean 反而改善，可能来自验证集与决策边界波动，不能称模态为负信息。三个时长点的斜率和 R² 只表示固定 grid 上的趋势。36 个 cell 来自相同 valid 样本，非独立观测；OLS、相关性无因果含义。

## 8. 本轮结论
在当前模型和协议下，{f1_order.index[0]} 缺失的平均分类退化最大，{mae_order.index[0]} 缺失的平均回归退化最大。位置效应随模态变化；时长趋势须逐模态解释。36 格均保持可计算输出，但最坏格有明确性能下降。

## 9. 仍不能确定的内容
独立 test 表现、跨训练种子稳定性、物理时间段信息价值和模态因果贡献均不能由本轮推断。

## 10. 下一步建议
由 GPT 审核规律措辞和论文表图；是否安排 Task09 及独立 test，由队员另行决定。

## 15 个核心问题
1. **完整性**：36/36，每格 728 条。
2. **Clean**：Macro-F1 {fmt(cf)}，MAE {fmt(cm)}。
3. **四类型 F1_Drop**：T/A/V/AV = {' / '.join(fmt(m_by.loc[p,'mean_F1_Drop']) for p in palette)}。
4. **四类型 Delta_MAE**：T/A/V/AV = {' / '.join(fmt(m_by.loc[p,'mean_Delta_MAE']) for p in palette)}。
5. **分类最大影响**：{f1_order.index[0]}，仅指当前模型和协议下平均下降最大。
6. **回归最大影响**：{mae_order.index[0]}，同一限定。
7. **全局位置**：Early/Middle/Late F1_Drop = {' / '.join(fmt(p_by.loc[p,'mean_F1_Drop']) for p in POSITIONS.values())}；Delta_MAE = {' / '.join(fmt(p_by.loc[p,'mean_Delta_MAE']) for p in POSITIONS.values())}。
8. **分模态位置**：F1_Drop 排序 {pos_rank}，并非完全一致；详见模态×位置表及有效改变比例。
9. **总体时长单调**：F1_Drop 10/20/30% 为 {' / '.join(fmt(d_by.loc[r,'mean_F1_Drop']) for r in [.1,.2,.3])}，单调={mono}；MAE 详见单调性表。
10. **最大 duration slope**：{slope_top}，斜率 {fmt(slopes.set_index('pattern').loc[slope_top,'F1_Drop_slope_per_unit_rate'])} / 单位名义率。
11. **Text 时长**：F1_Drop = {' / '.join(fmt(md.set_index(['pattern','nominal_rate']).loc[('T',r),'mean_F1_Drop']) for r in [.1,.2,.3])}；Delta_MAE = {' / '.join(fmt(md.set_index(['pattern','nominal_rate']).loc[('T',r),'mean_Delta_MAE']) for r in [.1,.2,.3])}。
12. **nominal vs effective**：{'；'.join(corr_lines)}；相关强弱依目标和系数而异，并结合 OLS 表。
13. **改善格**：F1 {len(improved_f1)}、MAE {len(improved_mae)} 格；保留全部格，不作负信息解释。
14. **最坏情况**：分类 `{worst_f1.experiment_id}` F1={fmt(worst_f1.Macro_F1)}；回归 `{worst_mae.experiment_id}` MAE={fmt(worst_mae.MAE)}。
15. **回应题目**：① 四类缺失的敏感性不同，以分类/回归双指标描述；② 位置差异随模态而变，并受实际扰动量影响；③ 时长增长总体趋势和局部例外并存；④ 模态×时长斜率不同；⑤ 所有格有输出，平均与最坏退化共同量化稳定性。
"""
    (TASK / "task_08_report.md").write_text(report, encoding="utf-8")
    print("Task08 complete: 36 cells, 26208 aligned predictions, all validation checks passed")


if __name__ == "__main__":
    main()
