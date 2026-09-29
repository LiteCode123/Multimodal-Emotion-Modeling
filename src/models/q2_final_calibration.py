# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Read-only upstream Q2 Fix01 audit and bounded frozen-model validation.

Run: .venv/Scripts/python.exe -B task_12_q2_fix01.py
All generated files stay in results/q2/final/outputs/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score
from transformers import BertModel, BertTokenizerFast

import src.models.q2_mafmt as base
import src.models.q2_inference as prior


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/final"
OUT = TASK / "outputs"
NAMES = ("Negative", "Neutral", "Positive")
RATES = (0.10, 0.20, 0.30)
POSITIONS = ("Early", "Middle", "Late")
DELTA_GRID = tuple(round(i * 0.05, 2) for i in range(31))


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_bert(path: Path) -> BertModel:
    bert = BertModel.from_pretrained(str(path), local_files_only=True).eval()
    for parameter in bert.parameters():
        parameter.requires_grad_(False)
    require(bert.config.hidden_size == 768 and bert.config.vocab_size == 30522,
            "BERT configuration changed")
    return bert


def load_model(checkpoint_path: Path) -> base.MAFMT:
    model = base.MAFMT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(int(checkpoint["epoch"]) == 2, "checkpoint epoch changed")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def infer(model: base.MAFMT, features: tuple) -> tuple[np.ndarray, np.ndarray]:
    logits, raw = [], []
    with torch.inference_mode():
        for start in range(0, len(features[0]), 64):
            batch = base.tensor_batch(features, slice(start, start + 64))
            cls, reg, _ = model(*batch)
            logits.append(cls.numpy())
            raw.append(reg.numpy())
    return np.concatenate(logits), np.concatenate(raw)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def adjusted(logits: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    values = logits.copy()
    values[:, 1] -= delta
    return values.argmax(axis=1), softmax(values)


def metric_row(y_cls: np.ndarray, y_reg: np.ndarray, pred: np.ndarray,
               raw: np.ndarray) -> dict:
    clipped = np.clip(raw, -3, 3)
    recall = recall_score(y_cls, pred, labels=[0, 1, 2], average=None, zero_division=0)
    counts = np.bincount(pred, minlength=3)
    pearson = (float(np.corrcoef(y_reg, clipped)[0, 1])
               if np.std(y_reg) > 0 and np.std(clipped) > 0 else 0.0)
    return {
        "n": int(len(y_cls)), "Accuracy": float(accuracy_score(y_cls, pred)),
        "Macro_F1": float(f1_score(y_cls, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "Recall_Negative": float(recall[0]), "Recall_Neutral": float(recall[1]),
        "Recall_Positive": float(recall[2]),
        "MAE": float(mean_absolute_error(y_reg, clipped)), "Pearson": pearson,
        "pred_negative": int(counts[0]), "pred_neutral": int(counts[1]),
        "pred_positive": int(counts[2]),
    }


def audit_unk(files: list[Path], tokens: np.ndarray, audio: np.ndarray,
              vision: np.ndarray, valid: dict, train: dict, unaligned: Path,
              tokenizer: BertTokenizerFast) -> tuple[np.ndarray, dict, list[Path]]:
    old_unk = {}
    for split_name, block in (("train", train), ("valid", valid)):
        values = np.asarray(block["text_bert"])
        old_unk[split_name] = int(np.sum((values[:, 0] == 100) & (values[:, 1] == 1)))
    require(old_unk == {"train": 0, "valid": 0}, "Attachment 2 already has valid [UNK] tokens")

    unknown = (tokens[:, 0] == 100) & (tokens[:, 1] == 1)
    audio_zero = np.all(audio == 0, axis=2)
    vision_zero = np.all(vision == 0, axis=2)
    require(int(unknown.sum()) == 131 and int(unknown.any(axis=1).sum()) == 27,
            "Attachment 3 [UNK] count differs from the audit")
    require(np.all(audio_zero[unknown]) and np.all(vision_zero[unknown]),
            "[UNK] and audio/vision zero rows do not coincide")

    raw_paths = {re.search(r"(\d+)$", path.stem).group(1): path
                 for path in unaligned.glob("*.pkl")}
    require(len(raw_paths) == 30, "unaligned raw_text audit files missing")
    nonunk_total = nonunk_matching = original_nonunk = attention_matching = 0
    used_raw_paths = []
    for i, path in enumerate(files):
        sample_number = re.search(r"(\d+)$", path.stem).group(1)
        require(sample_number in raw_paths, f"raw_text counterpart missing: {path.name}")
        raw_path = raw_paths[sample_number]
        used_raw_paths.append(raw_path)
        with raw_path.open("rb") as stream:
            record = pickle.load(stream)
        require(set(record) == {"test"} and "raw_text" in record["test"],
                f"invalid raw_text audit file: {raw_path.name}")
        raw_text = str(np.asarray(record["test"]["raw_text"]).reshape(-1)[0])
        encoded = tokenizer(raw_text, padding="max_length", truncation=True,
                            max_length=50, return_token_type_ids=True)
        ids = np.asarray(encoded["input_ids"])
        attention = np.asarray(encoded["attention_mask"])
        original_mask = tokens[i, 1].astype(bool)
        keep = original_mask & ~unknown[i]
        nonunk_total += int(keep.sum())
        nonunk_matching += int(np.sum(tokens[i, 0, keep] == ids[keep]))
        original_nonunk += int(np.sum(ids[unknown[i]] != 100))
        attention_matching += int(np.array_equal(tokens[i, 1], attention))
    require(nonunk_matching == nonunk_total and original_nonunk == int(unknown.sum())
            and attention_matching == 30, "[UNK] replacements are not fully corroborated by raw_text")
    evidence = {
        "attachment2_train_valid_valid_unk": old_unk,
        "attachment3_unk_tokens": int(unknown.sum()),
        "attachment3_samples_with_unk": int(unknown.any(axis=1).sum()),
        "unk_on_audio_zero_rows": int(np.sum(unknown & audio_zero)),
        "unk_on_vision_zero_rows": int(np.sum(unknown & vision_zero)),
        "nonunk_tokens_matching_original": nonunk_matching,
        "nonunk_tokens_checked": nonunk_total,
        "unk_tokens_replacing_nonunk_original": original_nonunk,
        "attention_masks_matching_original": attention_matching,
    }
    return unknown, evidence, used_raw_paths


def perturb_valid(valid: dict, rate: float, position: str) -> tuple:
    tokens_observed = np.asarray(valid["text_bert"], dtype=np.int64).copy()
    tokens_canonical = tokens_observed.copy()
    audio = np.asarray(valid["audio"], dtype=np.float32).copy()
    vision = np.asarray(valid["vision"], dtype=np.float32).copy()
    artificial_observed = np.zeros((len(tokens_observed), 50, 3), np.float32)
    artificial_canonical = artificial_observed.copy()
    selected_counts, changed_audio, changed_vision = [], [], []
    for i in range(len(tokens_observed)):
        content = int(tokens_observed[i, 1].sum()) - 2
        require(content >= 1, f"valid sample {i} has no content token")
        length = min(content, max(1, math.floor(rate * content + 0.5)))
        offset = 0 if position == "Early" else (content - length) // 2 if position == "Middle" else content - length
        indices = np.arange(1 + offset, 1 + offset + length)
        changed_audio.append(int(np.sum(~np.all(audio[i, indices] == 0, axis=1))))
        changed_vision.append(int(np.sum(~np.all(vision[i, indices] == 0, axis=1))))
        tokens_observed[i, 0, indices] = 100
        tokens_canonical[i, :, indices] = 0
        audio[i, indices] = 0
        vision[i, indices] = 0
        artificial_observed[i, indices, 1:] = 1
        artificial_canonical[i, indices, :] = 1
        selected_counts.append(length)
    return (tokens_observed, tokens_canonical, audio, vision,
            artificial_observed, artificial_canonical,
            float(np.mean(selected_counts)), float(np.mean(changed_audio)),
            float(np.mean(changed_vision)))


def write_candidate(path: Path, ids: list[str], pred: np.ndarray, raw: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "predicted_polarity", "predicted_intensity"))
        writer.writerows((sid, NAMES[int(label)], f"{float(np.clip(value, -3, 3)):.6f}")
                         for sid, label, value in zip(ids, pred, raw))


def main() -> None:
    protocol = prior.read_json(prior.T04 / "q2_protocol.json")
    manifest = prior.read_json(prior.T06 / "baseline_model_manifest.json")
    full_manifest = prior.read_json(prior.T07 / "q2_mafmt_manifest.json")
    source = resolve_recorded_path(protocol["attachment3_directory"])
    unaligned = source.parent / "未对齐版本"
    attachment2 = resolve_recorded_path(protocol["attachment2_file"])
    checkpoint = prior.T07 / "q2_mafmt_best.pt"
    bert_path = resolve_recorded_path(manifest["text_encoder_local_path"])
    bert_weight = bert_path / "model.safetensors"
    old_formal = prior.TASK / "submission/attachment3_predictions.csv"
    old_diagnostics = prior.OUT / "q2_attachment3_inference_diagnostics.csv"
    required = [attachment2, source, unaligned, checkpoint, bert_weight, old_formal,
                old_diagnostics, prior.T04 / "q2_attachment3_missing_intervals.csv",
                prior.T04 / "q2_attachment3_missing_summary.csv"]
    require(all(path.exists() for path in required), "required upstream input is missing")
    require(protocol["feature_version"] == "aligned_50", "feature version changed")
    require(bert_path.name == manifest["text_encoder_revision"] == full_manifest["bert_revision"]
            == prior.EXPECTED_BERT_REVISION, "BERT revision mismatch")
    require(sha256(bert_weight) == manifest["text_encoder_weight_sha256"]
            == full_manifest["bert_weight_sha256"] == prior.EXPECTED_BERT_SHA256,
            "BERT weight mismatch")
    require(sha256(checkpoint) == prior.EXPECTED_CHECKPOINT_SHA256,
            "frozen Task07 checkpoint mismatch")
    require(sha256(old_formal) == (prior.OUT / "attachment3_predictions.sha256.txt")
            .read_text(encoding="ascii").split()[0], "Task11 formal CSV changed")
    files, tokens, audio3, vision3, artificial3, patterns, ratios, _ = prior.load_and_audit(
        source, prior.T04 / "q2_attachment3_missing_intervals.csv",
        prior.T04 / "q2_attachment3_missing_summary.csv")
    paths_to_protect = [attachment2, checkpoint, bert_weight, old_formal, old_diagnostics,
                        prior.T04 / "q2_protocol.json",
                        prior.T04 / "q2_attachment3_missing_intervals.csv",
                        prior.T04 / "q2_attachment3_missing_summary.csv",
                        ROOT / "src/data/q2_missingness.py",
                        ROOT / "src/models/q2_mafmt.py",
                        ROOT / "src/models/q2_inference.py", *files,
                        *sorted(unaligned.glob("*.pkl"))]
    upstream_hash_before = aggregate_hash(paths_to_protect)

    print("loading labeled valid and auditing [UNK]", flush=True)
    with attachment2.open("rb") as stream:
        dataset = pickle.load(stream)
    train, valid = dataset["train"], dataset["valid"]
    require(len(valid["id"]) == 728 and len(train["id"]) == 3395,
            "official split size changed")
    tokenizer = BertTokenizerFast.from_pretrained(str(bert_path), local_files_only=True)
    unknown, evidence, _ = audit_unk(files, tokens, audio3, vision3, valid,
                                     train, unaligned, tokenizer)
    del train, tokenizer, dataset
    gc.collect()
    y_cls, y_reg = base.labels(valid)

    bert = load_bert(bert_path)
    model = load_model(checkpoint)
    hidden = base.bert_sequences(bert, valid["text_bert"])
    clean = base.arrays(hidden, valid["audio"], valid["vision"],
                        valid["text_bert"][:, 1], np.zeros((728, 50, 3), np.float32))
    clean_logits, clean_raw = infer(model, clean)
    old_valid = pd.read_csv(base.OUT / "q2_main_valid_predictions.csv")
    clean_class_agreement = int(np.sum(clean_logits.argmax(axis=1) == old_valid.pred_class.to_numpy()))
    clean_raw_diff = float(np.max(np.abs(clean_raw - old_valid.pred_regression_raw.to_numpy())))
    require(clean_class_agreement == 728 and clean_raw_diff <= 1e-6,
            f"clean valid frozen inference mismatch: class {clean_class_agreement}/728, raw diff {clean_raw_diff:.9g}")
    valid_rows = []
    calibration = []
    for delta in DELTA_GRID:
        pred, _ = adjusted(clean_logits, delta)
        metrics = metric_row(y_cls, y_reg, pred, clean_raw)
        row = {"experiment": "clean_valid_calibration", "rate": 0, "position": "None",
               "representation": "clean", "neutral_logit_penalty": delta,
               "mean_masked_tokens": 0.0, "mean_changed_audio_rows": 0.0,
               "mean_changed_vision_rows": 0.0, **metrics}
        calibration.append(row)
        valid_rows.append(row)
    selected = max(calibration, key=lambda row: (round(row["Macro_F1"], 12),
                                                 round(row["Accuracy"], 12),
                                                 -row["neutral_logit_penalty"]))
    delta = float(selected["neutral_logit_penalty"])
    print(f"clean valid selected neutral logit penalty={delta:.2f}, Macro-F1={selected['Macro_F1']:.6f}", flush=True)

    for rate in RATES:
        for position in POSITIONS:
            (observed_tokens, canonical_tokens, audio, vision, observed_mask,
             canonical_mask, mean_selected, mean_audio, mean_vision) = perturb_valid(
                valid, rate, position)
            for representation, token_input, artificial in (
                ("observed_UNK", observed_tokens, observed_mask),
                ("canonical_missing", canonical_tokens, canonical_mask),
            ):
                hidden = base.bert_sequences(bert, token_input)
                features = base.arrays(hidden, audio, vision,
                                       token_input[:, 1], artificial)
                logits, raw = infer(model, features)
                for applied_delta in (0.0, delta):
                    pred, _ = adjusted(logits, applied_delta)
                    valid_rows.append({
                        "experiment": "valid_TAV_contiguous", "rate": rate,
                        "position": position, "representation": representation,
                        "neutral_logit_penalty": applied_delta,
                        "mean_masked_tokens": mean_selected,
                        "mean_changed_audio_rows": mean_audio,
                        "mean_changed_vision_rows": mean_vision,
                        **metric_row(y_cls, y_reg, pred, raw),
                    })
                del hidden, features
            print(f"validated TAV {rate:.0%} {position}", flush=True)

    print("replaying Attachment 3 with frozen model", flush=True)
    old_hidden = base.bert_sequences(bert, tokens)
    old_features = base.arrays(old_hidden, audio3, vision3, tokens[:, 1], artificial3)
    old_logits, old_raw = infer(model, old_features)
    previous = pd.read_csv(old_diagnostics)
    require(np.array_equal(old_logits.argmax(axis=1), previous.predicted_class_id.to_numpy())
            and np.max(np.abs(old_raw - previous.regression_raw.to_numpy())) <= 1e-6,
            "Task11 Attachment 3 predictions did not reproduce")

    canonical_tokens = tokens.copy()
    for channel in range(3):
        canonical_tokens[:, channel, :][unknown] = 0
    canonical_artificial = artificial3.copy()
    canonical_artificial[:, :, 0][unknown] = 1
    canonical_hidden = base.bert_sequences(bert, canonical_tokens)
    canonical_features = base.arrays(canonical_hidden, audio3, vision3,
                                     canonical_tokens[:, 1], canonical_artificial)
    candidate_logits, candidate_raw = infer(model, canonical_features)
    uncalibrated_pred, _ = adjusted(candidate_logits, 0.0)
    candidate_pred, candidate_prob = adjusted(candidate_logits, delta)

    # Reload both the text encoder and the Q2 checkpoint from disk for a second run.
    del bert, model, old_hidden, old_features, canonical_hidden, canonical_features
    gc.collect()
    bert_repeat = load_bert(bert_path)
    model_repeat = load_model(checkpoint)
    repeat_hidden = base.bert_sequences(bert_repeat, canonical_tokens)
    repeat_features = base.arrays(repeat_hidden, audio3, vision3,
                                  canonical_tokens[:, 1], canonical_artificial)
    repeat_logits, repeat_raw = infer(model_repeat, repeat_features)
    repeat_pred, _ = adjusted(repeat_logits, delta)
    repeat_raw_diff = float(np.max(np.abs(candidate_raw - repeat_raw)))
    require(np.array_equal(candidate_pred, repeat_pred) and repeat_raw_diff <= 1e-6,
            "independent candidate inference repeat failed")
    del bert_repeat, model_repeat, repeat_hidden, repeat_features
    gc.collect()

    OUT.mkdir(parents=True, exist_ok=True)
    valid_frame = pd.DataFrame(valid_rows)
    valid_file = OUT / "q2_fix01_valid_experiment.csv"
    valid_frame.to_csv(valid_file, index=False, float_format="%.9g")
    ids = [path.stem for path in files]
    candidate_file = OUT / "attachment3_predictions_fix01_candidate.csv"
    write_candidate(candidate_file, ids, candidate_pred, candidate_raw)
    previous_ids = previous.sample_id.astype(str).tolist()
    require(previous_ids == ids, "Task11 diagnostic sample order changed")
    old_labels = previous.predicted_polarity.astype(str).tolist()
    candidate_labels = [NAMES[int(value)] for value in candidate_pred]
    comparison = pd.DataFrame({
        "sample_id": ids,
        "original_polarity": old_labels,
        "canonical_polarity_without_calibration": [NAMES[int(x)] for x in uncalibrated_pred],
        "fix01_candidate_polarity": candidate_labels,
        "original_intensity": previous.regression_clipped.to_numpy(),
        "fix01_candidate_intensity": np.clip(candidate_raw, -3, 3),
        "original_neutral_probability": previous.class_prob_neutral.to_numpy(),
        "fix01_candidate_neutral_probability": candidate_prob[:, 1],
        "confirmed_text_unk_count": unknown.sum(axis=1),
        "audio_internal_missing_ratio": [value[0] for value in ratios],
        "vision_internal_missing_ratio": [value[1] for value in ratios],
        "class_changed": np.asarray(old_labels) != np.asarray(candidate_labels),
    })
    comparison_file = OUT / "q2_fix01_attachment3_comparison.csv"
    comparison.to_csv(comparison_file, index=False, float_format="%.9g")
    counts_old = np.bincount(previous.predicted_class_id.to_numpy(), minlength=3)
    counts_canonical = np.bincount(uncalibrated_pred, minlength=3)
    counts_candidate = np.bincount(candidate_pred, minlength=3)
    grid = valid_frame[valid_frame.experiment == "valid_TAV_contiguous"]
    means = (grid.groupby(["representation", "neutral_logit_penalty"])
             [["Macro_F1", "Accuracy", "MAE", "Pearson"]].mean().reset_index())
    winning_cells = {}
    for applied_delta in (0.0, delta):
        a = grid[(grid.representation == "observed_UNK") &
                 (grid.neutral_logit_penalty == applied_delta)].sort_values(["rate", "position"])
        b = grid[(grid.representation == "canonical_missing") &
                 (grid.neutral_logit_penalty == applied_delta)].sort_values(["rate", "position"])
        require(len(a) == len(b) == 9, "TAV grid incomplete")
        winning_cells[str(applied_delta)] = {
            "canonical_higher_Macro_F1": int(np.sum(b.Macro_F1.to_numpy() > a.Macro_F1.to_numpy())),
            "canonical_lower_MAE": int(np.sum(b.MAE.to_numpy() < a.MAE.to_numpy())),
        }
    upstream_hash_after = aggregate_hash(paths_to_protect)
    require(upstream_hash_before == upstream_hash_after, "upstream input modified during this Task")
    candidate_counts = {NAMES[i]: int(counts_candidate[i]) for i in range(3)}
    validation = {
        "protocol": "Task12_Q2_Fix01_v1",
        "model": "Task07 frozen Full MAF-MT",
        "bert_revision": prior.EXPECTED_BERT_REVISION,
        "checkpoint_sha256": sha256(checkpoint),
        "attachment2_used_for_selection": "official valid only",
        "attachment2_test_used_for_selection": False,
        "attachment3_labels_or_pseudolabels_used": False,
        "unaligned_raw_text_used_for_prediction": False,
        "text_unknown_audit": evidence,
        "neutral_logit_penalty_grid": list(DELTA_GRID),
        "selected_neutral_logit_penalty": delta,
        "clean_valid_macro_f1_before": float(calibration[0]["Macro_F1"]),
        "clean_valid_macro_f1_after": float(selected["Macro_F1"]),
        "tav_grid_conditions": 9,
        "tav_grid_samples_per_condition": 728,
        "tav_grid_mean_metrics": means.to_dict("records"),
        "tav_grid_canonical_winning_cells": winning_cells,
        "attachment3_counts_original": {NAMES[i]: int(counts_old[i]) for i in range(3)},
        "attachment3_counts_canonical_without_calibration": {NAMES[i]: int(counts_canonical[i]) for i in range(3)},
        "attachment3_counts_fix01_candidate": candidate_counts,
        "attachment3_class_changed_vs_original": int(comparison.class_changed.sum()),
        "attachment3_intensity_mean": float(np.mean(np.clip(candidate_raw, -3, 3))),
        "attachment3_intensity_min": float(np.min(np.clip(candidate_raw, -3, 3))),
        "attachment3_intensity_max": float(np.max(np.clip(candidate_raw, -3, 3))),
        "attachment3_repeat_raw_max_abs_diff": repeat_raw_diff,
        "attachment3_repeat_classification_exact": bool(np.array_equal(candidate_pred, repeat_pred)),
        "candidate_rows": len(ids),
        "candidate_unique_ids": len(set(ids)),
        "candidate_csv_sha256": sha256(candidate_file),
        "task11_original_csv_sha256": sha256(old_formal),
        "upstream_aggregate_sha256_before": upstream_hash_before,
        "upstream_aggregate_sha256_after": upstream_hash_after,
        "upstream_files_unchanged": upstream_hash_before == upstream_hash_after,
        "target_5_negative_13_neutral_12_positive_reached": candidate_counts == {
            "Negative": 5, "Neutral": 13, "Positive": 12},
    }
    write_json(OUT / "q2_fix01_validation.json", validation)
    print(json.dumps({
        "selected_delta": delta,
        "clean_valid_macro_f1": [validation["clean_valid_macro_f1_before"],
                                 validation["clean_valid_macro_f1_after"]],
        "attachment3_counts": candidate_counts,
        "attachment3_class_changed": validation["attachment3_class_changed_vs_original"],
        "candidate_csv_sha256": validation["candidate_csv_sha256"],
        "upstream_files_unchanged": validation["upstream_files_unchanged"],
    }, ensure_ascii=False, indent=2), flush=True)


def evaluate_frozen_test() -> None:
    """Evaluate the already selected rule on the labeled clean test once.

    This entry point never chooses a delta or rewrites the Task11 submission.
    """
    validation_path = OUT / "q2_fix01_validation.json"
    candidate_path = OUT / "attachment3_predictions_fix01_candidate.csv"
    require(validation_path.is_file() and candidate_path.is_file(),
            "run the Task12 valid and Attachment 3 experiment first")
    lock = prior.read_json(validation_path)
    delta = float(lock["selected_neutral_logit_penalty"])
    require(delta == 0.10 and sha256(candidate_path) == lock["candidate_csv_sha256"],
            "Task12 frozen rule or candidate CSV changed")
    protocol = prior.read_json(prior.T04 / "q2_protocol.json")
    manifest = prior.read_json(prior.T06 / "baseline_model_manifest.json")
    attachment2 = resolve_recorded_path(protocol["attachment2_file"])
    checkpoint = prior.T07 / "q2_mafmt_best.pt"
    bert_path = resolve_recorded_path(manifest["text_encoder_local_path"])
    require(sha256(checkpoint) == lock["checkpoint_sha256"] and
            sha256(bert_path / "model.safetensors") == prior.EXPECTED_BERT_SHA256,
            "frozen model or BERT weights changed")
    print("evaluating locked delta on labeled clean test", flush=True)
    with attachment2.open("rb") as stream:
        test = pickle.load(stream)["test"]
    require(len(test["id"]) == 727, "official test sample count changed")
    y_cls, y_reg = base.labels(test)
    bert = load_bert(bert_path)
    model = load_model(checkpoint)
    hidden = base.bert_sequences(bert, test["text_bert"])
    features = base.arrays(hidden, test["audio"], test["vision"],
                           test["text_bert"][:, 1], np.zeros((727, 50, 3), np.float32))
    logits, raw = infer(model, features)
    old = pd.read_csv(ROOT / "results/q2/test/outputs/q2_main_test_predictions.csv")
    old_class_agreement = int(np.sum(logits.argmax(axis=1) == old.pred_class.to_numpy()))
    old_raw_diff = float(np.max(np.abs(raw - old.pred_regression_raw.to_numpy())))
    require(old.sample_id.astype(str).tolist() == list(test["id"]) and
            old_class_agreement == 727 and old_raw_diff <= 1e-6,
            f"Task10 clean test reproduction failed: {old_class_agreement}/727, raw diff {old_raw_diff:.9g}")
    original_pred, _ = adjusted(logits, 0.0)
    fix_pred, _ = adjusted(logits, delta)
    changed = original_pred != fix_pred
    result = {
        "protocol": "Task12_Q2_Fix01_clean_test_v1",
        "evaluation_only": True,
        "delta_locked_before_test": delta,
        "checkpoint_sha256": sha256(checkpoint),
        "bert_weight_sha256": sha256(bert_path / "model.safetensors"),
        "n_test": 727,
        "task10_class_reproduction": old_class_agreement,
        "task10_raw_max_abs_diff": old_raw_diff,
        "original": metric_row(y_cls, y_reg, original_pred, raw),
        "fix01_calibrated": metric_row(y_cls, y_reg, fix_pred, raw),
        "class_changed": int(changed.sum()),
        "changed_true_class_counts": {
            NAMES[i]: int(np.sum(changed & (y_cls == i))) for i in range(3)},
        "test_used_for_model_or_delta_selection": False,
        "attachment3_predictions_rewritten": False,
    }
    write_json(OUT / "q2_fix01_clean_test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-frozen-test", action="store_true",
                        help="evaluate the locked Task12 rule on Attachment 2 clean test")
    args = parser.parse_args()
    if args.evaluate_frozen_test:
        evaluate_frozen_test()
    else:
        main()
