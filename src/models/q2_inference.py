# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task11: frozen Task07 inference on the 30 aligned Attachment 3 samples.

Run with .venv/Scripts/python.exe task_11_q2_attachment3_inference.py.
This script never opens the labeled Attachment 2 dataset or trains a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np
import pandas as pd
import torch
from transformers import BertModel

from src.models.q2_mafmt import MAFMT, arrays, tensor_batch


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/inference"
OUT = TASK / "outputs"
SUBMISSION = TASK / "submission"
T04 = ROOT / "results/q2/protocol/outputs"
T06 = ROOT / "results/q2/baseline/models"
T07 = ROOT / "results/q2/model/models"
NAMES = ("Negative", "Neutral", "Positive")
EXPECTED_BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
EXPECTED_BERT_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
EXPECTED_CHECKPOINT_SHA256 = "02c4e5b32faf59f179e7a45dd1c77632e891878bd2d37c113ba733a923de3c23"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def zero_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(row_mask.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_audit(source: Path, intervals_path: Path, summary_path: Path):
    files = sorted(source.glob("*.pkl"), key=natural_key)
    require(len(files) == 30, f"expected 30 aligned_50 pkl files, got {len(files)}")
    ids = [path.stem for path in files]
    require(len(set(ids)) == 30, "duplicate source filename stem")

    frozen_intervals = pd.read_csv(intervals_path)
    frozen_intervals = frozen_intervals[frozen_intervals.version == "aligned_50"]
    frozen_summary = pd.read_csv(summary_path)
    frozen_summary = frozen_summary[frozen_summary.version == "aligned_50"].set_index("sample_id")
    require(set(frozen_summary.index) == set(ids) and len(frozen_summary) == 30,
            "Task04 aligned summary sample set differs from source")

    observed_intervals = []
    texts, audios, visions, artificial, patterns, ratios = [], [], [], [], [], []
    interval_keys = ("sample_id", "modality", "start_index", "end_index", "length", "sequence_length", "interval_type")
    for path in files:
        with path.open("rb") as stream:
            record = pickle.load(stream)
        require(set(record) == {"test"}, f"unexpected top-level fields in {path.name}")
        block = record["test"]
        require(set(block) == {"text_bert", "audio", "vision"}, f"unexpected sample fields in {path.name}")
        text, audio, vision = (np.asarray(block[name]) for name in ("text_bert", "audio", "vision"))
        require(text.shape == (1, 3, 50) and text.dtype == np.float32, f"text shape/dtype: {path.name}")
        require(audio.shape == (1, 50, 74) and vision.shape == (1, 50, 35), f"AV shape: {path.name}")
        require(all(np.isfinite(x).all() for x in (text, audio, vision)), f"NaN/Inf input: {path.name}")
        require(np.array_equal(text, text.astype(np.int64).astype(np.float32)), f"nonintegral text tokens: {path.name}")
        require(np.isin(text[0, 1:], (0, 1)).all(), f"invalid BERT masks: {path.name}")
        require((text[0, 0] >= 0).all() and (text[0, 0] < 30522).all(), f"invalid token IDs: {path.name}")
        require(text[0, 1].sum() > 0, f"no valid text tokens: {path.name}")

        masks, all_zero = {}, {}
        for modality, value in (("audio", audio[0]), ("vision", vision[0])):
            zero = np.all(value == 0, axis=1)
            all_zero[modality] = zero
            internal = np.zeros(50, dtype=bool)
            nonzero = np.flatnonzero(~zero)
            for start, end in zero_runs(zero):
                kind = ("internal_candidate" if nonzero.size and start > nonzero[0] and end < nonzero[-1]
                        else "boundary_or_padding_ambiguous")
                observed_intervals.append((path.stem, modality, start, end, end - start + 1, 50, kind))
                if kind == "internal_candidate":
                    internal[start:end + 1] = True
            masks[modality] = internal
        pattern = ("audio+vision" if masks["audio"].any() and masks["vision"].any() else
                   "audio" if masks["audio"].any() else "vision" if masks["vision"].any() else "none")
        frozen = frozen_summary.loc[path.stem]
        require(int(frozen.audio_internal_zero_rows) == int(masks["audio"].sum()) and
                int(frozen.vision_internal_zero_rows) == int(masks["vision"].sum()),
                f"Task04 internal row counts disagree: {path.name}")
        require(np.isclose(float(frozen.audio_internal_zero_ratio), masks["audio"].mean()) and
                np.isclose(float(frozen.vision_internal_zero_ratio), masks["vision"].mean()) and
                frozen.missing_pattern == pattern, f"Task04 summary differs: {path.name}")
        require(np.isclose(float(frozen.audio_missing_ratio), all_zero["audio"].mean()) and
                np.isclose(float(frozen.vision_missing_ratio), all_zero["vision"].mean()),
                f"Task04 raw zero-row summary differs: {path.name}")
        require(int(frozen.internal_audio_vision_overlap_rows) == int((masks["audio"] & masks["vision"]).sum()) and
                bool(frozen.internal_audio_vision_equal_mask) == bool(np.array_equal(masks["audio"], masks["vision"])),
                f"Task04 AV overlap differs: {path.name}")

        # Task07 artificial input is 1=missing, unlike Task04's 0=removed convention.
        sample_mask = np.zeros((50, 3), dtype=np.float32)
        sample_mask[:, 1] = masks["audio"]
        sample_mask[:, 2] = masks["vision"]
        texts.append(text[0].astype(np.int64))
        audios.append(audio[0].astype(np.float32))
        visions.append(vision[0].astype(np.float32))
        artificial.append(sample_mask)
        patterns.append(pattern)
        ratios.append((float(masks["audio"].mean()), float(masks["vision"].mean())))

    frozen_tuples = [tuple(row[key] for key in interval_keys) for _, row in frozen_intervals.iterrows()]
    require(Counter(observed_intervals) == Counter(frozen_tuples), "Task04 frozen interval audit differs")
    require(sum(row[-1] == "internal_candidate" and row[1] == "audio" for row in observed_intervals) == 83 and
            sum(row[-1] == "internal_candidate" and row[1] == "vision" for row in observed_intervals) == 83,
            "Task04 expected internal interval distribution differs")
    require(patterns.count("audio+vision") == 27 and patterns.count("none") == 3,
            "Task04 expected pattern distribution differs")
    return files, np.stack(texts), np.stack(audios), np.stack(visions), np.stack(artificial), patterns, ratios, len(observed_intervals)


def infer_once(tokens: np.ndarray, audio: np.ndarray, vision: np.ndarray,
               artificial: np.ndarray, bert_path: Path, checkpoint_path: Path):
    bert = BertModel.from_pretrained(str(bert_path), local_files_only=True).eval()
    for parameter in bert.parameters():
        parameter.requires_grad_(False)
    require(bert.config.hidden_size == 768 and bert.config.vocab_size == 30522, "BERT configuration mismatch")
    require(not bert.training and all(not p.requires_grad for p in bert.parameters()), "BERT is not frozen")
    hidden = np.empty((len(tokens), 50, 768), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(tokens), 32):
            batch = tokens[start:start + 32]
            output = bert(input_ids=torch.from_numpy(batch[:, 0].copy()),
                          attention_mask=torch.from_numpy(batch[:, 1].copy()),
                          token_type_ids=torch.from_numpy(batch[:, 2].copy())).last_hidden_state
            hidden[start:start + len(batch)] = output.numpy().astype(np.float16)
    del bert

    features = arrays(hidden, audio, vision, tokens[:, 1], artificial)
    model = MAFMT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    predictions, raw, probabilities, gates = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(tokens), 64):
            batch = tensor_batch(features, slice(start, start + 64))
            logits, regression, gate = model(*batch)
            probabilities.append(logits.softmax(-1).numpy())
            predictions.append(logits.argmax(-1).numpy())
            raw.append(regression.numpy())
            valid = tokens[start:start + len(logits), 1].astype(bool)
            gate_np = gate.numpy()
            gates.extend(gate_np[i, valid[i]].mean(axis=0) for i in range(len(logits)))
    return (np.concatenate(predictions), np.concatenate(raw),
            np.concatenate(probabilities), np.asarray(gates), int(checkpoint["epoch"]))


def main() -> None:
    protocol = read_json(T04 / "q2_protocol.json")
    baseline = read_json(T06 / "baseline_model_manifest.json")
    full = read_json(T07 / "q2_mafmt_manifest.json")
    source = resolve_recorded_path(protocol["attachment3_directory"])
    checkpoint = T07 / "q2_mafmt_best.pt"
    bert_path = resolve_recorded_path(baseline["text_encoder_local_path"])
    bert_weights = bert_path / "model.safetensors"
    required = [source, checkpoint, bert_weights, T04 / "q2_attachment3_missing_intervals.csv",
                T04 / "q2_attachment3_missing_summary.csv"]
    require(all(path.exists() for path in required), "required source/audit/model input missing")
    require(protocol["feature_version"] == "aligned_50" and source.name == "对齐版本", "wrong feature version")
    require(protocol["attachment3_is_unlabeled"] and protocol["attachment3_for_inference_only"], "protocol mismatch")
    require(protocol["polarity_rule"]["class_ids"] == {"negative": 0, "neutral": 1, "positive": 2}, "class mapping mismatch")
    require(protocol["attachment3_text_missing_policy"].startswith("text feature rows are absent"), "text missing protocol mismatch")
    require(baseline["text_encoder_frozen"] and baseline["feature_version"] == "aligned_50", "baseline BERT manifest mismatch")
    require(bert_path.name == baseline["text_encoder_revision"] == full["bert_revision"] == EXPECTED_BERT_REVISION,
            "BERT revision mismatch")
    require(sha256(bert_weights) == baseline["text_encoder_weight_sha256"] == full["bert_weight_sha256"] == EXPECTED_BERT_SHA256,
            "BERT weight hash mismatch")
    checkpoint_hash = sha256(checkpoint)
    require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, "Task07 checkpoint hash mismatch")

    files, tokens, audio, vision, artificial, patterns, ratios, n_intervals = load_and_audit(
        source, T04 / "q2_attachment3_missing_intervals.csv", T04 / "q2_attachment3_missing_summary.csv")
    require(np.all(artificial[:, :, 0] == 0), "text padding was incorrectly masked")
    require(np.all(artificial[:, :, 1:][~np.stack((np.all(audio == 0, -1), np.all(vision == 0, -1)), -1)] == 0),
            "nonzero AV row was masked")

    first = infer_once(tokens, audio, vision, artificial, bert_path, checkpoint)
    second = infer_once(tokens, audio, vision, artificial, bert_path, checkpoint)
    pred, raw, probs, gates, epoch = first
    require(epoch == second[4] == full["best_epoch"], "checkpoint epoch mismatch")
    require(np.array_equal(pred, second[0]), "classification repeat mismatch")
    repeat_diff = float(np.max(np.abs(raw - second[1])))
    require(repeat_diff <= 1e-6, "regression repeat mismatch")
    require(np.all(np.isfinite(raw)) and np.all(np.isfinite(probs)) and np.all(np.isfinite(gates)), "nonfinite predictions")
    require(np.isin(pred, (0, 1, 2)).all(), "invalid predicted class")
    clipped = np.clip(raw, -3, 3)
    ids = [path.stem for path in files]
    labels = [NAMES[int(value)] for value in pred]
    inconsistency = [bool((p == 0 and r > 0) or (p == 2 and r < 0) or (p == 1 and abs(r) >= 1))
                     for p, r in zip(pred, clipped)]
    # Neutral discrepancy is reported with the explicit |intensity| >= 1 diagnostic threshold.
    counts = Counter(labels)
    summary = {
        "n_samples": 30,
        "negative_count": counts["Negative"], "neutral_count": counts["Neutral"], "positive_count": counts["Positive"],
        "intensity_mean": float(clipped.mean()), "intensity_std": float(clipped.std()),
        "intensity_min": float(clipped.min()), "intensity_max": float(clipped.max()),
        "intensity_median": float(np.median(clipped)),
        "mean_max_class_probability": float(probs.max(axis=1).mean()),
        "min_max_class_probability": float(probs.max(axis=1).min()),
        "polarity_intensity_inconsistency_count": sum(inconsistency),
        "neutral_inconsistency_threshold_abs_intensity": 1.0,
        "repeat_raw_max_abs_diff": repeat_diff,
    }
    require(len(ids) == len(set(ids)) == len(labels) == len(raw) == 30, "output/source mismatch")
    require(all(np.isfinite(clipped)) and np.all((-3 <= clipped) & (clipped <= 3)), "invalid clipped intensity")

    OUT.mkdir(exist_ok=True)
    SUBMISSION.mkdir(exist_ok=True)
    formal = SUBMISSION / "attachment3_predictions.csv"
    with formal.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "predicted_polarity", "predicted_intensity"))
        writer.writerows((sample_id, label, f"{float(value):.6f}") for sample_id, label, value in zip(ids, labels, clipped))
    formal_hash = sha256(formal)
    (OUT / "attachment3_predictions.sha256.txt").write_text(
        f"{formal_hash}  attachment3_predictions.csv\n", encoding="ascii")

    diagnostics = pd.DataFrame({
        "row_index": np.arange(30), "sample_id": ids, "source_filename": [path.name for path in files],
        "predicted_class_id": pred, "predicted_polarity": labels,
        "class_prob_negative": probs[:, 0], "class_prob_neutral": probs[:, 1], "class_prob_positive": probs[:, 2],
        "regression_raw": raw, "regression_clipped": clipped,
        "audio_internal_missing_ratio": [r[0] for r in ratios],
        "vision_internal_missing_ratio": [r[1] for r in ratios], "missing_pattern": patterns,
        "mean_text_gate": gates[:, 0], "mean_audio_gate": gates[:, 1], "mean_vision_gate": gates[:, 2],
        "polarity_intensity_inconsistency": inconsistency,
    })
    diagnostics.to_csv(OUT / "q2_attachment3_inference_diagnostics.csv", index=False, float_format="%.9g")
    paper = diagnostics[["sample_id", "predicted_polarity", "regression_clipped", "missing_pattern",
                         "audio_internal_missing_ratio", "vision_internal_missing_ratio"]].rename(
                             columns={"regression_clipped": "predicted_intensity"})
    paper.to_csv(OUT / "q2_attachment3_paper_table.csv", index=False, float_format="%.6f")
    pd.DataFrame([summary]).to_csv(OUT / "q2_attachment3_prediction_summary.csv", index=False, float_format="%.9g")

    with formal.open("r", encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    forbidden = ("参赛单位", "队伍编号", "队员姓名", "实验室", "team_id")
    formal_text = formal.read_text(encoding="utf-8-sig")
    checks = {
        "aligned_50_used": protocol["feature_version"] == "aligned_50" and source.name == "对齐版本",
        "input_files_30": len(files) == 30,
        "output_rows_30": len(actual) == 30,
        "unique_sample_ids_30": len({row["sample_id"] for row in actual}) == 30,
        "ids_match_filename_stems_in_natural_order": [row["sample_id"] for row in actual] == ids,
        "unlabeled_test_fields_only": True,
        "no_pseudolabel_or_training": True,
        "task07_full_checkpoint_sha256": checkpoint_hash == EXPECTED_CHECKPOINT_SHA256,
        "bert_revision_matches_task07_task10": bert_path.name == EXPECTED_BERT_REVISION,
        "bert_sha256_matches_task07_task10": sha256(bert_weights) == EXPECTED_BERT_SHA256,
        "bert_eval_inference_no_finetuning": True,
        "text_float32_to_int64_lossless": True,
        "audio_shapes_30_correct": audio.shape == (30, 50, 74),
        "vision_shapes_30_correct": vision.shape == (30, 50, 35),
        "all_inputs_finite": all(np.isfinite(x).all() for x in (tokens, audio, vision)),
        "missing_intervals_match_task04_frozen_audit": n_intervals == len(pd.read_csv(T04 / "q2_attachment3_missing_intervals.csv").query("version == 'aligned_50'")),
        "boundary_ambiguous_excluded_from_artificial_mask": True,
        "text_padding_not_artificial_missing": bool(np.all(artificial[:, :, 0] == 0)),
        "task07_artificial_mask_one_means_missing": True,
        "class_mapping_0_negative_1_neutral_2_positive": protocol["polarity_rule"]["class_ids"] == {"negative": 0, "neutral": 1, "positive": 2},
        "regression_from_task07_raw_head": True,
        "formal_intensity_clipped_minus3_plus3": all(-3 <= float(row["predicted_intensity"]) <= 3 for row in actual),
        "repeat_classification_exact": bool(np.array_equal(pred, second[0])),
        "repeat_regression_raw_within_1e_minus6": repeat_diff <= 1e-6,
        "all_outputs_finite": bool(np.isfinite(raw).all() and np.isfinite(probs).all()),
        "polarity_in_three_labels": all(row["predicted_polarity"] in NAMES for row in actual),
        "intensity_in_range": bool(np.all((-3 <= clipped) & (clipped <= 3))),
        "formal_columns_exact": list(actual[0]) == ["sample_id", "predicted_polarity", "predicted_intensity"],
        "formal_csv_has_no_index_column": "row_index" not in actual[0],
        "formal_csv_no_identity_terms": not any(term.lower() in formal_text.lower() for term in forbidden),
        "formal_sha256_saved": (OUT / "attachment3_predictions.sha256.txt").read_text(encoding="ascii").startswith(formal_hash),
    }
    validation = {
        "checks": checks, "all_passed": all(checks.values()),
        "source_directory": str(source.relative_to(ROOT)),
        "task04_protocol_sha256": sha256(T04 / "q2_protocol.json"),
        "task04_intervals_sha256": sha256(T04 / "q2_attachment3_missing_intervals.csv"),
        "task04_summary_sha256": sha256(T04 / "q2_attachment3_missing_summary.csv"),
        "task07_checkpoint_sha256": checkpoint_hash,
        "bert_revision": EXPECTED_BERT_REVISION,
        "bert_weight_sha256": EXPECTED_BERT_SHA256,
        "checkpoint_epoch": epoch,
        "repeat_raw_max_abs_diff": repeat_diff,
        "formal_csv_sha256": formal_hash,
    }
    write_json(OUT / "q2_attachment3_validation.json", validation)
    require(validation["all_passed"], "Task11 validation failed")
    print(json.dumps({"summary": summary, "formal_csv_sha256": formal_hash,
                      "validation_all_passed": validation["all_passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
