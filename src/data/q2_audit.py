# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Read-only audit of the Q2 official feature files. Run from the project root."""
from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/protocol"
OUT = TASK / "outputs"
RAW = ROOT / "data/raw/E题数据"
A2 = RAW / "附件2-数据集特征文件"
A3 = RAW / "附件3-模态缺失特征样本"
SPLITS = ("train", "valid", "test")


def write_csv(name, rows, columns):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    d = np.diff(padded)
    return [(int(s), int(e - 1)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def field_stats(version, split, field, value):
    n = len(value)
    arr = np.asarray(value)
    row = dict(version=version, split=split, field=field,
               container_type=type(value).__name__, dtype=str(arr.dtype),
               overall_shape=str(arr.shape), sample_shape=str(np.asarray(value[0]).shape) if n else "",
               n_samples=n, min="", max="", mean="", std="", nan_count="",
               inf_count="", zero_ratio="", zero_row_count="", zero_row_samples="",
               empty_count="", examples="")
    if not np.issubdtype(arr.dtype, np.number):
        row["empty_count"] = sum(x is None or str(x) == "" for x in value)
        row["examples"] = json.dumps([str(x)[:120] for x in value[:3]], ensure_ascii=False)
        return row, None
    count = zeros = nan = inf = 0
    total = total_sq = 0.0
    lo, hi = float("inf"), float("-inf")
    zero_rows = zero_samples = 0
    positions = np.zeros(arr.shape[1], dtype=np.int64) if arr.ndim == 3 and field in ("text", "audio", "vision") else None
    for start in range(0, n, 64):
        b = arr[start:start + 64]
        finite = np.isfinite(b)
        nan += int(np.isnan(b).sum())
        inf += int(np.isinf(b).sum())
        zeros += int((b == 0).sum())
        good = b[finite]
        if good.size:
            x = good.astype(np.float64, copy=False)
            count += x.size
            total += float(x.sum())
            total_sq += float(np.square(x).sum())
            lo = min(lo, float(x.min()))
            hi = max(hi, float(x.max()))
        if positions is not None:
            z = np.all(b == 0, axis=2)
            positions += z.sum(axis=0)
            zero_rows += int(z.sum())
            zero_samples += int(z.any(axis=1).sum())
    mean = total / count if count else None
    row.update(min=lo if count else "", max=hi if count else "", mean=mean if count else "",
               std=float(np.sqrt(max(0, total_sq / count - mean * mean))) if count else "",
               nan_count=nan, inf_count=inf, zero_ratio=zeros / arr.size if arr.size else "",
               zero_row_count=zero_rows, zero_row_samples=zero_samples, empty_count=0)
    return row, positions.tolist() if positions is not None else None


def audit_a2():
    fields, splits, zeros, metadata = [], [], [], {}
    all_ids = {}
    aligned_ids = {}
    xlsx = pd.read_excel(A2 / "label.xlsx")
    xlsx["audit_id"] = xlsx["video_id"].astype(str) + "$_$" + xlsx["clip_id"].astype(str)
    xmap = xlsx.set_index("audit_id")
    cross = {}
    for version in ("aligned_50", "unaligned_50"):
        print("loading", version, flush=True)
        with (A2 / (version + ".pkl")).open("rb") as f:
            data = pickle.load(f)
        metadata[version] = {"top_keys": list(data), "splits": {}}
        all_ids[version] = {}
        for split in SPLITS:
            block = data[split]
            keys = list(block)
            counts = {k: len(v) for k, v in block.items()}
            n = len(block["id"])
            ids = list(block["id"])
            all_ids[version][split] = set(ids)
            if version == "aligned_50":
                aligned_ids[split] = ids
            metadata[version]["splits"][split] = {"fields": keys, "counts": counts,
                "index_consistent": all(v == n for v in counts.values()),
                "id_unique": len(set(ids)) == n,
                "xlsx_id_set_match": set(ids) == set(xlsx.loc[xlsx["mode"] == split, "audit_id"])}
            if not metadata[version]["splits"][split]["index_consistent"]:
                raise ValueError(f"inconsistent sample counts: {version}/{split}: {counts}")
            for key, value in block.items():
                stat, pos = field_stats(version, split, key, value)
                fields.append(stat)
                if pos is not None:
                    zeros.append(dict(version=version, split=split, modality=key,
                                      samples_with_zero_rows=stat["zero_row_samples"],
                                      total_zero_rows=stat["zero_row_count"], by_position=pos))
            labels = np.asarray(block["regression_labels"]).reshape(-1)
            class_labels = np.asarray(block["classification_labels"]).reshape(-1)
            pol = np.sign(labels)
            diff = [abs(float(y) - float(xmap.loc[i, "label"])) for i, y in zip(ids, labels)]
            class_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
            ann_bad = sum(int(c) != class_map.get(str(xmap.loc[i, "annotation"]), -1)
                          for i, c in zip(ids, class_labels))
            splits.append(dict(version=version, split=split, n_samples=n,
                n_negative=int((pol < 0).sum()), n_neutral=int((pol == 0).sum()),
                n_positive=int((pol > 0).sum()), label_min=float(labels.min()),
                label_max=float(labels.max()), label_mean=float(labels.mean()),
                label_median=float(np.median(labels)), label_std=float(labels.std()),
                n_unique_labels=int(np.unique(labels).size), n_nan=int(np.isnan(labels).sum()),
                n_inf=int(np.isinf(labels).sum()), n_duplicate_ids=n - len(set(ids)),
                xlsx_max_label_difference=max(diff), xlsx_annotation_mismatch=ann_bad,
                near_zero_nonzero=int(((np.abs(labels) < 1e-7) & (labels != 0)).sum())))
            if version == "unaligned_50":
                cross[split] = {"same_id_order": ids == aligned_ids[split]}
        metadata[version]["id_overlap"] = {f"{a}_{b}": len(all_ids[version][a] & all_ids[version][b])
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
        del data
    metadata["cross_version"] = cross
    return fields, splits, zeros, metadata, xlsx


def audit_a3():
    intervals, summaries, schema = [], [], {}
    for version, directory in (("aligned_50", "对齐版本"), ("unaligned_50", "未对齐版本")):
        files = sorted((A3 / directory).glob("*.pkl"))
        schema[version] = {"n_samples": len(files), "files": [], "field_shapes": defaultdict(Counter), "field_dtypes": defaultdict(Counter)}
        for path in files:
            with path.open("rb") as f:
                obj = pickle.load(f)
            block = obj["test"]
            schema[version]["files"].append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                                "top_keys": list(obj), "fields": list(block)})
            for key, val in block.items():
                arr = np.asarray(val)
                schema[version]["field_shapes"][key][str(arr.shape)] += 1
                schema[version]["field_dtypes"][key][str(arr.dtype)] += 1
            masks = {}
            sample_id = path.stem
            for modality in ("text", "audio", "vision"):
                if modality not in block:
                    masks[modality] = None
                    continue
                arr = np.asarray(block[modality])[0]
                if arr.ndim != 2:
                    masks[modality] = None
                    continue
                z = np.all(arr == 0, axis=1)
                masks[modality] = z
                observed = np.flatnonzero(~z)
                for s, e in runs(z):
                    kind = ("internal_candidate" if observed.size and s > observed[0] and e < observed[-1]
                            else "boundary_or_padding_ambiguous")
                    intervals.append(dict(version=version, sample_id=sample_id, modality=modality,
                        start_index=s, end_index=e, length=e - s + 1, sequence_length=len(z),
                        missing_ratio=(e - s + 1) / len(z), interval_type=kind))
            a, v = masks["audio"], masks["vision"]
            overlap = int(np.logical_and(a, v).sum()) if a is not None and v is not None and len(a) == len(v) else 0
            union = int(np.logical_or(a, v).sum()) if a is not None and v is not None and len(a) == len(v) else 0
            # Distinguish raw zero runs from candidate internal gaps. Leading/trailing
            # zero rows are structurally ambiguous, so only bracketed runs are candidates.
            ai, vi = a.copy(), v.copy()
            for mask in (ai, vi):
                observed = np.flatnonzero(~mask)
                if observed.size:
                    mask[:observed[0]] = False
                    mask[observed[-1] + 1:] = False
                else:
                    mask[:] = False
            candidate = ("audio+vision" if ai.any() and vi.any() else "audio" if ai.any()
                         else "vision" if vi.any() else "none")
            summaries.append(dict(version=version, sample_id=sample_id,
                text_missing_ratio="" if masks["text"] is None else float(masks["text"].mean()),
                audio_missing_ratio=float(a.mean()), vision_missing_ratio=float(v.mean()),
                audio_vision_overlap_ratio=overlap / union if union else 0,
                audio_vision_overlap_rows=overlap, audio_vision_equal_mask=bool(np.array_equal(a, v)),
                audio_internal_zero_rows=int(ai.sum()), vision_internal_zero_rows=int(vi.sum()),
                audio_internal_zero_ratio=float(ai.mean()), vision_internal_zero_ratio=float(vi.mean()),
                internal_audio_vision_overlap_rows=int(np.logical_and(ai, vi).sum()),
                internal_audio_vision_equal_mask=bool(np.array_equal(ai, vi)),
                missing_pattern=candidate, text_available="text_bert" in block or "raw_text" in block))
        schema[version]["field_shapes"] = {k: dict(v) for k, v in schema[version]["field_shapes"].items()}
        schema[version]["field_dtypes"] = {k: dict(v) for k, v in schema[version]["field_dtypes"].items()}
    return intervals, summaries, schema


def main():
    OUT.mkdir(exist_ok=True)
    for p in (A2 / "aligned_50.pkl", A2 / "unaligned_50.pkl", A2 / "label.xlsx"):
        if not p.is_file():
            raise FileNotFoundError(p)
    if len(list(A3.rglob("*.pkl"))) != 60:
        raise ValueError("Expected 60 attachment-3 files")
    fields, splits, zeros, metadata, xlsx = audit_a2()
    intervals, summaries, a3schema = audit_a3()
    write_csv("q2_field_summary.csv", fields, list(fields[0]))
    write_csv("q2_split_summary.csv", splits, list(splits[0]))
    write_csv("q2_attachment3_missing_intervals.csv", intervals, list(intervals[0]))
    write_csv("q2_attachment3_missing_summary.csv", summaries, list(summaries[0]))
    source_files = [A2 / "aligned_50.pkl", A2 / "unaligned_50.pkl", A2 / "label.xlsx"] + sorted(A3.rglob("*.pkl"))
    detail = {"source_files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in source_files],
              "attachment2": metadata, "attachment3": a3schema, "zero_row_positions": zeros,
              "label_xlsx": {"columns": list(xlsx.columns[:-1]), "n_rows": len(xlsx),
                             "label_min": float(xlsx.label.min()), "label_max": float(xlsx.label.max())}}
    save_json("q2_audit_details.json", detail)
    print("audit complete", {"fields": len(fields), "intervals": len(intervals), "samples_a3": len(summaries)}, flush=True)


if __name__ == "__main__":
    main()
