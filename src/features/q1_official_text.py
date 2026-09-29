"""Rebuild Q1 50-bin text features from label-100.xlsx, preserving other modalities.

Run: .venv/Scripts/python.exe -B task_03_official_text_rebuild.py
All outputs are new files in results/q1/official/.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q1/official"
OUT = TASK / "outputs"
FIG = TASK / "figures"
MANIFEST = ROOT / "results/q1/media/outputs/manifest_100.csv"
BASE = ROOT / "data/intermediate/q1/multimodal_features_50bin.csv"
OLD_SUMMARY = TASK / "outputs/q1_100_sample_alignment_summary.csv"
LABEL = ROOT / "data/raw/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx"
FEATURE_OUT = OUT / "q1_official_text_multimodal_features_50bin.csv"
WORD_OUT = OUT / "q1_official_word_alignment_100.csv"
SUMMARY_OUT = OUT / "q1_official_text_sample_summary_100.csv"
SCHEMA_OUT = OUT / "q1_official_text_schema.json"
VALIDATION_OUT = OUT / "q1_official_text_validation.json"
TYPICAL = "-s9qJ7ATP7w$_$5"
LEXICAL = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
N_BINS = 50
DIM = 768
SR = 16000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite historical output: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(word: str) -> str:
    return word.lower().replace("’", "'")


def official_words(text: str) -> list[dict]:
    return [{"word": m.group(), "norm": normalize(m.group()), "char_start": m.start(),
             "char_end": m.end()} for m in LEXICAL.finditer(text)]


def lcs_pairs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = 1 + dp[i+1][j+1] if a[i] == b[j] else max(dp[i+1][j], dp[i][j+1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j] and dp[i][j] == dp[i+1][j+1] + 1:
            pairs.append((i, j)); i += 1; j += 1
        elif dp[i+1][j] >= dp[i][j+1]:
            i += 1
        else:
            j += 1
    return pairs


def decode_audio(path: Path, t0: float, t1: float) -> np.ndarray:
    import av
    from av.audio.resampler import AudioResampler
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if len(streams) != 1:
            raise ValueError(f"Expected one audio stream: {path}")
        resampler = AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(streams[0]):
            chunks.extend(x.to_ndarray().reshape(-1) for x in resampler.resample(frame))
        chunks.extend(x.to_ndarray().reshape(-1) for x in resampler.resample(None))
    if not chunks:
        raise ValueError(f"Empty decoded audio: {path}")
    wave = np.concatenate(chunks).astype(np.float32)
    return wave[max(0, round(t0*SR)):min(len(wave), round(t1*SR))]


def transcribe_words(model, wave: np.ndarray, duration: float) -> list[dict]:
    if len(wave) == 0:
        return []
    segments, _ = model.transcribe(wave, language="en", word_timestamps=True,
                                   beam_size=5, vad_filter=False)
    words = []
    for segment in segments:
        for item in segment.words or []:
            raw = str(item.word).strip()
            start = max(0.0, min(duration, float(item.start)))
            end = max(start, min(duration, float(item.end)))
            for match in LEXICAL.finditer(raw):
                words.append({"word": match.group(), "raw_word": raw,
                              "norm": normalize(match.group()), "start": start, "end": end})
    return words


def align_words(words: list[dict], asr: list[dict], duration: float) -> tuple[list[dict], str, str, int]:
    n = len(words)
    pairs = lcs_pairs([w["norm"] for w in words], [w["norm"] for w in asr])
    ratio = len(pairs) / n if n else 0.0
    anchor_ok = all(asr[j]["end"] > asr[j]["start"] for _, j in pairs)
    # Whisper word intervals can overlap slightly while remaining ordered.
    anchor_ok &= all(asr[j1]["start"] <= asr[j2]["start"] + 1e-8 and
                     asr[j1]["end"] <= asr[j2]["end"] + 1e-8
                     for (_, j1), (_, j2) in zip(pairs, pairs[1:]))
    if not n:
        return [], "empty_official", "no_text", 0
    reason = ""
    if not asr:
        reason = "empty_asr"
    elif ratio < 0.5:
        reason = "lcs_ratio_below_0.5"
    elif not anchor_ok:
        reason = "invalid_or_out_of_order_anchor_times"
    if reason:
        for i, w in enumerate(words):
            w.update(start=duration*i/n, end=duration*(i+1)/n,
                     time_source="proportional_fallback", matched_asr_word="",
                     matched_asr_raw="", asr_word_index="", asr_start_sec="", asr_end_sec="",
                     fallback_reason=reason)
        # Preserve lexical correspondence even though these timestamps are not used.
        for i, j in pairs:
            words[i].update(matched_asr_word=asr[j]["word"], matched_asr_raw=asr[j]["raw_word"],
                            asr_word_index=j, asr_start_sec=asr[j]["start"], asr_end_sec=asr[j]["end"])
        return words, "proportional_fallback", "unverified_proportional", len(pairs)

    for i, j in pairs:
        words[i].update(start=asr[j]["start"], end=asr[j]["end"], time_source="asr_lcs_anchor",
                        matched_asr_word=asr[j]["word"], matched_asr_raw=asr[j]["raw_word"],
                        asr_word_index=j, asr_start_sec=asr[j]["start"],
                        asr_end_sec=asr[j]["end"], fallback_reason="")
    anchors = [i for i, _ in pairs]
    first = anchors[0]
    for i in range(first):
        words[i].update(start=words[first]["start"]*i/first,
                        end=words[first]["start"]*(i+1)/first,
                        time_source="interpolated_leading", matched_asr_word="",
                        matched_asr_raw="", asr_word_index="", asr_start_sec="", asr_end_sec="",
                        fallback_reason="unmatched_official_word")
    for left, right in zip(anchors, anchors[1:]):
        gap = right-left-1
        for i in range(left+1, right):
            start, end = sorted((words[left]["end"], words[right]["start"]))
            words[i].update(start=start+(end-start)*(i-left-1)/gap,
                            end=start+(end-start)*(i-left)/gap,
                            time_source="interpolated_between_anchors", matched_asr_word="",
                            matched_asr_raw="", asr_word_index="", asr_start_sec="", asr_end_sec="",
                            fallback_reason="unmatched_official_word")
    last = anchors[-1]
    tail = n-last-1
    for i in range(last+1, n):
        start = words[last]["end"]
        words[i].update(start=start+(duration-start)*(i-last-1)/tail,
                        end=start+(duration-start)*(i-last)/tail,
                        time_source="interpolated_trailing", matched_asr_word="",
                        matched_asr_raw="", asr_word_index="", asr_start_sec="", asr_end_sec="",
                        fallback_reason="unmatched_official_word")
    if any(not (0 <= w["start"] <= w["end"] <= duration+1e-8) for w in words):
        raise ValueError("Aligned word time outside effective interval")
    return words, "lcs_timestamps_interpolated", "asr_anchor_assisted_unverified", len(pairs)


def embed_official(text: str, words: list[dict], tokenizer, bert) -> np.ndarray:
    import torch
    if not words:
        return np.zeros((0, DIM), np.float32)
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=False)
    if enc["input_ids"].shape[1] > 512:
        raise ValueError("Official text exceeds BERT 512 token limit; explicit chunking required")
    offsets = enc.pop("offset_mapping")[0].numpy()
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0].cpu().numpy()
    vectors = []
    for w in words:
        ix = [j for j, (a,b) in enumerate(offsets) if b > a and int(a) < w["char_end"] and int(b) > w["char_start"]]
        if not ix:
            raise ValueError(f"No BERT pieces for official word {w['word']!r}")
        vectors.append(hidden[ix].mean(axis=0))
    result = np.asarray(vectors, np.float32)
    if result.shape != (len(words), DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid official BERT word vectors")
    return result


def bin_features(words: list[dict], vectors: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, duration, N_BINS+1)
    out = np.zeros((N_BINS, DIM), np.float32)
    mask = np.zeros(N_BINS, np.uint8)
    for k, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        overlap = np.asarray([max(0.0, min(w["end"], right)-max(w["start"], left)) for w in words])
        if overlap.sum() > 1e-9:
            out[k] = np.average(vectors, axis=0, weights=overlap).astype(np.float32)
            mask[k] = 1
    return out, mask


def draw_typical(words: list[dict], rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.parse_math"] = False
    import matplotlib.pyplot as plt
    focus = [r for r in rows if int(r["bin_id"]) in (7,8,9)]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    left = float(focus[0]["bin_start_sec"]); right = float(focus[-1]["bin_end_sec"])
    for row in focus:
        a,b = float(row["bin_start_sec"]), float(row["bin_end_sec"])
        ax.axvspan(a,b,alpha=.12,color="#4679a8")
        bx.bar((a+b)/2, float(row["text_present"]), width=(b-a)*.9, color="#4679a8")
        norm = np.linalg.norm([float(row[f"text_{j:03d}"]) for j in range(DIM)])
        bx.text((a+b)/2, .5, f"bin {row['bin_id']}\n||v||={norm:.2f}",
                ha="center", va="center", color="white", fontsize=8)
    for i,w in enumerate(words):
        a,b = max(left,w["start"]), min(right,w["end"])
        if b-a > 1e-9:
            y = .25 + (i%3)*.32
            ax.plot([a,b],[y,y],lw=9,solid_capstyle="butt",color="#ee9b46" if w["time_source"]=="asr_lcs_anchor" else "#ae788f")
            ax.text((a+b)/2,y+.055,w["word"],ha="center",fontsize=9)
    ax.set_xlim(left,right); ax.set_ylim(0,1.3); ax.set_yticks([])
    ax.set_title("Official transcript word intervals, bins 7-9")
    bx.set_ylim(0,1.1); bx.set_yticks([0,1]); bx.set_ylabel("text_present")
    bx.set_xlabel("Effective local time (s)")
    fig.savefig(path,dpi=180)
    plt.close(fig)


def finalize_existing() -> None:
    """Resume figure and exported-file audit after a completed CSV write."""
    for path in (MANIFEST, BASE, LABEL, FEATURE_OUT, WORD_OUT, SUMMARY_OUT, SCHEMA_OUT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if VALIDATION_OUT.exists():
        raise FileExistsError(VALIDATION_OUT)
    manifest = read_csv(MANIFEST)
    base = read_csv(BASE)
    exported = read_csv(FEATURE_OUT)
    words = read_csv(WORD_OUT)
    samples = read_csv(SUMMARY_OUT)
    with BASE.open("r", encoding="utf-8-sig", newline="") as f:
        fields = next(csv.reader(f))
    text_cols = [f"text_{j:03d}" for j in range(DIM)]
    replaced = set(text_cols) | {"text_present", "text_status"}
    untouched = [c for c in fields if c not in replaced]
    ids = [m["sample_id"] for m in manifest]
    labels = pd.read_excel(LABEL)
    official = {f"{str(r.video_id).strip()}$_${int(r.clip_id)}": r for r in labels.itertuples(index=False)}
    label_ok = len(ids)==len(set(ids))==len(official)==100 and all(
        m["official_text"]==str(official[m["sample_id"]].text) and
        float(m["official_label"])==float(official[m["sample_id"]].label)
        for m in manifest)
    base_by_key = {(r["sample_id"],int(r["bin_id"])):r for r in base}
    keys = [(r["sample_id"],int(r["bin_id"])) for r in exported]
    key_ok = len(base)==len(base_by_key)==len(exported)==len(set(keys))==5000 and set(keys)==set(base_by_key)
    if not key_ok:
        raise AssertionError("Exported keys differ from upstream")
    nontext_ok = all(all(r[c]==base_by_key[(r["sample_id"],int(r["bin_id"]))][c] for c in untouched) for r in exported)
    word_map = defaultdict(list)
    for w in words:
        word_map[w["sample_id"]].append((float(w["start_sec"]),float(w["end_sec"])))
    sample_by_id = {s["sample_id"]:s for s in samples}
    duration_ok = all(
        math.isclose(float(s["D_star_sec"]),float(m["effective_t1"])-float(m["effective_t0"]),abs_tol=1e-8)
        for m in manifest for s in [sample_by_id[m["sample_id"]]])
    time_ok = all(0<=a<=b<=float(sample_by_id[sid]["D_star_sec"])+1e-8
                  for sid, intervals in word_map.items() for a,b in intervals)
    mask_ok = True
    axis_ok = True
    finite_ok = True
    zero_when_empty = True
    present_nonzero = True
    for r in exported:
        sid,k = r["sample_id"],int(r["bin_id"])
        left,right = float(r["bin_start_sec"]),float(r["bin_end_sec"])
        duration = float(sample_by_id[sid]["D_star_sec"])
        axis_ok &= math.isclose(left,duration*k/N_BINS,abs_tol=1e-8) and math.isclose(right,duration*(k+1)/N_BINS,abs_tol=1e-8)
        coverage = any(min(b,right)-max(a,left)>1e-9 for a,b in word_map[sid])
        mask_ok &= int(r["text_present"])==int(coverage)
        vector = np.asarray([float(r[c]) for c in text_cols])
        finite_ok &= bool(np.isfinite(vector).all())
        zero_when_empty &= bool(int(r["text_present"]) or np.all(vector==0))
        present_nonzero &= bool(not int(r["text_present"]) or np.any(vector!=0))
    count_ok = all(sum(r["sample_id"]==sid for r in exported)==N_BINS for sid in ids)
    word_count_ok = all(sum(w["sample_id"]==sid for w in words)==int(sample_by_id[sid]["official_word_count"]) for sid in ids)
    typical_words = [{**w,"start":float(w["start_sec"]),"end":float(w["end_sec"]),"word":w["official_word"]}
                     for w in words if w["sample_id"]==TYPICAL]
    typical_rows = [r for r in exported if r["sample_id"]==TYPICAL]
    figure_path = FIG/"typical_official_word_alignment_bins_7_9.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    draw_typical(typical_words,typical_rows,figure_path)
    method_counts = dict(Counter(s["time_method"] for s in samples))
    checks = {"official_text_label_manifest_exact":label_ok,"keys_match_upstream":key_ok,
              "each_sample_50_bins":count_ok,"nontext_fields_exactly_preserved":nontext_ok,
              "D_star_matches_manifest":duration_ok,"bin_axis_matches_D_star":axis_ok,
              "word_times_in_D_star":time_ok,"official_word_counts_match":word_count_ok,
              "text_mask_equals_word_overlap":mask_ok,"text_vectors_finite":finite_ok,
              "empty_bin_vectors_zero":zero_when_empty,"present_bin_vectors_nonzero":present_nonzero}
    validation = {"status":"PASS" if all(checks.values()) else "FAIL",**checks,
                  "sample_count":len(ids),"row_count":len(exported),"unique_keys":len(set(keys)),
                  "columns":len(fields),"text_dimensions":DIM,"word_rows":len(words),
                  "method_counts":method_counts,
                  "changed_text_bin_samples":sum(int(s["delta_vs_old_asr_bins"])!=0 for s in samples),
                  "new_text_valid_bins_total":sum(int(s["new_text_valid_bins"]) for s in samples),
                  "old_asr_text_valid_bins_total":sum(int(s["old_asr_text_valid_bins"]) for s in samples),
                  "prior_official_summary_bin_mismatches":sum(int(s["delta_vs_prior_official_summary_bins"])!=0 for s in samples),
                  "typical_sample":sample_by_id[TYPICAL]}
    VALIDATION_OUT.write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(validation,ensure_ascii=False,indent=2))
    if validation["status"]!="PASS":
        raise AssertionError("Final validation failed")


def enrich_fallback_matches() -> None:
    """Complete lexical audit columns for this run's already exported fallback rows."""
    if not (WORD_OUT.is_file() and SUMMARY_OUT.is_file() and MANIFEST.is_file()):
        raise FileNotFoundError("Existing word, summary, and manifest CSVs are required")
    rows = read_csv(WORD_OUT)
    summaries = {r["sample_id"]:r for r in read_csv(SUMMARY_OUT)}
    manifest = {r["sample_id"]:r for r in read_csv(MANIFEST)}
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["sample_id"]].append(row)
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en",device="cpu",compute_type="int8")
    fallback_ids = [sid for sid,s in summaries.items() if s["time_method"]=="proportional_fallback"]
    for ordinal,sid in enumerate(fallback_ids,1):
        m = manifest[sid]
        duration = float(m["effective_t1"])-float(m["effective_t0"])
        wave = decode_audio(ROOT/m["video_path"],float(m["effective_t0"]),float(m["effective_t1"]))
        asr = transcribe_words(model,wave,duration)
        words,method,_,matched = align_words(official_words(m["official_text"]),asr,duration)
        if method!="proportional_fallback" or matched!=int(summaries[sid]["lcs_matched_words"]) or len(words)!=len(by_id[sid]):
            raise ValueError(f"ASR reproduction differs for {sid}; refusing audit update")
        for existing,word in zip(by_id[sid],words):
            if existing["official_word"]!=word["word"] or not math.isclose(float(existing["start_sec"]),word["start"],abs_tol=1e-8):
                raise ValueError(f"Official fallback word/time differs for {sid}")
            for field in ("matched_asr_word","matched_asr_raw","asr_word_index","asr_start_sec","asr_end_sec"):
                existing[field] = str(word[field])
        print(f"fallback lexical audit {ordinal}/{len(fallback_ids)} {sid}: LCS={matched}",flush=True)
    temp = WORD_OUT.with_suffix(".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    write_csv(temp,rows,list(rows[0]))
    temp.replace(WORD_OUT)
    print(f"Recorded fallback LCS counterparts for {len(fallback_ids)} samples")


def main() -> None:
    required = [MANIFEST, BASE, OLD_SUMMARY, LABEL]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    new_paths = [FEATURE_OUT, WORD_OUT, SUMMARY_OUT, SCHEMA_OUT, VALIDATION_OUT,
                 FIG / "typical_official_word_alignment_bins_7_9.png"]
    if any(p.exists() for p in new_paths):
        raise FileExistsError("A new output already exists; remove it explicitly before rerunning")
    manifest = read_csv(MANIFEST)
    summary_old = read_csv(OLD_SUMMARY)
    labels = pd.read_excel(LABEL)
    official = {f"{str(r.video_id).strip()}$_${int(r.clip_id)}": r for r in labels.itertuples(index=False)}
    ids = [r["sample_id"] for r in manifest]
    assert len(ids) == len(set(ids)) == len(official) == len(summary_old) == 100
    assert set(ids) == set(official) == {r[list(r)[1]] for r in summary_old}
    for row in manifest:
        ref = official[row["sample_id"]]
        if row["official_text"] != str(ref.text) or float(row["official_label"]) != float(ref.label) or row["annotation"] != str(ref.annotation):
            raise ValueError(f"Manifest differs from label-100.xlsx: {row['sample_id']}")
        if not (resolve_recorded_path(row["video_path"])).is_file():
            raise FileNotFoundError(resolve_recorded_path(row["video_path"]))
    base = read_csv(BASE)
    with BASE.open("r", encoding="utf-8-sig", newline="") as f:
        base_fields = next(csv.reader(f))
    if len(base_fields) != 843 or len(base) != 5000:
        raise ValueError("Expected 843-column, 5000-row face-enhanced upstream feature table")
    text_cols = [f"text_{j:03d}" for j in range(DIM)]
    replace_cols = set(text_cols) | {"text_present", "text_status"}
    if not replace_cols.issubset(base_fields):
        raise ValueError("Upstream text fields missing")
    base_by_id = defaultdict(dict)
    for row in base:
        sid, k = row["sample_id"], int(row["bin_id"])
        if k in base_by_id[sid]:
            raise ValueError(f"Duplicate upstream key: {sid}, {k}")
        base_by_id[sid][k] = row
    if set(base_by_id) != set(ids) or any(set(base_by_id[sid]) != set(range(N_BINS)) for sid in ids):
        raise ValueError("Upstream sample/bin keys differ from manifest")

    from faster_whisper import WhisperModel
    from transformers import AutoModel, AutoTokenizer
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    asr_model = WhisperModel("small.en", device="cpu", compute_type="int8")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to("cpu")
    bert.eval()
    word_rows = []; new_rows = []; sample_rows = []
    methods = Counter(); changed = []
    old_by_id = {r[list(r)[1]]: r for r in summary_old}
    typical_words = None; typical_rows = None
    for ordinal, m in enumerate(manifest, 1):
        sid = m["sample_id"]
        duration = float(m["effective_t1"])-float(m["effective_t0"])
        if not math.isclose(duration, float(m["duration_sec"]), abs_tol=1e-6):
            raise ValueError(f"Manifest D* mismatch: {sid}")
        words = official_words(m["official_text"])
        wave = decode_audio(ROOT/m["video_path"], float(m["effective_t0"]), float(m["effective_t1"]))
        asr = transcribe_words(asr_model, wave, duration)
        words, method, reliability, matched = align_words(words, asr, duration)
        vectors = embed_official(m["official_text"], words, tokenizer, bert)
        feats, mask = bin_features(words, vectors, duration)
        if not np.isfinite(feats).all():
            raise ValueError(f"Non-finite text feature: {sid}")
        methods[method] += 1
        for wi,w in enumerate(words):
            word_rows.append({"sample_id":sid,"word_index":wi,"official_text":m["official_text"],
                              "official_word":w["word"],"normalized_word":w["norm"],
                              "char_start":w["char_start"],"char_end":w["char_end"],
                              "start_sec":w["start"],"end_sec":w["end"],
                              "time_source":w["time_source"],"matched_asr_word":w["matched_asr_word"],
                              "matched_asr_raw":w["matched_asr_raw"],"asr_word_index":w["asr_word_index"],
                              "asr_start_sec":w["asr_start_sec"],"asr_end_sec":w["asr_end_sec"],
                              "fallback_reason":w["fallback_reason"],"time_reliability":reliability,
                              "effective_duration_sec":duration})
        sample_new = []
        for k in range(N_BINS):
            row = dict(base_by_id[sid][k])
            expected_a, expected_b = duration*k/N_BINS, duration*(k+1)/N_BINS
            if not (math.isclose(float(row["bin_start_sec"]),expected_a,abs_tol=1e-8) and
                    math.isclose(float(row["bin_end_sec"]),expected_b,abs_tol=1e-8) and
                    float(row["official_label"])==float(m["official_label"])):
                raise ValueError(f"Upstream axis/label mismatch: {sid}, {k}")
            row["text_present"] = str(int(mask[k]))
            row["text_status"] = "official_" + method
            row.update({col:repr(float(feats[k,j])) for j,col in enumerate(text_cols)})
            for col in base_fields:
                if col not in replace_cols and row[col] != base_by_id[sid][k][col]:
                    raise AssertionError(f"Non-text field changed: {sid}, {k}, {col}")
            sample_new.append(row)
        new_rows.extend(sample_new)
        old_bins = sum(int(base_by_id[sid][k]["text_present"]) for k in range(N_BINS))
        new_bins = int(mask.sum())
        if old_bins != new_bins:
            changed.append(sid)
        original_summary = old_by_id[sid]
        old_official_bins = int(original_summary[list(original_summary)[10]])
        sample_rows.append({"sample_id":sid,"official_text":m["official_text"],
                            "official_label":m["official_label"],"annotation":m["annotation"],
                            "effective_t0":m["effective_t0"],"effective_t1":m["effective_t1"],
                            "D_star_sec":duration,"official_word_count":len(words),"asr_word_count":len(asr),
                            "lcs_matched_words":matched,"word_match_rate":matched/len(words) if words else 0,
                            "time_method":method,"time_reliability":reliability,
                            "matched_anchor_words":sum(w["time_source"]=="asr_lcs_anchor" for w in words),
                            "interpolated_words":sum(w["time_source"].startswith("interpolated") for w in words),
                            "proportional_words":sum(w["time_source"]=="proportional_fallback" for w in words),
                            "new_text_valid_bins":new_bins,"new_text_bin_rate":new_bins/N_BINS,
                            "old_asr_text_valid_bins":old_bins,"delta_vs_old_asr_bins":new_bins-old_bins,
                            "prior_official_summary_bins":old_official_bins,
                            "delta_vs_prior_official_summary_bins":new_bins-old_official_bins,
                            "audio_valid_bins":sum(int(base_by_id[sid][k]["audio_valid"]) for k in range(N_BINS)),
                            "vision_valid_bins":sum(int(base_by_id[sid][k]["vision_present"]) for k in range(N_BINS)),
                            "face_valid_bins":sum(int(base_by_id[sid][k]["face_present"]) for k in range(N_BINS))})
        if sid == TYPICAL:
            typical_words = words
            typical_rows = sample_new
        print(f"{ordinal:3d}/100 {sid}: official={len(words)} asr={len(asr)} LCS={matched} bins={new_bins} {method}",flush=True)

    if len(new_rows)!=5000 or len(word_rows)!=sum(r["official_word_count"] for r in sample_rows):
        raise AssertionError("Output row counts incorrect")
    if typical_words is None or typical_rows is None:
        raise AssertionError("Typical sample absent")
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    write_csv(FEATURE_OUT,new_rows,base_fields)
    write_csv(WORD_OUT,word_rows,list(word_rows[0]))
    write_csv(SUMMARY_OUT,sample_rows,list(sample_rows[0]))
    schema={"feature_count":len(base_fields),"samples":100,"bins_per_sample":N_BINS,
            "source_feature_table":str(BASE.relative_to(ROOT)),
            "official_source":str(LABEL.relative_to(ROOT)),
            "text_vector_columns":text_cols,"replaced_fields":sorted(replace_cols),
            "preserved_fields":[c for c in base_fields if c not in replace_cols],
            "text_model":"bert-base-uncased pretrained, eval/no_grad, last-layer WordPiece mean per official word",
            "text_input":"exact official transcript from label-100.xlsx",
            "asr_role":"faster-whisper small.en word timestamps only; normalized LCS matching",
            "aggregation":"positive word/bin time overlap weighted mean; zero vector if no overlap",
            "time_axis":"effective_t1-effective_t0; 50 bins in effective local seconds",
            "time_reliability":"ASR-assisted and interpolated intervals are unverified; proportional fallback is not successful word-level alignment"}
    SCHEMA_OUT.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")
    finalize_existing()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing",action="store_true",help="Audit already written CSVs and build the typical-sample figure")
    parser.add_argument("--enrich-fallback-matches",action="store_true",help="Complete LCS counterpart columns in this run's fallback word CSV")
    args = parser.parse_args()
    if args.finalize_existing and args.enrich_fallback_matches:
        parser.error("Choose one recovery action")
    if args.finalize_existing:
        finalize_existing()
    elif args.enrich_fallback_matches:
        enrich_fallback_matches()
    else:
        main()
