# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Audit Q3 Attachment 4 key bins against raw text, unaligned features and media.

This script never modifies Task13--17 outputs. ASR times are marked as proxies;
they are not presented as verified timestamps of aligned audio/visual features.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q3/evidence"
OUT = TASK / "outputs"
A4 = ROOT / "data/raw/E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件"
UPSTREAM = ROOT / "results/q3/inference/outputs"
SOURCE_CSV = UPSTREAM / "q3_attachment4_predictions_and_explanations.csv"
VIDEO_DIR = A4 / "对齐版本/videos"
WORD_RE = re.compile(r"[\w]+(?:[’'][\w]+)*", re.UNICODE)
EXAMPLES = ("16", "05", "07", "13")


def raw_words(raw: str) -> list[dict]:
    return [
        {"text": match.group(), "start": match.start(), "end": match.end()}
        for match in WORD_RE.finditer(raw)
    ]


def normalized_word(word: str) -> str:
    return "".join(ch for ch in word.lower().replace("’", "'") if ch.isalnum())


def media_info(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = int(stream.frames or 0)
        duration = (
            container.duration / 1_000_000
            if container.duration is not None
            else float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        )
    if fps <= 0 or frames <= 0 or duration <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {"fps": fps, "frames": frames, "duration_sec": duration}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_matches(aligned: np.ndarray, unaligned: np.ndarray, content_end: int) -> dict[int, dict]:
    results = {}
    unaligned = np.asarray(unaligned)
    if unaligned.ndim != 2 or aligned.shape[1] != unaligned.shape[1]:
        raise ValueError("Feature dimensions do not match")
    for bin_index in range(1, content_end):
        vector = np.asarray(aligned[bin_index])
        if not np.any(vector):
            results[bin_index] = {"status": "all_zero_feature", "rows": []}
            continue
        exact = np.flatnonzero(np.all(unaligned == vector, axis=1)).tolist()
        status = "unique_exact" if len(exact) == 1 else "ambiguous_exact" if exact else "unmatched"
        results[bin_index] = {"status": status, "rows": exact}
    return results


def asr_word_map(raw: str, video_path: Path, model: WhisperModel, duration: float) -> tuple[list[dict], dict]:
    words = raw_words(raw)
    segments, _ = model.transcribe(
        str(video_path), language="en", word_timestamps=True, beam_size=5, vad_filter=False
    )
    heard = [word for segment in segments for word in (segment.words or [])]
    raw_norm = [normalized_word(word["text"]) for word in words]
    heard_norm = [normalized_word(word.word) for word in heard]
    matcher = SequenceMatcher(None, raw_norm, heard_norm, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            raw_index = block.a + offset
            heard_word = heard[block.b + offset]
            start = float(heard_word.start)
            end = float(heard_word.end)
            if 0 <= start < end <= duration + 0.05:
                words[raw_index]["asr_start"] = start
                words[raw_index]["asr_end"] = min(end, duration)
    matched = sum("asr_start" in word for word in words)
    return words, {
        "raw_word_count": len(words),
        "asr_word_count": len(heard),
        "direct_exact_word_matches": matched,
        "direct_exact_word_coverage": matched / len(words) if words else 0.0,
        "asr_text": "".join(word.word for word in heard).strip(),
    }


def interval_proxy(words: list[dict], char_start: int, char_end: int) -> tuple[float | None, float | None, str]:
    touched = [word for word in words if word["start"] < char_end and char_start < word["end"]]
    if not touched:
        return None, None, "no_transcript_word"
    if any("asr_start" not in word for word in touched):
        return None, None, "not_all_words_matched_to_asr"
    return min(word["asr_start"] for word in touched), max(word["asr_end"] for word in touched), "asr_proxy_only"


def bounded_interpolation(words: list[dict]) -> list[dict]:
    result = [dict(word) for word in words]
    direct = [index for index, word in enumerate(result) if "asr_start" in word]
    for left, right in zip(direct, direct[1:]):
        missing = right - left - 1
        gap_start = result[left]["asr_end"]
        gap_end = result[right]["asr_start"]
        if not (1 <= missing <= 3 and gap_start < gap_end and gap_end - gap_start <= 2.0):
            continue
        weights = [max(1, len(normalized_word(result[index]["text"]))) for index in range(left + 1, right)]
        total = sum(weights)
        cursor = gap_start
        for index, weight in zip(range(left + 1, right), weights):
            end = cursor + (gap_end - gap_start) * weight / total
            result[index]["asr_start"] = cursor
            result[index]["asr_end"] = end
            result[index]["time_source"] = "bounded_interpolation"
            cursor = end
    return result


def review_time(words: list[dict], char_start: int, char_end: int) -> tuple[float | None, float | None, str]:
    touched = [word for word in words if word["start"] < char_end and char_start < word["end"]]
    if not touched or any("asr_start" not in word for word in touched):
        return None, None, "unavailable"
    quality = "bounded_interpolation" if any(word.get("time_source") == "bounded_interpolation" for word in touched) else "direct_asr_word_match"
    return min(word["asr_start"] for word in touched), max(word["asr_end"] for word in touched), quality


def video_frame_times(path: Path) -> list[tuple[int, float]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [(index, float(frame.time)) for index, frame in enumerate(container.decode(stream)) if frame.time is not None]


def read_video_frame(path: Path, target_index: int) -> Image.Image:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == target_index:
                return frame.to_image()
    raise IndexError(f"Frame {target_index} missing: {path}")


def interval_feature_summary(matches: dict[int, dict], start: int, end: int) -> dict:
    items = [matches[index] for index in range(start, end)]
    statuses = Counter(item["status"] for item in items)
    row_lists = [item["rows"] for item in items]
    unique_rows = [item["rows"][0] for item in items if item["status"] == "unique_exact"]
    monotonic = len(unique_rows) == len(items) and unique_rows == sorted(unique_rows)
    return {
        "feature_match_status": "all_unique_exact" if statuses.get("unique_exact", 0) == len(items) else "partial_or_unmatched",
        "feature_match_counts": dict(statuses),
        "unaligned_candidate_rows_by_bin": row_lists,
        "unaligned_unique_rows_monotonic": monotonic,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_interval(rows: list[dict], importance: list[float], require_alnum: bool = False) -> dict | None:
    eligible = [row for row in rows if not require_alnum or row["raw_text_excerpt_has_alnum"] is True]
    if not eligible:
        return None
    return max(eligible, key=lambda row: sum(importance[row["bin_start_inclusive"]:row["bin_end_exclusive"]]))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    source_hash_before = sha256_file(SOURCE_CSV)
    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        upstream_rows = list(csv.DictReader(handle))
    expected_ids = [f"{number:02d}" for number in range(1, 21)]
    if [row["sample_id"] for row in upstream_rows] != expected_ids:
        raise ValueError("Task17 sample IDs are not exactly 01--20 in order")
    for sample_id in expected_ids:
        for source in (
            A4 / "对齐版本" / f"{sample_id}.pkl",
            A4 / "未对齐版本" / f"{sample_id}.pkl",
            VIDEO_DIR / f"{sample_id}.mp4",
        ):
            if not source.is_file():
                raise FileNotFoundError(source)

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "data/models/bert-base-uncased"), use_fast=True, local_files_only=True)
    asr_model = WhisperModel(str(ROOT / "data/models/whisper-base.en"), device="cpu", compute_type="int8", local_files_only=True)
    interval_rows = []
    sample_rows = []
    cards = ["# Task18 附件4原始证据映射卡", "", "verified 表示原始文本字符映射经 token ID 逐位核对；ASR 时间只是语音代理时间，不是对齐音视频特征的已核实媒体时间。", ""]
    all_bin_counts = {"Audio": Counter(), "Visual": Counter()}

    for upstream in upstream_rows:
        sample_id = upstream["sample_id"]
        with (A4 / "对齐版本" / f"{sample_id}.pkl").open("rb") as handle:
            aligned = pickle.load(handle)
        with (A4 / "未对齐版本" / f"{sample_id}.pkl").open("rb") as handle:
            unaligned = pickle.load(handle)
        if str(aligned["id"]).zfill(2) != sample_id or str(unaligned["id"]).zfill(2) != sample_id:
            raise ValueError(f"Input ID mismatch for {sample_id}")
        raw = str(aligned["raw_text"])
        if raw != str(unaligned["raw_text"]) or raw != upstream["raw_text"]:
            raise ValueError(f"Raw text mismatch for {sample_id}")
        if not np.array_equal(aligned["text_bert"], unaligned["text_bert"]):
            raise ValueError(f"text_bert mismatch for {sample_id}")
        bert = np.asarray(aligned["text_bert"])
        valid_length = int(bert[1].sum())
        content_end = valid_length - 1
        tokenized = tokenizer(raw, truncation=True, max_length=50, return_offsets_mapping=True)
        token_ids_match = tokenized["input_ids"] == bert[0, :valid_length].tolist()
        attention_match = tokenized["attention_mask"] == bert[1, :valid_length].tolist()
        text_verified = bool(token_ids_match and attention_match and len(tokenized["offset_mapping"]) == valid_length)
        offsets = tokenized["offset_mapping"] if text_verified else None
        video = VIDEO_DIR / f"{sample_id}.mp4"
        media = media_info(video)
        asr_words, asr_stats = asr_word_map(raw, video, asr_model, media["duration_sec"])
        review_words = bounded_interpolation(asr_words)
        frame_times = video_frame_times(video)
        matches = {}
        for modality, key in (("Audio", "audio"), ("Visual", "vision")):
            matches[modality] = feature_matches(np.asarray(aligned[key]), np.asarray(unaligned[key]), content_end)
            all_bin_counts[modality].update(value["status"] for value in matches[modality].values())

        counts = Counter()
        example_lines = [f"## 样本 {sample_id}", "", f"- 预测：{upstream['pred_polarity']}；强度 {float(upstream['pred_intensity']):.3f}。", f"- Text/Audio/Visual 贡献：{float(upstream['text_contribution']):.3f}/{float(upstream['audio_contribution']):.3f}/{float(upstream['visual_contribution']):.3f}。", f"- BERT token ID 逐位一致：{text_verified}；ASR 原文词直接匹配：{asr_stats['direct_exact_word_matches']}/{asr_stats['raw_word_count']}。", ""]
        for modality, field in (("Text", "text"), ("Audio", "audio"), ("Visual", "visual")):
            intervals = json.loads(upstream[f"{field}_key_intervals"])
            positions = set(json.loads(upstream[f"{field}_key_positions"]))
            reconstructed = {index for start, end in intervals for index in range(start, end)}
            if positions != reconstructed:
                raise ValueError(f"Task17 interval/position mismatch: {sample_id} {modality}")
            for start, end in intervals:
                if not (1 <= start < end <= content_end):
                    raise ValueError(f"Key interval outside content: {sample_id} {modality} [{start},{end})")
                char_start = min((offsets[index][0] for index in range(start, end)), default=None) if text_verified else None
                char_end = max((offsets[index][1] for index in range(start, end)), default=None) if text_verified else None
                excerpt = raw[char_start:char_end] if text_verified and char_start is not None and char_end is not None else ""
                valid_chars = bool(text_verified and char_start is not None and char_end is not None and 0 <= char_start < char_end <= len(raw) and excerpt)
                excerpt_has_alnum = bool(valid_chars and any(character.isalnum() for character in excerpt))
                proxy_start, proxy_end, proxy_status = interval_proxy(asr_words, char_start, char_end) if valid_chars else (None, None, "token_mapping_unverified")
                review_start, review_end, review_quality = review_time(review_words, char_start, char_end) if valid_chars and excerpt_has_alnum else (None, None, "unavailable")
                visual_zero = modality == "Visual" and not np.any(aligned["vision"])
                if visual_zero:
                    review_start, review_end, review_quality = None, None, "visual_all_zero"
                frame_index = ""
                frame_time = ""
                if modality == "Visual" and review_start is not None and frame_times:
                    midpoint = (review_start + review_end) / 2
                    nearest, nearest_time = min(frame_times, key=lambda pair: abs(pair[1] - midpoint))
                    if abs(nearest_time - midpoint) <= 0.25:
                        frame_index = nearest
                        frame_time = round(nearest_time, 3)
                feature = (
                    {"feature_match_status": "not_applicable", "feature_match_counts": {}, "unaligned_candidate_rows_by_bin": [], "unaligned_unique_rows_monotonic": None}
                    if modality == "Text" else interval_feature_summary(matches[modality], start, end)
                )
                if visual_zero:
                    feature["feature_match_status"] = "visual_all_zero_no_content_evidence"
                raw_status = "verified" if modality == "Text" and valid_chars else "unavailable"
                counts[f"{modality}_{raw_status}"] += 1
                if proxy_status == "asr_proxy_only":
                    counts[f"{modality}_asr_proxy"] += 1
                interval_rows.append({
                    "sample_id": sample_id,
                    "modality": modality,
                    "bin_start_inclusive": start,
                    "bin_end_exclusive": end,
                    "raw_evidence_status": raw_status,
                    "raw_text_char_start": char_start if valid_chars else "",
                    "raw_text_char_end": char_end if valid_chars else "",
                    "raw_text_excerpt": excerpt if valid_chars else "",
                    "raw_text_excerpt_has_alnum": excerpt_has_alnum if modality == "Text" else "",
                    "asr_proxy_start_sec": round(proxy_start, 3) if proxy_start is not None else "",
                    "asr_proxy_end_sec": round(proxy_end, 3) if proxy_end is not None else "",
                    "asr_proxy_status": proxy_status,
                    "review_start_sec": round(review_start, 3) if review_start is not None else "",
                    "review_end_sec": round(review_end, 3) if review_end is not None else "",
                    "review_time_quality": review_quality,
                    "review_frame_index": frame_index,
                    "review_frame_pts_sec": frame_time,
                    "review_basis": "official_aligned_position+bert_offset+asr_word_time" if review_start is not None else "",
                    "verified_media_start_sec": "",
                    "verified_media_end_sec": "",
                    "verified_frame_index": "",
                    "feature_match_status": feature["feature_match_status"],
                    "feature_match_counts": json.dumps(feature["feature_match_counts"], ensure_ascii=False),
                    "unaligned_candidate_rows_by_bin": json.dumps(feature["unaligned_candidate_rows_by_bin"]),
                    "unaligned_unique_rows_monotonic": feature["unaligned_unique_rows_monotonic"],
                    "visual_all_zero": visual_zero,
                })
                if sample_id in EXAMPLES:
                    detail = f"- {modality} [{start},{end})："
                    if raw_status == "verified":
                        detail += f"原文 `{excerpt}`（字符 [{char_start},{char_end})，verified）"
                        if not excerpt_has_alnum:
                            detail += "；仅标点，不作为有语义的文本证据"
                    else:
                        detail += f"原始媒体定位 unavailable；未对齐特征匹配 {feature['feature_match_status']}"
                    if proxy_start is not None:
                        detail += f"；ASR 代理语音段 {proxy_start:.3f}–{proxy_end:.3f}s（不等于已核实特征时间）"
                    if modality == "Visual" and frame_index != "":
                        detail += f"；视频回看候选帧 #{frame_index}（PTS {frame_time:.3f}s，{review_quality}）"
                    example_lines.append(detail + "。")
        if sample_id in EXAMPLES:
            if not np.any(aligned["vision"]):
                example_lines.append("- 视觉特征全零：任何视觉 bin 均不可作为真实视觉内容证据。")
            example_lines.extend(["", ""])
            cards.extend(example_lines)
        sample_rows.append({
            "sample_id": sample_id,
            "pred_polarity": upstream["pred_polarity"],
            "pred_intensity": upstream["pred_intensity"],
            "dominant_modality": upstream["dominant_modality"],
            "text_token_ids_match": token_ids_match,
            "text_attention_match": attention_match,
            "text_raw_mapping_verified": text_verified and counts["Text_unavailable"] == 0,
            "text_verified_intervals": counts["Text_verified"],
            "text_total_intervals": counts["Text_verified"] + counts["Text_unavailable"],
            "audio_verified_media_intervals": 0,
            "audio_total_intervals": counts["Audio_unavailable"],
            "visual_verified_frame_intervals": 0,
            "visual_total_intervals": counts["Visual_unavailable"],
            "audio_asr_proxy_intervals": counts["Audio_asr_proxy"],
            "visual_asr_proxy_intervals": counts["Visual_asr_proxy"],
            "asr_direct_word_coverage": round(asr_stats["direct_exact_word_coverage"], 6),
            "audio_unaligned_length_field": int(unaligned["audio_lengths"]),
            "visual_unaligned_length_field": int(unaligned["vision_lengths"]),
            "audio_nonzero_beyond_declared_length": bool(np.any(np.asarray(unaligned["audio"])[int(unaligned["audio_lengths"]):])),
            "visual_nonzero_beyond_declared_length": bool(np.any(np.asarray(unaligned["vision"])[int(unaligned["vision_lengths"]):])),
            "video_duration_sec": round(media["duration_sec"], 6),
            "video_fps": media["fps"],
            "video_frame_count": media["frames"],
            "visual_all_zero": not np.any(aligned["vision"]),
            "asr_text": asr_stats["asr_text"],
        })

    write_csv(OUT / "q3_raw_evidence_intervals.csv", interval_rows, list(interval_rows[0]))
    write_csv(OUT / "q3_raw_evidence_samples.csv", sample_rows, list(sample_rows[0]))
    (OUT / "q3_raw_evidence_example_cards.md").write_text("\n".join(cards), encoding="utf-8")
    enhanced_rows = []
    for upstream in upstream_rows:
        sample_id = upstream["sample_id"]
        own = [row for row in interval_rows if row["sample_id"] == sample_id]
        parts = [
            f"预测{upstream['pred_polarity']}，强度{float(upstream['pred_intensity']):.3f}",
            f"模态作用Text/Audio/Visual={float(upstream['text_contribution']):.3f}/{float(upstream['audio_contribution']):.3f}/{float(upstream['visual_contribution']):.3f}",
        ]
        if upstream["low_modality_margin"].lower() == "true":
            parts.append("前两模态作用接近")
        else:
            parts.append(f"主要参考{upstream['dominant_modality']}")
        text_row = top_interval(
            [row for row in own if row["modality"] == "Text"],
            json.loads(upstream["text_global_importance_50"]),
            require_alnum=True,
        )
        if text_row:
            parts.append(f"原文关键片段“{text_row['raw_text_excerpt']}”（字符{text_row['raw_text_char_start']}:{text_row['raw_text_char_end']}）")
        for modality, label, field in (("Audio", "语音", "audio"), ("Visual", "视觉", "visual")):
            top = top_interval(
                [row for row in own if row["modality"] == modality],
                json.loads(upstream[f"{field}_global_importance_50"]),
            )
            if not top:
                continue
            if top["visual_all_zero"]:
                parts.append("视觉特征全零，无可用视觉内容证据")
                continue
            bin_range = f"[{top['bin_start_inclusive']},{top['bin_end_exclusive']})"
            if top["review_time_quality"] in ("direct_asr_word_match", "bounded_interpolation"):
                quality = "ASR直接词匹配" if top["review_time_quality"] == "direct_asr_word_match" else "锚点间插值"
                locator = f"{float(top['review_start_sec']):.2f}–{float(top['review_end_sec']):.2f}秒"
                if modality == "Visual" and top["review_frame_index"] != "":
                    locator += f"，视频帧#{top['review_frame_index']}（PTS {float(top['review_frame_pts_sec']):.2f}秒）"
                parts.append(f"{label}关键位置{bin_range}的回看参考：{locator}，{quality}")
            else:
                parts.append(f"{label}关键位置{bin_range}，原视频时间未可靠对齐")
        parts.append("回看时间依据官方对齐位置及语音词对齐估计；非原始特征提取时间戳")
        enhanced_rows.append({
            "sample_id": sample_id,
            "polarity": upstream["pred_polarity"],
            "intensity": upstream["pred_intensity"],
            "explanation": "；".join(parts) + "。",
        })
    write_csv(
        OUT / "attachment4_predictions_with_review_locations.csv",
        enhanced_rows,
        ["sample_id", "polarity", "intensity", "explanation"],
    )
    review_frames = [
        row for row in interval_rows
        if row["sample_id"] in ("05", "07") and row["modality"] == "Visual" and row["review_frame_index"] != ""
    ]
    if review_frames:
        fig_dir = TASK / "figures"
        fig_dir.mkdir(exist_ok=True)
        width, height = 320, 220
        sheet = Image.new("RGB", (width * len(review_frames), height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, row in enumerate(review_frames):
            frame = read_video_frame(VIDEO_DIR / f"{row['sample_id']}.mp4", int(row["review_frame_index"]))
            frame.thumbnail((width, height - 35))
            sheet.paste(frame, (width * column + (width - frame.width) // 2, 0))
            draw.text(
                (width * column + 7, height - 30),
                f"ID {row['sample_id']} bin [{row['bin_start_inclusive']},{row['bin_end_exclusive']})  {row['review_frame_pts_sec']}s",
                fill="black",
            )
        sheet.save(fig_dir / "q3_visual_review_candidates_05_07.png")
    rate_hypothesis_check = {}
    for modality, rate in (("Audio", 20.0), ("Visual", 15.0)):
        comparable = [
            row for row in interval_rows
            if row["modality"] == modality
            and row["asr_proxy_status"] == "asr_proxy_only"
            and row["feature_match_status"] == "all_unique_exact"
        ]
        overlaps = 0
        center_gaps = []
        for row in comparable:
            indexes = [index for group in json.loads(row["unaligned_candidate_rows_by_bin"]) for index in group]
            estimated_start = min(indexes) / rate
            estimated_end = (max(indexes) + 1) / rate
            proxy_start = float(row["asr_proxy_start_sec"])
            proxy_end = float(row["asr_proxy_end_sec"])
            overlaps += max(estimated_start, proxy_start) < min(estimated_end, proxy_end)
            center_gaps.append(abs((estimated_start + estimated_end - proxy_start - proxy_end) / 2))
        rate_hypothesis_check[modality] = {
            "hypothesized_hz": rate,
            "comparable_intervals": len(comparable),
            "row_time_vs_asr_proxy_overlap_count": overlaps,
            "median_center_gap_sec": float(np.median(center_gaps)) if center_gaps else None,
            "interpretation": "Exploratory falsification of direct row/rate mapping, not a timestamp validation",
        }
    summary = {
        "sample_count": len(sample_rows),
        "interval_count": len(interval_rows),
        "text_verified_samples": sum(row["text_raw_mapping_verified"] for row in sample_rows),
        "text_verified_intervals": sum(row["raw_evidence_status"] == "verified" and row["modality"] == "Text" for row in interval_rows),
        "text_total_intervals": sum(row["modality"] == "Text" for row in interval_rows),
        "text_punctuation_only_intervals": sum(row["modality"] == "Text" and not row["raw_text_excerpt_has_alnum"] for row in interval_rows),
        "audio_verified_media_intervals": 0,
        "audio_total_intervals": sum(row["modality"] == "Audio" for row in interval_rows),
        "visual_verified_frame_intervals": 0,
        "visual_total_intervals": sum(row["modality"] == "Visual" for row in interval_rows),
        "audio_asr_proxy_intervals": sum(row["modality"] == "Audio" and row["asr_proxy_status"] == "asr_proxy_only" for row in interval_rows),
        "visual_asr_proxy_intervals": sum(row["modality"] == "Visual" and row["asr_proxy_status"] == "asr_proxy_only" for row in interval_rows),
        "official_position_review_locations": {
            modality: {
                "direct_asr": sum(row["modality"] == modality and row["review_time_quality"] == "direct_asr_word_match" for row in interval_rows),
                "bounded_interpolation": sum(row["modality"] == modality and row["review_time_quality"] == "bounded_interpolation" for row in interval_rows),
                "unavailable": sum(row["modality"] == modality and row["review_time_quality"] in ("unavailable", "visual_all_zero") for row in interval_rows),
                "candidate_video_frames": sum(row["modality"] == modality and row["review_frame_index"] != "" for row in interval_rows),
            }
            for modality in ("Text", "Audio", "Visual")
        },
        "feature_bin_match_counts": {name: dict(counts) for name, counts in all_bin_counts.items()},
        "direct_rate_mapping_hypothesis_check": rate_hypothesis_check,
        "visual_all_zero_ids": [row["sample_id"] for row in sample_rows if row["visual_all_zero"]],
        "visual_nonzero_beyond_declared_length_ids": [row["sample_id"] for row in sample_rows if row["visual_nonzero_beyond_declared_length"]],
        "audio_nonzero_beyond_declared_length_ids": [row["sample_id"] for row in sample_rows if row["audio_nonzero_beyond_declared_length"]],
        "all_verified_media_fields_blank": all(
            not row["verified_media_start_sec"] and not row["verified_media_end_sec"] and not row["verified_frame_index"]
            for row in interval_rows
        ),
        "timestamp_gap": "Neither unaligned feature rows nor audio_lengths/vision_lengths have verified row-to-media-time/frame metadata. ASR word times are transcript proxies only.",
        "upstream_predictions_sha256_before": source_hash_before,
        "upstream_predictions_sha256_after": sha256_file(SOURCE_CSV),
    }
    summary["upstream_predictions_modified"] = summary["upstream_predictions_sha256_before"] != summary["upstream_predictions_sha256_after"]
    (OUT / "q3_raw_evidence_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
