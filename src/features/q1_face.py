"""Add MediaPipe facial geometry to the existing Q1 vision bins only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
import csv
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q1/face"
BASE_TASK = ROOT / "data/intermediate/q1/base_final"
BASE_FINAL = BASE_TASK / "final"
TASK2_OUT = ROOT / "results/q1/media/outputs"
OUT, FINAL, FIG, MODEL_DIR = TASK / "outputs", TASK / "final", TASK / "figures", TASK / "models"
MANIFEST = TASK2_OUT / "manifest_100.csv"
BASE_VISION = TASK2_OUT / "vision_features_50bin.csv"
BASE_MULTIMODAL = TASK2_OUT / "multimodal_features_50bin.csv"
TEXT = TASK2_OUT / "text_features_50bin.csv"
AUDIO = TASK2_OUT / "audio_features_50bin.csv"
MODEL = MODEL_DIR / "face_landmarker.task"
FACE_CACHE = OUT / "_face_features_checkpoint.csv"
FACE_CACHE_META = OUT / "_face_features_checkpoint.json"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
K = 50
POSITIONS = (0.25, 0.50, 0.75)
EPS = 1e-8

STATIC_FIELDS = [
    "face_mouth_open_ratio", "face_mouth_corner_curvature", "face_ear_left",
    "face_ear_right", "face_ear_mean", "face_brow_eye_left",
    "face_brow_eye_right", "face_brow_eye_mean",
]
MOTION_FIELDS = ["face_motion_mean", "face_motion_max"]
FACE_FIELDS = STATIC_FIELDS + MOTION_FIELDS
STATUS_FIELDS = ["face_present", "facial_valid_frame_count", "face_motion_valid"]
ALL_NEW_FIELDS = FACE_FIELDS + STATUS_FIELDS
LOW_LEVEL_FIELDS = ["brightness_mean", "brightness_std", "contrast_mean", "saturation_mean",
                    "saturation_std", "sharpness_mean", "motion_strength", "vision_present",
                    "valid_frame_count", "vision_reason"]

# MediaPipe Face Mesh landmark IDs (468-point topology, available in the 478-point task output).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOTION_LANDMARKS = tuple(sorted(set(
    (13, 14, 61, 291) + LEFT_EYE + RIGHT_EYE +
    (105, 66, 63, 159, 158, 157, 334, 296, 293, 386, 385, 384)
)))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require_inputs():
    missing = [str(p) for p in [MANIFEST, BASE_VISION, BASE_MULTIMODAL, TEXT, AUDIO,
                                 BASE_FINAL / "q1_feature_schema.json",
                                 BASE_FINAL / "q1_paper_ready_method_cn.md",
                                 BASE_FINAL / "q1_mathematical_model.md", MODEL] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Task03-Fix02 required input(s) missing; refusing to reconstruct them: " + ", ".join(missing))


def normalized_landmarks(landmarks, width, height):
    """Center on the eye midpoint, rotate the eye line horizontal, scale eye distance to one."""
    p = np.asarray([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64)
    if p.shape[0] < 468 or not np.isfinite(p[:468]).all():
        return None
    eye_vec = p[263] - p[33]
    eye_dist = float(np.linalg.norm(eye_vec))
    if not math.isfinite(eye_dist) or eye_dist <= EPS:
        return None
    center = (p[33] + p[263]) / 2.0
    theta = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
    c, s = math.cos(theta), math.sin(theta)
    # Row-vector form of a -theta rotation.
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return ((p - center) @ rotation) / eye_dist


def distance(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def static_geometry(p):
    mouth_h = distance(p, 61, 291)
    mor = distance(p, 13, 14) / (mouth_h + EPS)
    lip_center_y = (p[13, 1] + p[14, 1]) / 2.0
    corner_mean_y = (p[61, 1] + p[291, 1]) / 2.0
    corner = (lip_center_y - corner_mean_y)  # inter-ocular distance is one after normalization
    ear_l = (distance(p, 160, 144) + distance(p, 158, 153)) / (2.0 * distance(p, 33, 133) + EPS)
    ear_r = (distance(p, 385, 380) + distance(p, 387, 373)) / (2.0 * distance(p, 362, 263) + EPS)
    brow_l = np.mean([distance(p, 105, 159), distance(p, 66, 158), distance(p, 63, 157)])
    brow_r = np.mean([distance(p, 334, 386), distance(p, 296, 385), distance(p, 293, 384)])
    return {
        "face_mouth_open_ratio": mor,
        "face_mouth_corner_curvature": corner,
        "face_ear_left": ear_l,
        "face_ear_right": ear_r,
        "face_ear_mean": (ear_l + ear_r) / 2.0,
        "face_brow_eye_left": float(brow_l),
        "face_brow_eye_right": float(brow_r),
        "face_brow_eye_mean": float((brow_l + brow_r) / 2.0),
    }


def empty_face_row(sample_id, bin_id):
    row = {"sample_id": sample_id, "bin_id": bin_id}
    row.update({field: 0.0 for field in FACE_FIELDS})
    row.update({"face_present": 0, "facial_valid_frame_count": 0, "face_motion_valid": 0})
    return row


def detect_one(detector, rgb):
    import mediapipe as mp
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(image)
    if not result.face_landmarks:
        return None
    # num_faces=1 is fixed in the FaceLandmarker options: at most the primary face is handled.
    height, width = rgb.shape[:2]
    return normalized_landmarks(result.face_landmarks[0], width, height)


def extract_faces(manifest):
    import av
    import cv2
    import mediapipe as mp

    cv2.setNumThreads(1)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    rows, sample_rows, issues = [], [], []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i, rec in enumerate(manifest.to_dict("records"), start=1):
            sid = str(rec["sample_id"])
            duration = float(rec["duration_sec"])
            t0 = float(rec["effective_t0"])
            edges = np.linspace(0.0, duration, K + 1)
            targets = [t0 + edges[k] + frac * (edges[k + 1] - edges[k])
                       for k in range(K) for frac in POSITIONS]
            detected = [[] for _ in range(K)]
            chosen_n = 0
            target_i = 0
            last_rgb = None
            try:
                path = resolve_recorded_path(str(rec["video_path"]))
                with av.open(str(path)) as container:
                    streams = [s for s in container.streams if s.type == "video"]
                    if not streams:
                        raise RuntimeError("no video stream")
                    stream = streams[0]
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            continue
                        last_rgb = frame.to_ndarray(format="rgb24")
                        frame_time = float(frame.pts * stream.time_base)
                        while target_i < len(targets) and targets[target_i] <= frame_time:
                            p = detect_one(detector, last_rgb)
                            if p is not None:
                                detected[target_i // 3].append((target_i % 3, p))
                            chosen_n += 1
                            target_i += 1
                # Match the existing Task02 Vision sampler's end-of-stream rule: use the last decoded
                # frame for any remaining targets when the effective audio/video interval runs longer.
                while target_i < len(targets) and last_rgb is not None:
                    p = detect_one(detector, last_rgb)
                    if p is not None:
                        detected[target_i // 3].append((target_i % 3, p))
                    chosen_n += 1
                    target_i += 1
                if chosen_n < len(targets):
                    issues.append({"sample_id": sid, "issue_type": "video_has_no_decodable_frames",
                                   "message": f"decoded sample positions={chosen_n}/150"})
            except Exception as exc:
                issues.append({"sample_id": sid, "issue_type": "face_video_processing_error", "message": repr(exc)})

            sample_face_bins = sample_motion_bins = sample_valid_frames = 0
            for k, observations in enumerate(detected):
                observations.sort(key=lambda item: item[0])
                points = [p for _, p in observations]
                n = len(points)
                if n:
                    stat = {key: float(np.mean([v[key] for v in map(static_geometry, points)])) for key in STATIC_FIELDS}
                    motion_values = [float(np.mean(np.linalg.norm(a[list(MOTION_LANDMARKS)] - b[list(MOTION_LANDMARKS)], axis=1)))
                                     for a, b in zip(points[:-1], points[1:])]
                    motion_valid = int(len(motion_values) > 0)
                    motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
                    motion_max = float(np.max(motion_values)) if motion_values else 0.0
                else:
                    stat = {key: 0.0 for key in STATIC_FIELDS}
                    motion_valid, motion_mean, motion_max = 0, 0.0, 0.0
                row = {"sample_id": sid, "bin_id": k, **stat,
                       "face_motion_mean": motion_mean, "face_motion_max": motion_max,
                       "face_present": int(n > 0), "facial_valid_frame_count": n,
                       "face_motion_valid": motion_valid}
                rows.append(row)
                sample_face_bins += int(n > 0)
                sample_motion_bins += motion_valid
                sample_valid_frames += n
            sample_rows.append({"sample_id": sid, "bins_with_detected_face": sample_face_bins,
                                "bins_without_detected_face": K - sample_face_bins,
                                "face_bin_coverage": sample_face_bins / K,
                                "facial_valid_frames": sample_valid_frames,
                                "bins_with_valid_face_motion": sample_motion_bins})
            print(f"face {i}/{len(manifest)} {sid}: detected_bins={sample_face_bins}/50", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sample_rows), issues, options


def make_vision_table(old_vision, face):
    keys = ["sample_id", "bin_id"]
    if old_vision.duplicated(keys).any() or face.duplicated(keys).any():
        raise ValueError("duplicate (sample_id, bin_id) in vision or facial feature rows")
    if len(old_vision) != len(face) or set(map(tuple, old_vision[keys].astype(str).to_numpy())) != set(map(tuple, face[keys].astype(str).to_numpy())):
        raise ValueError("facial bins differ from the inherited 50-bin Vision rows")
    if not set(LOW_LEVEL_FIELDS).issubset(old_vision.columns):
        raise ValueError(f"legacy low-level Vision fields missing: {set(LOW_LEVEL_FIELDS)-set(old_vision.columns)}")
    return old_vision.merge(face, on=keys, how="left", validate="one_to_one", sort=False)


def align_existing_modalities(baseline, text_source, audio_source, vision_source):
    """Rebuild the shared table from the original modality CSVs plus baseline index/label columns."""
    keys = ["sample_id", "bin_id"]
    time_cols = ["bin_start_sec", "bin_end_sec", "bin_center_sec", "normalized_time"]
    meta_cols = ["sample_id", "official_label", "bin_id"] + time_cols
    if not set(meta_cols).issubset(baseline.columns):
        raise ValueError("baseline is missing required index/label/time columns")
    base = baseline.sort_values(keys, kind="stable").reset_index(drop=True)
    rebuilt = base[meta_cols].copy()
    groups = [
        ("Text", text_source, [c for c in base.columns if c == "text_present" or c.startswith("text_")]),
        ("Audio", audio_source, [c for c in base.columns if c.startswith(("audio_", "rms_", "zcr_", "spectral_", "mfcc", "f0_"))]),
        ("Vision", vision_source, [c for c in base.columns if c in LOW_LEVEL_FIELDS]),
    ]
    for name, source, columns in groups:
        absent = set(columns) - set(source.columns)
        if absent:
            raise ValueError(f"{name} source CSV missing aligned fields: {sorted(absent)}")
        if source.duplicated(keys).any():
            raise ValueError(f"duplicate key in original {name} feature CSV")
        source_keys = set(map(tuple, source[keys].astype(str).to_numpy()))
        base_keys = set(map(tuple, base[keys].astype(str).to_numpy()))
        if source_keys != base_keys:
            raise ValueError(f"{name} keys differ from the frozen baseline")
        # The source modality tables each carry the shared time coordinates; verify, then use the baseline values.
        probe = base[meta_cols].merge(source[keys + time_cols], on=keys, how="left", validate="one_to_one", suffixes=("_base", "_src"))
        for col in time_cols:
            if not np.allclose(pd.to_numeric(probe[f"{col}_base"], errors="coerce"),
                               pd.to_numeric(probe[f"{col}_src"], errors="coerce"), atol=1e-9, rtol=0, equal_nan=True):
                raise ValueError(f"{name} time axis differs from baseline: {col}")
        addon = source[keys + columns].sort_values(keys, kind="stable")
        rebuilt = rebuilt.merge(addon, on=keys, how="left", validate="one_to_one", sort=False)
    # Preserve the established column order and keep baseline contents byte-for-value equivalent.
    missing = set(base.columns) - set(rebuilt.columns)
    if missing:
        raise ValueError(f"modality re-alignment omitted baseline fields: {sorted(missing)}")
    rebuilt = rebuilt[base.columns]
    for col in base.columns:
        if not base[col].equals(rebuilt[col]):
            # CSV float parsing can vary by a final ulp after joins; only allow this for numeric values.
            if pd.api.types.is_numeric_dtype(base[col]) and np.allclose(
                    pd.to_numeric(base[col], errors="coerce"), pd.to_numeric(rebuilt[col], errors="coerce"),
                    atol=1e-12, rtol=0, equal_nan=True):
                # Retain the exact frozen baseline representation after proving the modality join agrees.
                rebuilt[col] = base[col]
                continue
            raise ValueError(f"reconstructed baseline field changed during re-alignment: {col}")
    return rebuilt


def append_face_columns_preserving_csv(source_path, destination, face, extra_columns):
    """Copy every inherited CSV cell as text and append only the new facial fields."""
    key_cols = ["sample_id", "bin_id"]
    lookup = {}
    for row in face.to_dict("records"):
        key = (str(row["sample_id"]), int(row["bin_id"]))
        if key in lookup:
            raise ValueError(f"duplicate face key while writing CSV: {key}")
        lookup[key] = [str(int(row[c])) if c in STATUS_FIELDS else format(float(row[c]), ".17g")
                       for c in extra_columns]
    destination.parent.mkdir(parents=True, exist_ok=True)
    used = set()
    with source_path.open("r", encoding="utf-8-sig", newline="") as src, destination.open("w", encoding="utf-8-sig", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        header = next(reader)
        if not set(key_cols).issubset(header):
            raise ValueError(f"CSV source lacks join keys: {source_path}")
        if set(extra_columns).intersection(header):
            raise ValueError(f"CSV source already contains facial columns: {source_path}")
        sample_idx, bin_idx = header.index("sample_id"), header.index("bin_id")
        writer.writerow(header + list(extra_columns))
        for line_no, row in enumerate(reader, start=2):
            key = (row[sample_idx], int(float(row[bin_idx])))
            if key not in lookup:
                raise ValueError(f"no face values for {key} in {source_path} row {line_no}")
            used.add(key)
            writer.writerow(row + lookup[key])
    if used != set(lookup):
        raise ValueError(f"some face keys were not written to {destination}: {len(set(lookup)-used)}")


def make_schema(base_schema, aligned):
    schema = json.loads(json.dumps(base_schema))
    schema["n_samples"], schema["n_bins"], schema["n_rows"] = 100, K, len(aligned)
    schema["feature_count"] = len(aligned.columns)
    schema["vision_facial_method"] = {
        "library": "MediaPipe", "version": importlib.metadata.version("mediapipe"),
        "api": "mediapipe.tasks.vision.FaceLandmarker (IMAGE mode)", "num_faces": 1,
        "model_asset_name": MODEL.name, "model_asset_path": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "sample_positions_within_bin": list(POSITIONS), "geometry": "2D normalized landmarks; eye midpoint centering, eye-line rotation, inter-ocular scale=1",
        "blendshapes": False, "training": False,
    }
    defs = {
        "face_mouth_open_ratio": ("Mouth vertical distance (landmarks 13,14) divided by horizontal mouth width (61,291), after geometric normalization; bin mean.", "dimensionless ratio", "face_present"),
        "face_mouth_corner_curvature": ("Vertical position of lip center (13,14) minus mean mouth-corner position (61,291), normalized by inter-ocular distance; bin mean. A geometric descriptor, not a smile/emotion label.", "inter-ocular distance units", "face_present"),
        "face_ear_left": ("Left-eye EAR from landmarks 33,160,158,133,153,144; bin mean.", "dimensionless ratio", "face_present"),
        "face_ear_right": ("Right-eye EAR from landmarks 362,385,387,263,373,380; bin mean.", "dimensionless ratio", "face_present"),
        "face_ear_mean": ("Mean of left and right eye EAR, averaged across valid sampled frames in the bin.", "dimensionless ratio", "face_present"),
        "face_brow_eye_left": ("Mean normalized distances between left brow and upper-eye landmark pairs (105,159), (66,158), (63,157); bin mean.", "inter-ocular distance units", "face_present"),
        "face_brow_eye_right": ("Mean normalized distances between right brow and upper-eye landmark pairs (334,386), (296,385), (293,384); bin mean.", "inter-ocular distance units", "face_present"),
        "face_brow_eye_mean": ("Mean of left and right normalized brow-eye distances, averaged across valid sampled frames.", "inter-ocular distance units", "face_present"),
        "face_motion_mean": ("Mean, within the bin, of average per-landmark Euclidean displacement across consecutive valid sampled face frames after centering, eye-line rotation, and inter-ocular scaling.", "inter-ocular distance units per sampled step", "face_motion_valid"),
        "face_motion_max": ("Maximum, within the bin, of average per-landmark Euclidean displacement across consecutive valid sampled face frames after geometric normalization.", "inter-ocular distance units per sampled step", "face_motion_valid"),
        "face_present": ("1 iff at least one of the three existing sampled frames yielded valid primary-face landmarks in the bin.", "binary", "self"),
        "facial_valid_frame_count": ("Number of the existing 25%, 50%, 75% sampled frames with usable facial landmarks; integer 0–3.", "frames (0–3)", "face_present"),
        "face_motion_valid": ("1 iff at least two sampled frames in this bin yielded usable facial landmarks for displacement calculation.", "binary", "self"),
    }
    existing = {f["field_name"] for f in schema["features"]}
    unexpected = existing.intersection(ALL_NEW_FIELDS)
    if unexpected:
        raise ValueError(f"facial fields already exist in baseline schema: {sorted(unexpected)}")
    for name in ALL_NEW_FIELDS:
        desc, unit, mask = defs[name]
        is_status = name in STATUS_FIELDS
        schema["features"].append({
            "field_name": name, "modality": "vision", "dimension/scalar": 1,
            "dtype": "int64" if is_status else "float64", "definition": desc,
            "unit": unit, "missing_rule": "Zero padding when no valid face is detected; consult face_present. Motion zeros with face_motion_valid=0 are not observed displacement." if name in MOTION_FIELDS else ("Zero padding when face_present=0; not a measured zero." if name in STATIC_FIELDS else "Availability/count status; no temporal row removal."),
            "padding_rule": "Retain every original bin_id 0–49; no face detection failure removes a row.",
            "mask_field": mask, "source_method": "MediaPipe Face Landmarker 2D landmarks; Task03-Fix02",
        })
    # Clarify the inherited Vision definitions without altering Text/Audio schema entries.
    for item in schema["features"]:
        if item["field_name"] in LOW_LEVEL_FIELDS:
            item["source_method"] = "OpenCV low-level Vision statistics inherited unchanged from Task02"
    schema["vision_feature_groups"] = {
        "low_level_statistics": ["brightness_mean", "brightness_std", "contrast_mean", "saturation_mean", "saturation_std", "sharpness_mean", "motion_strength"],
        "facial_geometry": FACE_FIELDS,
        "face_status": STATUS_FIELDS,
    }
    return schema


def validate(manifest, baseline, aligned, mapping, sample_summary, face_issues, model_info):
    keys = ["sample_id", "bin_id"]
    counts = aligned.groupby("sample_id").size()
    face_numeric = aligned[FACE_FIELDS].apply(pd.to_numeric, errors="coerce")
    all_numeric = aligned.select_dtypes(include=[np.number])
    face_nan = int(face_numeric.isna().sum().sum())
    face_inf = int(np.isinf(face_numeric.to_numpy(dtype=float)).sum())
    all_inf = int(np.isinf(all_numeric.to_numpy(dtype=float)).sum())
    expected_nan = {"f0_mean", "f0_std"}
    all_nan_cols = {c: int(n) for c, n in all_numeric.isna().sum().items() if n}
    unexpected_nan = {c: n for c, n in all_nan_cols.items() if c not in expected_nan}
    present = pd.to_numeric(aligned.face_present, errors="coerce").fillna(-1).astype(int)
    frame_count = pd.to_numeric(aligned.facial_valid_frame_count, errors="coerce").fillna(-1).astype(int)
    motion_valid = pd.to_numeric(aligned.face_motion_valid, errors="coerce").fillna(-1).astype(int)
    mask_consistency = bool(((present == 1) == (frame_count > 0)).all() and
                            ((motion_valid == 1) == (frame_count >= 2)).all() and
                            frame_count.between(0, 3).all())
    no_face_padding = bool(np.allclose(aligned.loc[present == 0, FACE_FIELDS].to_numpy(float), 0.0))
    no_motion_padding = bool(np.allclose(aligned.loc[motion_valid == 0, MOTION_FIELDS].to_numpy(float), 0.0))
    baseline_unchanged = True
    for col in baseline.columns:
        if col not in aligned.columns or not baseline[col].equals(aligned[col]):
            baseline_unchanged = False
            break
    per_sample = aligned.groupby("sample_id").agg(face_bins=("face_present", "sum"), motion_bins=("face_motion_valid", "sum"))
    bins_with_face = int(present.sum())
    samples_with_face = int((per_sample.face_bins > 0).sum())
    coverage = bins_with_face / len(aligned) if len(aligned) else 0.0
    # Coverage is explicitly an acceptance gate so a technically successful but sparse detector is not called an enhancement.
    coverage_sufficient = coverage >= 0.50 and samples_with_face >= 80
    expected_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    actual_keys = set(map(tuple, aligned[keys].astype(str).to_numpy()))
    structure = (len(manifest) == 100 and manifest.sample_id.nunique() == 100 and len(aligned) == 5000 and
                 aligned.sample_id.nunique() == 100 and bool(counts.eq(50).all()) and
                 not aligned.duplicated(keys).any() and expected_keys == actual_keys and
                 aligned.groupby("sample_id").bin_id.apply(lambda x: set(pd.to_numeric(x).astype(int)) == set(range(50))).all())
    face_pass = (model_info["model_ran"] and len(face_issues) == 0 and
                 set(ALL_NEW_FIELDS).issubset(aligned.columns) and mask_consistency and no_face_padding and
                 no_motion_padding and face_nan == 0 and face_inf == 0 and coverage_sufficient)
    accepted = bool(structure and face_pass and baseline_unchanged and all_inf == 0 and not unexpected_nan)
    count_distribution = {str(i): int((frame_count == i).sum()) for i in range(4)}
    present_distribution = {"0": int((present == 0).sum()), "1": int((present == 1).sum())}
    motion_distribution = {"0": int((motion_valid == 0).sum()), "1": int((motion_valid == 1).sum())}
    summary = {
        "samples_with_detected_face": samples_with_face,
        "samples_without_any_detected_face": int(len(manifest) - samples_with_face),
        "bins_with_detected_face": bins_with_face,
        "bins_without_detected_face": int(len(aligned) - bins_with_face),
        "face_bin_coverage": coverage,
        "face_present_distribution": present_distribution,
        "facial_valid_frame_count": count_distribution,
        "face_motion_valid_distribution": motion_distribution,
        "bins_with_valid_face_motion": int(motion_valid.sum()),
        "nan_count": face_nan,
        "inf_count": face_inf,
        "face_padding_mask_consistent": mask_consistency and no_face_padding and no_motion_padding,
        "legacy_vision_and_text_audio_unchanged": baseline_unchanged,
        "coverage_rule": "accepted only if face_bin_coverage >= 0.50 and samples_with_detected_face >= 80/100",
        "coverage_sufficient": coverage_sufficient,
        "mediapipe": model_info,
        "processing_issue_count": len(face_issues),
        "processing_issues": face_issues,
        "data_structure_pass": bool(structure),
        "face_pipeline_pass": bool(face_pass),
        "q1_vision_enhancement_accepted": accepted,
    }
    validation = {
        "data_structure_pass": bool(structure),
        "face_pipeline_pass": bool(face_pass),
        "q1_vision_enhancement_accepted": accepted,
        "sample_count": int(manifest.sample_id.nunique()), "row_count": int(len(aligned)),
        "bins_per_sample": 50, "duplicate_keys": int(aligned.duplicated(keys).sum()),
        "bins_per_sample_pass": bool(counts.eq(50).all()),
        "baseline_columns_unchanged": bool(baseline_unchanged),
        "legacy_low_level_vision_columns_present": set(LOW_LEVEL_FIELDS).issubset(aligned.columns),
        "face_mask_padding_consistent": bool(mask_consistency and no_face_padding and no_motion_padding),
        "face_nan_count": face_nan, "face_inf_count": face_inf, "all_numeric_inf_count": all_inf,
        "unexpected_nan_by_field": unexpected_nan, "allowed_nan_by_field": {k:v for k,v in all_nan_cols.items() if k in expected_nan},
        "face_coverage": coverage, "samples_with_face": samples_with_face,
        "samples_without_face": int(100-samples_with_face), "bins_with_face": bins_with_face,
        "bins_without_face": int(5000-bins_with_face), "bins_with_valid_face_motion": int(motion_valid.sum()),
        "facial_valid_frame_count_distribution": count_distribution,
        "face_present_distribution": present_distribution,
        "face_motion_valid_distribution": motion_distribution,
        "text_recomputed": False, "audio_recomputed": False, "timeline_changed": False,
        "face_processing_issues": face_issues,
    }
    return summary, validation


def update_docs(schema, summary, validation, model_info):
    FINAL.mkdir(parents=True, exist_ok=True)
    for name in ("q1_paper_ready_method_cn.md", "q1_mathematical_model.md"):
        source = BASE_FINAL / name
        target = FINAL / name
        text = source.read_text(encoding="utf-8")
        if name == "q1_paper_ready_method_cn.md":
            start, end = "## 1.5 视觉特征", "## 1.6 三模态对齐与缺失编码"
            replacement = r'''## 1.5 视觉特征

视觉模态由低层视觉统计和面部表情几何两组描述量组成。低层视觉部分沿用既有结果，包括整体亮度、对比度、饱和度、清晰度及全局运动强度；本轮不修改其数值。人脸部分在原有每个 bin 的 25%、50%、75% 三个采样帧上运行 MediaPipe Face Landmarker，仅定位一张主要人脸的二维关键点，不进行情绪分类或模型训练。

为降低人脸尺度和轻微旋转对几何量的影响，以左右眼外侧关键点 33、263 的中点为原点，将两点连线旋转至水平，并将双眼距离缩放为 1。设归一化后的第 (l) 个点为 (mathbf p'_l=(x'_l,y'_l))。嘴部开合比定义为

$$
MOR=\frac{\|\mathbf p'_{13}-\mathbf p'_{14}\|_2}{\|\mathbf p'_{61}-\mathbf p'_{291}\|_2+\varepsilon}.
$$

嘴角几何量为上下唇中点与嘴角平均纵坐标之差（此时尺度已由双眼距离归一化）：

$$
C_{mouth}=(y'_{13}+y'_{14})/2-(y'_{61}+y'_{291})/2.
$$

由于 \(\mathbf p'\) 已将双眼距离缩放为 1，上式直接使用归一化后的纵坐标差。

左右眼分别按经典 EAR 定义计算，再取左右均值；眉眼距离分别对左右三组对应关键点的欧氏距离取平均，均在归一化坐标中计算。上述量仅描述嘴部开合、嘴角形态、眼睛开合和眉眼相对位置，不直接编码为快乐、悲伤等情绪类别。

面部运动选取嘴部、左右眼和眉部关键点集合 \(\mathcal E\)。对同一 bin 中相邻有效采样帧 (t,t+1)，计算

$$
M_{face}^{(t)}=\frac{1}{|\mathcal E|}\sum_{l\in\mathcal E}\|\mathbf p'_{l,t+1}-\mathbf p'_{l,t}\|_2,
$$

并在 bin 内汇总其均值与最大值。单帧有脸时静态几何特征仍有效，但运动特征补零并令 `face_motion_valid=0`。无有效人脸时 10 维几何特征全部补零，令 `face_present=0`；`facial_valid_frame_count` 记录 0–3 个成功帧数。任何人脸缺失均不删除原有时间 bin。全局 `motion_strength` 继续保留，与标准化后的局部面部运动特征分别表征整幅画面变化和面部局部变化。

MediaPipe 仅用于人脸关键点定位；其实际版本、API、CPU 执行方式和模型资产路径记录在 `q1_face_feature_summary.json`。'''
        else:
            start, end = "## 5. Vision representation", "## 6. Multimodal sequence and padding"
            replacement = '''## 5. Vision representation

The Vision representation has two groups. The inherited low-level OpenCV statistics—brightness, contrast, saturation, sharpness, and global `motion_strength`—are preserved unchanged. On the same three existing sample positions (25%, 50%, 75%) within each bin, MediaPipe Face Landmarker locates at most one primary face. It does not classify emotion and no model is trained.

For geometric normalization, the midpoint of landmarks 33 and 263 is translated to the origin, their eye line is rotated to horizontal, and the inter-ocular distance is scaled to one. Static features are mouth opening ratio (vertical lip distance / horizontal mouth width), normalized mouth-corner vertical geometry, left/right/mean Eye Aspect Ratio (EAR), and left/right/mean brow-eye distance. They are averaged across valid face frames in a bin. Facial motion is the mean and maximum average landmark displacement across consecutive valid sampled frames after this normalization, using mouth, eye, and brow landmarks. These are geometric descriptors, not hard-coded emotion labels.

The 10 geometry fields are `face_mouth_open_ratio`, `face_mouth_corner_curvature`, `face_ear_left`, `face_ear_right`, `face_ear_mean`, `face_brow_eye_left`, `face_brow_eye_right`, `face_brow_eye_mean`, `face_motion_mean`, and `face_motion_max`. `face_present`, `facial_valid_frame_count`, and `face_motion_valid` distinguish observed values from padding. A bin with no valid face retains all rows and receives zero-padded geometry; when fewer than two valid frames exist, motion fields are zero with `face_motion_valid=0`. The original global `motion_strength` remains a separate feature. MediaPipe version, API, CPU mode, parameters, and model path are recorded in `q1_face_feature_summary.json`.
'''
        if start not in text or end not in text:
            raise ValueError(f"expected Vision section boundary missing from {name}")
        before, tail = text.split(start, 1)
        _, after = tail.split(end, 1)
        target.write_text(before + replacement + "\n" + end + after, encoding="utf-8")

    (FINAL / "q1_feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (FINAL / "q1_face_feature_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (FINAL / "q1_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pass_text = "YES" if validation["data_structure_pass"] else "NO"
    face_text = "YES" if validation["face_pipeline_pass"] else "NO"
    accept_text = "YES" if validation["q1_vision_enhancement_accepted"] else "NO"
    face_cov = summary["face_bin_coverage"]
    summary_md = f'''# Q1 数据结构与人脸特征校验摘要

## 验收结论

- `DATA_STRUCTURE_PASS = {pass_text}`：{validation['sample_count']} 个样本、{validation['row_count']} 行、每样本 50 个 bin、重复键 {validation['duplicate_keys']}。
- `FACE_PIPELINE_PASS = {face_text}`：MediaPipe 实际运行，新增特征及缺失 mask 通过校验。
- `Q1_VISION_ENHANCEMENT_ACCEPTED = {accept_text}`。

## 人脸覆盖

- 有至少一个有效人脸 bin 的样本：{summary['samples_with_detected_face']}/100；无检测人脸样本：{summary['samples_without_any_detected_face']}/100。
- 有效人脸 bin：{summary['bins_with_detected_face']}/5000；无脸 bin：{summary['bins_without_detected_face']}/5000；覆盖率：{face_cov:.2%}。
- `facial_valid_frame_count` 分布（0/1/2/3）：{summary['facial_valid_frame_count']}。
- `face_present` 分布（0/1）：{summary['face_present_distribution']}；`face_motion_valid` 分布（0/1）：{summary['face_motion_valid_distribution']}。
- 运动有效 bin：{summary['bins_with_valid_face_motion']}/5000。
- 面部几何字段 NaN：{summary['nan_count']}；Inf：{summary['inf_count']}；缺失补零与 mask 一致：{summary['face_padding_mask_consistent']}。

覆盖充分性判据在运行前固定为：bin 覆盖率至少 50%，且至少 80/100 个样本有检测人脸。该门槛用于避免仅因程序无报错就判定增强有效；实际覆盖量仍应在论文或后续建模中如实说明。

## 实现与输入完整性

- MediaPipe {model_info['mediapipe_version']}，API `{model_info['api']}`，CPU 推理，`num_faces=1`。
- 模型：`{model_info['model_asset_name']}`，本地路径 `{model_info['model_asset_path']}`；SHA-256 `{model_info['model_sha256']}`。
- 使用原有每 bin 三个采样点；Text、Audio、有效时间轴和 50-bin 划分未重算或更改。
- 既有低层 Vision、Text、Audio 和基线非人脸字段逐列一致：{validation['baseline_columns_unchanged']}。
- MediaPipe 视频处理异常数：{summary['processing_issue_count']}。
'''
    (FINAL / "q1_validation_summary.md").write_text(summary_md, encoding="utf-8")

    flow_mmd = '''flowchart TD
    A[100 source clips] --> B[Reuse the existing effective timeline and 50-bin index]
    B --> T[Text: reuse existing 768-d word features]
    B --> U[Audio: reuse existing acoustic features]
    B --> V[Vision: reuse the same 25% / 50% / 75% frames per bin]
    V --> L[Preserve low-level statistics<br/>Brightness · Contrast · Saturation<br/>Sharpness · Global Motion]
    V --> F[MediaPipe Face Landmarker<br/>one primary face · CPU · 2D landmarks]
    F --> G[Normalize landmarks<br/>center · eye-line rotation · inter-ocular scale]
    G --> H[Mouth · Eyes · Brows<br/>10 geometric features + 3 status fields]
    T --> J[Join by sample_id + bin_id]
    U --> J
    L --> J
    H --> J
    J --> M[Retain all 5000 rows<br/>zero padding + explicit face masks]
'''
    (FINAL / "q1_pipeline_flowchart.mmd").write_text(flow_mmd, encoding="utf-8")
    (FINAL / "q1_pipeline_flowchart.md").write_text(
        "# Q1 三模态流程图\n\nText、Audio 与既有 50-bin 时间轴保持不变。Vision 保留原有低层统计，并在相同的三个采样位置增加 MediaPipe 人脸几何特征；按 `(sample_id, bin_id)` 对齐，缺失的人脸特征以 mask 和零填充表示。\n",
        encoding="utf-8")

    report = f'''# Task03-Fix02 视觉人脸几何特征增强执行报告

## 1. 本轮范围

在既有 50-bin Vision 序列的原采样位置补充 MediaPipe 人脸几何特征。未训练模型，未重算 Text/Audio，未改变有效时间轴、bin 边界或标签。

## 2. 输入复用与运行记录

- `TEXT_RECOMPUTED = NO`
- `AUDIO_RECOMPUTED = NO`
- `TIMELINE_CHANGED = NO`
- `VISION_RECOMPUTED = YES`（重建 Vision 输出表；既有低层特征值逐列复用不变，并对相同视频采样位置运行人脸关键点推理）
- `FACIAL_FEATURES_ADDED = YES`
- 原 manifest、Text/Audio 特征、Task02 Vision、Q1 基线对齐表和 schema 均在运行前确认存在；上游文件未覆盖。

MediaPipe：真实版本 {model_info['mediapipe_version']}；API `{model_info['api']}`；CPU；`num_faces=1`；置信度参数均为 0.5；模型 `{model_info['model_asset_name']}`，路径 `{model_info['model_asset_path']}`，SHA-256 `{model_info['model_sha256']}`。

## 3. 覆盖率与特征校验

- 有脸样本：{summary['samples_with_detected_face']}/100；无脸样本：{summary['samples_without_any_detected_face']}/100。
- 有脸 bins：{summary['bins_with_detected_face']}/5000；bin 覆盖率：{summary['face_bin_coverage']:.2%}。
- `facial_valid_frame_count`：{summary['facial_valid_frame_count']}；有效面部运动 bins：{summary['bins_with_valid_face_motion']}。
- `face_present` 分布：{summary['face_present_distribution']}；`face_motion_valid` 分布：{summary['face_motion_valid_distribution']}。
- 新增面部几何字段 NaN={summary['nan_count']}、Inf={summary['inf_count']}；缺失补零与 mask 一致={summary['face_padding_mask_consistent']}。
- 覆盖充分性判据（至少 50% bins 且至少 80 个样本出现人脸）：{summary['coverage_sufficient']}。

## 4. 数据结构

- `DATA_STRUCTURE_PASS = {'YES' if validation['data_structure_pass'] else 'NO'}`
- 样本 {validation['sample_count']}；行数 {validation['row_count']}；每样本 50 bins；重复键 {validation['duplicate_keys']}。
- 基线列（包含 Text、Audio、原 Vision 统计、索引与标签）逐列未变：{validation['baseline_columns_unchanged']}。

## 5. 图片

- `PIPELINE_FIGURE_UPDATED = YES`：Vision 分支现在明确包含低层统计与面部几何两组特征。
- `TEMPORAL_ALIGNMENT_FIGURE_UPDATED = NO`：三模态共用的 50-bin 时间轴和 25%/50%/75% 采样规则未变化，原图没有把 Vision 限定为仅低层统计。
- 覆盖 QC 图：`figures/facial_feature_coverage.png`。

## 6. 最终验收

- `FACE_PIPELINE_PASS = {'YES' if validation['face_pipeline_pass'] else 'NO'}`
- `Q1_VISION_ENHANCEMENT_ACCEPTED = {'YES' if validation['q1_vision_enhancement_accepted'] else 'NO'}`

是否接受由真实运行的 MediaPipe、数据结构、mask/padding、数值检查和覆盖率共同决定，明细见 `final/q1_face_feature_summary.json`、`final/q1_validation.json`。

## 7. 本轮输出

- `outputs/vision_features_50bin.csv`：原低层 Vision 字段与新增人脸特征。
- `outputs/multimodal_features_50bin.csv`：沿用全部基线字段并追加 13 个脸部字段。
- `outputs/facial_feature_summary.csv` / `.json` 与 `final/q1_face_feature_summary.json`：覆盖率统计。
- `final/q1_feature_schema.json`、`final/q1_validation.json`、`final/q1_validation_summary.md`。
- `final/q1_mathematical_model.md`、`final/q1_paper_ready_method_cn.md`、`final/q1_pipeline_flowchart.mmd/.png/.svg`。
- `figures/facial_feature_coverage.png`；模型资产在 `models/face_landmarker.task`。
'''
    (TASK / "task_03_fix02_vision_facial_enhancement_report.md").write_text(report, encoding="utf-8")


def draw_coverage(summary, sample_summary):
    FIG.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((n for n in ["Microsoft YaHei", "Noto Sans SC", "DengXian", "SimHei"] if n in available), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    bins = 5000
    vals = [summary["bins_with_detected_face"], summary["bins_without_detected_face"], summary["bins_with_valid_face_motion"]]
    labels = ["检测到人脸", "未检测到人脸", "面部运动有效"]
    colors = ["#168A83", "#B8C5D1", "#4F6DB8"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    bars = ax.barh(labels, vals, color=colors, height=.58)
    ax.set_xlim(0, bins * 1.12); ax.set_xlabel("bin 数（总计 5000）", color="#34495E")
    ax.set_title("MediaPipe 人脸特征覆盖检查", color="#18324B", fontsize=16, weight="bold", pad=16)
    ax.grid(axis="x", color="#E6EBF0", linewidth=.8); ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(val + bins*.012, bar.get_y()+bar.get_height()/2, f"{val:,}  ({val/bins:.1%})", va="center", color="#263746", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.spines["bottom"].set_color("#CAD4DE")
    ax.tick_params(axis="y", length=0, labelsize=11); ax.tick_params(axis="x", colors="#66788A")
    fig.text(.99,.012,f"样本覆盖：{summary['samples_with_detected_face']}/100  ·  每样本 50 个 bin",ha="right",fontsize=8.5,color="#66788A")
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(FIG / "facial_feature_coverage.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_pipeline():
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((n for n in ["Microsoft YaHei", "Noto Sans SC", "DengXian", "SimHei"] if n in available), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=180)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.set(xlim=(0,13),ylim=(0,7.2)); ax.axis("off")
    navy, ink, teal, orange, blue, edge = "#18324B", "#263746", "#178F88", "#D77A24", "#526DB2", "#CFD9E2"
    ax.text(6.5,6.85,"问题1　三模态特征提取与时序对齐",ha="center",va="center",fontsize=19,weight="bold",color=navy)
    ax.text(6.5,6.48,"Task03-Fix02：仅增强 Vision；Text、Audio 与 50-bin 时间轴沿用既有结果",ha="center",va="center",fontsize=10,color="#66788A")
    def card(x,y,w,h,title,body,color,fill):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04,rounding_size=.12",facecolor=fill,edgecolor=color,linewidth=1.25)
        ax.add_patch(p); ax.text(x+w/2,y+h-.3,title,ha="center",va="center",fontsize=12,weight="bold",color=color)
        ax.text(x+w/2,y+h/2-.13,body,ha="center",va="center",fontsize=9.1,color=ink,linespacing=1.5)
    card(.45,4.9,2.25,1.02,"统一时间轴","100 样本 · 50 bins\n每 bin 沿用 3 个采样位置",navy,"#F5F8FB")
    card(3.25,4.9,2.35,1.02,"Text / Audio","已有特征直接复用\n不重新计算",teal,"#E7F5F2")
    card(9.35,4.9,3.05,1.02,"Vision：原采样帧","每 bin 25% / 50% / 75%\n原低层统计完整保留",blue,"#EBEFF9")
    ax.add_patch(FancyArrowPatch((2.72,5.4),(3.18,5.4),arrowstyle="-|>",mutation_scale=12,color="#66788A",lw=1.3))
    ax.add_patch(FancyArrowPatch((8.95,5.4),(9.28,5.4),arrowstyle="-|>",mutation_scale=12,color="#66788A",lw=1.3))
    card(.72,2.82,5.45,1.48,"低层视觉统计（原值沿用）","亮度 · 对比度 · 饱和度 · 清晰度\n全局运动强度 motion_strength",teal,"#E7F5F2")
    card(6.82,2.82,5.45,1.48,"面部表情几何（新增）","MediaPipe Face Landmarker · 单张主脸 · CPU\n嘴部开合与嘴角 · EAR · 眉眼距离 · 面部运动",orange,"#FBF0E3")
    ax.add_patch(FancyArrowPatch((10.85,4.84),(3.45,4.37),arrowstyle="-|>",connectionstyle="arc3,rad=.12",mutation_scale=12,color=teal,lw=1.2))
    ax.add_patch(FancyArrowPatch((10.85,4.84),(9.55,4.37),arrowstyle="-|>",connectionstyle="arc3,rad=0",mutation_scale=12,color=orange,lw=1.2))
    card(2.15,.78,8.7,1.15,"按 sample_id + bin_id 对齐","保留全部 5000 行；人脸缺失用 face_present / facial_valid_frame_count 标记\n静态/运动几何零填充与状态字段配套",navy,"#F5F8FB")
    ax.add_patch(FancyArrowPatch((3.45,2.76),(5.05,1.98),arrowstyle="-|>",connectionstyle="arc3,rad=-.08",mutation_scale=12,color=teal,lw=1.2))
    ax.add_patch(FancyArrowPatch((9.55,2.76),(7.95,1.98),arrowstyle="-|>",connectionstyle="arc3,rad=.08",mutation_scale=12,color=orange,lw=1.2))
    ax.text(6.5,.36,"MediaPipe 定位关键点；情绪由后续模型学习，不由本步骤硬编码",ha="center",va="center",fontsize=9.5,color="#66788A")
    fig.tight_layout(pad=.3)
    for ext, kw in (("png", {"dpi":400}), ("svg", {})):
        fig.savefig(FINAL / f"q1_pipeline_flowchart.{ext}", bbox_inches="tight", pad_inches=.15, facecolor="white", **kw)
    plt.close(fig)


def main():
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=True); FINAL.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    old_vision = pd.read_csv(BASE_VISION, low_memory=False)
    baseline = pd.read_csv(BASE_MULTIMODAL, low_memory=False)
    text_input = pd.read_csv(TEXT, low_memory=False)
    audio_input = pd.read_csv(AUDIO, low_memory=False)
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Face Landmarker model asset missing or suspiciously small: {MODEL}")
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if FACE_CACHE.is_file() and FACE_CACHE_META.is_file():
        cache_meta = json.loads(FACE_CACHE_META.read_text(encoding="utf-8"))
    else:
        cache_meta = {}
    if cache_meta.get("model_sha256") == model_hash and cache_meta.get("sample_count") == len(manifest):
        face = pd.read_csv(FACE_CACHE, low_memory=False)
        sample_summary = pd.read_csv(OUT / "_face_sample_checkpoint.csv", low_memory=False)
        issues = json.loads((OUT / "_face_issues_checkpoint.json").read_text(encoding="utf-8"))
        print(f"reusing completed face extraction checkpoint: {len(face)} bins", flush=True)
    else:
        face, sample_summary, issues, _options = extract_faces(manifest)
        face.to_csv(FACE_CACHE, index=False, encoding="utf-8-sig")
        sample_summary.to_csv(OUT / "_face_sample_checkpoint.csv", index=False, encoding="utf-8-sig")
        (OUT / "_face_issues_checkpoint.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        FACE_CACHE_META.write_text(json.dumps({"model_sha256": model_hash, "sample_count": len(manifest)}, indent=2), encoding="utf-8")
    vision = make_vision_table(old_vision, face)
    # Keep all inherited text/audio/metadata and low-level vision columns exactly as loaded; append only new fields.
    if set(ALL_NEW_FIELDS).intersection(baseline.columns):
        raise ValueError("baseline already contains some Task03-Fix02 face fields; refusing ambiguous overwrite")
    vision = vision.sort_values(["sample_id", "bin_id"], kind="stable").reset_index(drop=True)
    baseline_sorted = baseline.sort_values(["sample_id", "bin_id"], kind="stable").reset_index(drop=True)
    aligned_base = align_existing_modalities(baseline_sorted, text_input, audio_input, vision)
    aligned = aligned_base.merge(face[["sample_id", "bin_id"] + ALL_NEW_FIELDS],
                                 on=["sample_id", "bin_id"], how="left", validate="one_to_one", sort=False)
    aligned = aligned.sort_values(["sample_id", "bin_id"], kind="stable").reset_index(drop=True)
    schema_base = json.loads((BASE_FINAL / "q1_feature_schema.json").read_text(encoding="utf-8"))
    schema = make_schema(schema_base, aligned)
    model_info = {
        "model_ran": True, "mediapipe_version": importlib.metadata.version("mediapipe"),
        "api": "mediapipe.tasks.vision.FaceLandmarker (IMAGE mode)", "running_mode": "IMAGE",
        "delegate": "CPU (MediaPipe XNNPACK)", "num_faces": 1,
        "min_face_detection_confidence": 0.5, "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5, "output_face_blendshapes": False,
        "output_facial_transformation_matrixes": False,
        "model_asset_name": MODEL.name, "model_asset_path": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "model_source_url": MODEL_URL, "model_sha256": model_hash, "model_asset_bytes": MODEL.stat().st_size,
        "sampling_positions_per_bin": list(POSITIONS), "training": False,
        "python_version": sys.version.split()[0], "opencv_version": __import__("cv2").__version__,
    }
    summary, validation = validate(manifest, baseline_sorted, aligned, None, sample_summary, issues, model_info)
    # Add task-level extraction metadata after actual run; all versions are measured locally.
    summary["sampling_positions_per_bin"] = list(POSITIONS)
    summary["video_sample_count"] = int(len(manifest))
    summary["face_landmarker_input_calls"] = int(len(manifest) * K * len(POSITIONS) - sum(
        150 - int(x["message"].split("=")[-1].split("/")[0])
        for x in issues if "decoded sample positions=" in x.get("message", "")))
    summary["legacy_vision_values_exactly_reused"] = True
    validation["vision_recomputed"] = True
    validation["feature_count"] = int(len(aligned.columns))
    validation["mediapipe_version"] = model_info["mediapipe_version"]
    validation["mediapipe_api"] = model_info["api"]

    # Preserve every inherited source cell exactly as text; append only new facial columns.
    append_face_columns_preserving_csv(BASE_VISION, OUT / "vision_features_50bin.csv", face, ALL_NEW_FIELDS)
    append_face_columns_preserving_csv(BASE_MULTIMODAL, OUT / "multimodal_features_50bin.csv", face, ALL_NEW_FIELDS)
    sample_summary.to_csv(OUT / "facial_feature_summary.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "facial_feature_summary.json", summary)
    # Also emit the explicitly requested final coverage summary.
    write_json(FINAL / "q1_face_feature_summary.json", summary)
    write_json(FINAL / "q1_feature_schema.json", schema)
    write_json(FINAL / "q1_validation.json", validation)
    update_docs(schema, summary, validation, model_info)
    draw_coverage(summary, sample_summary)
    draw_pipeline()
    for checkpoint in (FACE_CACHE, FACE_CACHE_META, OUT / "_face_sample_checkpoint.csv", OUT / "_face_issues_checkpoint.json"):
        checkpoint.unlink(missing_ok=True)
    print(json.dumps({k: summary[k] for k in ["samples_with_detected_face", "samples_without_any_detected_face",
          "bins_with_detected_face", "bins_without_detected_face", "face_bin_coverage",
          "facial_valid_frame_count", "bins_with_valid_face_motion", "nan_count", "inf_count",
          "data_structure_pass", "face_pipeline_pass", "q1_vision_enhancement_accepted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
