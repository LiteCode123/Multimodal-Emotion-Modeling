# 本程序及代码是在人工智能工具辅助下完成的，DeepSeek，DeepSeek-V4.1-Flash，杭州深度求索人工智能基础技术研究有限公司，2026年9月10日。
"""Task05 Q2 continuous missingness simulator and fixed validation grid.

Run from project root: .venv/Scripts/python.exe task_05_q2_missing_simulator.py
The public function simulate_missingness(sample, config, seed, return_original=False)
never mutates its input and does not inspect raw zeros to build artificial masks.
"""
from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q2/missingness"
OUT = TASK / "outputs"
SOURCE = ROOT / "results/q2/protocol/outputs/q2_protocol.json"
SEED = 20260924
PATTERNS = ("text", "audio", "vision", "audio+vision synchronized")
RATES = (0.1, 0.2, 0.3)
POSITIONS = ("early", "middle", "late")
SHORT = {"text": "T", "audio": "A", "vision": "V", "audio+vision synchronized": "AV"}
POS_SHORT = {"early": "E", "middle": "M", "late": "L"}
FIELDS = ("text_bert", "audio", "vision")


def load_protocol():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    p = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected = {"feature_version": "aligned_50", "train_split": "train",
                "validation_split": "valid", "test_split": "test",
                "text_source": "text_bert", "audio_source": "audio", "vision_source": "vision",
                "temporal_length": 50, "text_input_shape": [3, 50],
                "audio_shape": [50, 74], "vision_shape": [50, 35]}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"Task04 protocol conflict: {key}={p.get(key)!r}, expected {value!r}")
    if (p["missing_simulation"]["rates"] != list(RATES)
            or p["missing_simulation"]["controlled_cells"] != 36):
        raise ValueError("Task04 missing simulation grid conflict")
    return p


def _sample_features(sample):
    features = {k: np.asarray(sample[k]) for k in FIELDS}
    if features["text_bert"].shape != (3, 50) or features["audio"].shape != (50, 74) or features["vision"].shape != (50, 35):
        raise ValueError("sample shape conflicts with Task04 aligned_50 protocol")
    text = features["text_bert"]
    if not np.all(np.isfinite(text)) or not np.all(text == text.astype(np.int64)):
        raise ValueError("text_bert must contain finite integral token data")
    if not np.all(np.isin(text[1], (0, 1))):
        raise ValueError("text attention mask is not binary")
    return features


def eligible_token_indices(text_bert):
    """Protect [CLS]/[SEP] by position and BERT IDs 101/102 if present."""
    ids = np.asarray(text_bert[0])
    attention = np.asarray(text_bert[1])
    valid = np.flatnonzero(attention == 1)
    if len(valid) <= 2:
        return np.empty(0, dtype=np.int64)
    interior = valid[1:-1]
    return interior[(ids[interior] != 101) & (ids[interior] != 102)]


def _length(rate, n):
    if n == 0:
        return 0
    # Half-up rounding; never silently select zero if eligible positions exist.
    return min(n, max(1, int(np.floor(rate * n + 0.5))))


def _start(n, length, position, rng):
    if length == 0:
        return None
    if position == "early":
        return 0
    if position == "middle":
        return (n - length) // 2
    if position == "late":
        return n - length
    if position == "random":
        return int(rng.integers(0, n - length + 1))
    raise ValueError(f"unknown position {position}")


def _text_window(eligible, length, position, rng):
    if length == 0:
        return None, None
    # Eligible positions may have a protected special token in the middle.
    # Pick only truly contiguous windows, never span an excluded position.
    starts = [i for i in range(len(eligible) - length + 1)
              if int(eligible[i + length - 1] - eligible[i]) == length - 1]
    if not starts:
        return None, None
    if position == "early":
        first = starts[0]
    elif position == "middle":
        target = (int(eligible[0]) + int(eligible[-1]) - length + 1) / 2
        first = min(starts, key=lambda i: (abs(int(eligible[i]) - target), i))
    elif position == "late":
        first = starts[-1]
    elif position == "random":
        first = int(rng.choice(starts))
    else:
        raise ValueError(f"unknown position {position}")
    start = int(eligible[first])
    return start, start + length


def simulate_missingness(sample, config, seed, return_original=False):
    """Return independent masked arrays, artificial masks and measured metadata.

    config: {pattern, rate, position}; pattern/rate/position may be 'random'.
    For zero eligible text tokens the operation is a recorded no-op.
    """
    src = _sample_features(sample)
    rng = np.random.default_rng(seed)
    pattern = config["pattern"]
    rate = config["rate"]
    position = config["position"]
    if pattern == "random":
        pattern = PATTERNS[int(rng.integers(len(PATTERNS)))]
    if rate == "random":
        rate = RATES[int(rng.integers(len(RATES)))]
    if position == "random":
        position = "random"
    if pattern not in PATTERNS or float(rate) not in RATES or position not in (*POSITIONS, "random"):
        raise ValueError("invalid missingness config")
    rate = float(rate)
    masked = {k: v.copy() for k, v in src.items()}
    masks = {"text": np.ones(50, dtype=np.uint8),
             "audio": np.ones(50, dtype=np.uint8),
             "vision": np.ones(50, dtype=np.uint8)}
    eligible = eligible_token_indices(src["text_bert"])
    detail = {"pattern": pattern, "rate": rate, "position": position, "seed": int(seed),
              "start": None, "end": None, "nominal_length": 0,
              "eligible_token_count": int(len(eligible)), "eligible_token_indices": eligible.tolist(),
              "masked_token_count": 0, "text_start": None, "text_end": None,
              "audio_start": None, "audio_end": None, "vision_start": None, "vision_end": None,
              "selected_rows_audio": 0, "selected_rows_vision": 0,
              "selected_rows_originally_nonzero_audio": 0, "selected_rows_originally_nonzero_vision": 0,
              "selected_rows_originally_zero_audio": 0, "selected_rows_originally_zero_vision": 0,
              "effective_changed_rows_audio": 0, "effective_changed_rows_vision": 0,
              "effective_changed_ratio_audio": 0.0, "effective_changed_ratio_vision": 0.0,
              "nominal_missing_ratio": 0.0, "fallback": "none"}
    if pattern == "text":
        length = _length(rate, len(eligible))
        start, end = _text_window(eligible, length, position, rng)
        if start is None:
            detail["fallback"] = "no_eligible_tokens_or_contiguous_window"
        else:
            masked["text_bert"][:, start:end] = 0
            masks["text"][start:end] = 0
            detail.update(start=start, end=end, text_start=start, text_end=end,
                          nominal_length=length, masked_token_count=length,
                          nominal_missing_ratio=length / len(eligible))
    else:
        length = _length(rate, 50)
        start = _start(50, length, position, rng)
        end = start + length
        detail.update(start=start, end=end, nominal_length=length,
                      nominal_missing_ratio=length / 50)
        modalities = ("audio", "vision") if pattern == "audio+vision synchronized" else (pattern,)
        for name in modalities:
            orig = src[name]
            nonzero = int(np.any(orig[start:end] != 0, axis=1).sum())
            masked[name][start:end, :] = 0
            masks[name][start:end] = 0
            detail[f"{name}_start"] = start
            detail[f"{name}_end"] = end
            detail[f"selected_rows_{name}"] = length
            detail[f"selected_rows_originally_nonzero_{name}"] = nonzero
            detail[f"selected_rows_originally_zero_{name}"] = length - nonzero
            detail[f"effective_changed_rows_{name}"] = nonzero
            detail[f"effective_changed_ratio_{name}"] = nonzero / 50
    result = {**masked,
              "mask_text_artificial": masks["text"],
              "mask_audio_artificial": masks["audio"],
              "mask_vision_artificial": masks["vision"],
              "metadata": detail}
    if return_original:
        result["original_features"] = {k: v.copy() for k, v in src.items()}
    return result


def grid_cells():
    return [{"experiment_id": f"{SHORT[pattern]}_R{round(rate * 100)}_{POS_SHORT[pos]}",
             "pattern": pattern, "rate": rate, "position": pos}
            for pattern in PATTERNS for rate in RATES for pos in POSITIONS]


def _sample(block, index):
    return {k: block[k][index] for k in FIELDS}


def _case(name, callback, results):
    try:
        detail = callback()
        results.append({"test_name": name, "passed": True, "details": detail or "ok"})
    except Exception as exc:
        results.append({"test_name": name, "passed": False, "details": f"{type(exc).__name__}: {exc}"})


def run_tests(train, valid):
    results = []
    sample = _sample(train, 0)
    def fixed(pattern, rate, pos, src=sample):
        return simulate_missingness(src, {"pattern": pattern, "rate": rate, "position": pos}, SEED, True)
    def audio_early():
        r = fixed("audio", 0.1, "early")
        assert (r["mask_audio_artificial"][:5] == 0).all() and (r["mask_audio_artificial"][5:] == 1).all()
        assert np.array_equal(r["audio"][:5], np.zeros((5, 74)))
        assert np.array_equal(r["audio"][5:], sample["audio"][5:])
        assert np.array_equal(r["vision"], sample["vision"]) and np.array_equal(r["text_bert"], sample["text_bert"])
        assert r["metadata"]["start"] == 0 and r["metadata"]["end"] == 5
        return "[0,5); other modalities unchanged"
    _case("audio_10_early", audio_early, results)
    def vision_late():
        r = fixed("vision", 0.3, "late")
        assert (r["mask_vision_artificial"][35:] == 0).all() and (r["mask_vision_artificial"][:35] == 1).all()
        assert r["metadata"]["start"] == 35 and r["metadata"]["end"] == 50
        assert np.array_equal(r["audio"], sample["audio"])
        return "[35,50)"
    _case("vision_30_late", vision_late, results)
    def av_middle():
        r = fixed("audio+vision synchronized", 0.2, "middle")
        assert np.array_equal(r["mask_audio_artificial"], r["mask_vision_artificial"])
        assert r["metadata"]["audio_start"] == r["metadata"]["vision_start"] == 20
        assert r["metadata"]["audio_end"] == r["metadata"]["vision_end"] == 30
        return "identical [20,30) masks"
    _case("av_sync_middle", av_middle, results)
    def text_valid():
        r = fixed("text", 0.3, "middle")
        m = r["mask_text_artificial"]
        eligible = eligible_token_indices(sample["text_bert"])
        chosen = np.flatnonzero(m == 0)
        assert len(chosen) == _length(0.3, len(eligible))
        assert len(chosen) and np.all(np.diff(chosen) == 1)
        assert set(chosen).issubset(set(eligible))
        assert (r["text_bert"][1, chosen] == 0).all()
        assert np.array_equal(r["text_bert"][:, m == 1], sample["text_bert"][:, m == 1])
        assert np.array_equal(m[sample["text_bert"][1] == 0], np.ones(int((sample["text_bert"][1] == 0).sum())))
        return f"eligible={len(eligible)}, selected={len(chosen)}; padding/special protected"
    _case("text_only_eligible", text_valid, results)
    def original_zeros():
        r = fixed("audio", 0.1, "middle")
        zero = np.flatnonzero(np.all(sample["audio"] == 0, axis=1) & (r["mask_audio_artificial"] == 1))
        assert len(zero) > 0 and (r["mask_audio_artificial"][zero] == 1).all()
        return f"{len(zero)} original zero rows remain artificial-mask=1"
    _case("original_zero_rows_preserved", original_zeros, results)
    def same_seed():
        cfg = {"pattern": "random", "rate": "random", "position": "random"}
        a, b = simulate_missingness(sample, cfg, SEED), simulate_missingness(sample, cfg, SEED)
        assert a["metadata"] == b["metadata"]
        assert all(np.array_equal(a[k], b[k]) for k in (*FIELDS, "mask_text_artificial", "mask_audio_artificial", "mask_vision_artificial"))
        return "bitwise identical arrays and metadata"
    _case("same_seed_repeatability", same_seed, results)
    def different_seeds():
        cfg = {"pattern": "audio", "rate": 0.1, "position": "random"}
        starts = {simulate_missingness(sample, cfg, s)["metadata"]["start"] for s in range(10)}
        assert len(starts) > 1 and all(0 <= x <= 45 for x in starts)
        return f"{len(starts)} distinct legal starts in seeds 0..9"
    _case("different_seed_positions", different_seeds, results)
    def immutable():
        before = {k: v.copy() for k, v in sample.items()}
        fixed("audio+vision synchronized", 0.3, "late")
        assert all(np.array_equal(sample[k], before[k]) for k in FIELDS)
        return "all input arrays unchanged"
    _case("input_immutability", immutable, results)
    def grid_complete():
        g = grid_cells()
        assert len(g) == 36 and len({x["experiment_id"] for x in g}) == 36
        assert len({(x["pattern"], x["rate"], x["position"]) for x in g}) == 36
        return "36 unique cells"
    _case("grid_completeness", grid_complete, results)
    def short_text():
        tiny = {k: v.copy() for k, v in sample.items()}
        tiny["text_bert"][:] = 0
        tiny["text_bert"][0, :2] = (101, 102)
        tiny["text_bert"][1, :2] = 1
        r = simulate_missingness(tiny, {"pattern": "text", "rate": 0.3, "position": "early"}, SEED)
        assert r["metadata"]["fallback"] != "none"
        assert (r["mask_text_artificial"] == 1).all()
        return "zero eligible tokens -> recorded no-op"
    _case("short_text_fallback", short_text, results)
    def valid_interface():
        _sample_features(_sample(valid, 0))
        return "valid input shape accepted"
    _case("valid_interface", valid_interface, results)
    return results


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def grid_row(cell, index, sample_id, detail):
    return {"experiment_id": cell["experiment_id"], "sample_index": index,
            "sample_id": sample_id, "pattern": cell["pattern"], "rate": cell["rate"],
            "position": cell["position"], "text_start": detail["text_start"],
            "text_end": detail["text_end"], "text_eligible_count": detail["eligible_token_count"],
            "text_masked_count": detail["masked_token_count"],
            "audio_start": detail["audio_start"], "audio_end": detail["audio_end"],
            "vision_start": detail["vision_start"], "vision_end": detail["vision_end"],
            "nominal_missing_ratio": detail["nominal_missing_ratio"],
            "effective_changed_ratio_audio": detail["effective_changed_ratio_audio"],
            "effective_changed_ratio_vision": detail["effective_changed_ratio_vision"],
            "selected_rows_originally_nonzero_audio": detail["selected_rows_originally_nonzero_audio"],
            "selected_rows_originally_zero_audio": detail["selected_rows_originally_zero_audio"],
            "selected_rows_originally_nonzero_vision": detail["selected_rows_originally_nonzero_vision"],
            "selected_rows_originally_zero_vision": detail["selected_rows_originally_zero_vision"],
            "fallback": detail["fallback"]}


def main():
    protocol = load_protocol()
    source_file = resolve_recorded_path(protocol["attachment2_file"])
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    OUT.mkdir(exist_ok=True)
    with source_file.open("rb") as f:
        data = pickle.load(f)
    train, valid = data["train"], data["valid"]
    assert len(train["id"]) == 3395 and len(valid["id"]) == 728
    tests = run_tests(train, valid)
    if not all(x["passed"] for x in tests):
        write_csv(OUT / "q2_simulator_validation.csv", tests, list(tests[0]))
        raise AssertionError("simulator unit tests failed")
    rows = []
    for cell in grid_cells():
        for index, sample_id in enumerate(valid["id"]):
            r = simulate_missingness(_sample(valid, index), cell, SEED)
            rows.append(grid_row(cell, index, sample_id, r["metadata"]))
    if len(rows) != 36 * 728:
        raise AssertionError("incomplete valid grid")
    write_csv(OUT / "q2_valid_36grid.csv", rows, list(rows[0]))
    _case("valid_grid_26208", lambda: _check_grid(rows), tests)
    # Train-only stochastic smoke audit, one independent seed per sample.
    train_meta = []
    for index in range(len(train["id"])):
        r = simulate_missingness(_sample(train, index),
            {"pattern": "random", "rate": "random", "position": "random"}, SEED + index)
        train_meta.append(r["metadata"])
    _case("train_random_3395", lambda: _check_random(train_meta), tests)
    write_csv(OUT / "q2_simulator_validation.csv", tests, list(tests[0]))
    if not all(x["passed"] for x in tests):
        raise AssertionError("batch simulator validation failed")
    summary = []
    for cell in grid_cells():
        sub = [r for r in rows if r["experiment_id"] == cell["experiment_id"]]
        mods = ("audio", "vision") if cell["pattern"] == "audio+vision synchronized" else (cell["pattern"],)
        effective = [max(r["effective_changed_ratio_audio"], r["effective_changed_ratio_vision"])
                     if cell["pattern"] != "text" else r["nominal_missing_ratio"] for r in sub]
        summary.append({"phase": "valid_fixed", "pattern": cell["pattern"], "rate": cell["rate"],
            "position": cell["position"], "n_samples": len(sub),
            "mean_nominal_ratio": float(np.mean([r["nominal_missing_ratio"] for r in sub])),
            "mean_effective_ratio": float(np.mean(effective)),
            "min_effective_ratio": float(np.min(effective)), "max_effective_ratio": float(np.max(effective)),
            "mean_effective_audio": float(np.mean([r["effective_changed_ratio_audio"] for r in sub])),
            "mean_effective_vision": float(np.mean([r["effective_changed_ratio_vision"] for r in sub])),
            "mean_eligible_tokens": float(np.mean([r["text_eligible_count"] for r in sub])) if "text" in mods else "",
            "mean_masked_tokens": float(np.mean([r["text_masked_count"] for r in sub])) if "text" in mods else "",
            "fallback_count": sum(r["fallback"] != "none" for r in sub)})
    for pattern in PATTERNS:
        for rate in RATES:
            sub = [x for x in train_meta if x["pattern"] == pattern and x["rate"] == rate]
            effective = [max(x["effective_changed_ratio_audio"], x["effective_changed_ratio_vision"])
                         if pattern != "text" else x["nominal_missing_ratio"] for x in sub]
            summary.append({"phase": "train_random", "pattern": pattern, "rate": rate, "position": "random",
                "n_samples": len(sub), "mean_nominal_ratio": float(np.mean([x["nominal_missing_ratio"] for x in sub])),
                "mean_effective_ratio": float(np.mean(effective)), "min_effective_ratio": float(np.min(effective)),
                "max_effective_ratio": float(np.max(effective)),
                "mean_effective_audio": float(np.mean([x["effective_changed_ratio_audio"] for x in sub])),
                "mean_effective_vision": float(np.mean([x["effective_changed_ratio_vision"] for x in sub])),
                "mean_eligible_tokens": float(np.mean([x["eligible_token_count"] for x in sub])) if pattern == "text" else "",
                "mean_masked_tokens": float(np.mean([x["masked_token_count"] for x in sub])) if pattern == "text" else "",
                "fallback_count": sum(x["fallback"] != "none" for x in sub)})
    write_csv(OUT / "q2_missing_simulation_summary.csv", summary, list(summary[0]))
    missing_protocol = {"protocol_source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "feature_version": protocol["feature_version"], "seed": SEED, "patterns": list(PATTERNS),
        "rates": list(RATES), "positions": list(POSITIONS), "controlled_cells": 36,
        "audio_vision_length": 50, "mask_convention": "1=not artificially removed, 0=artificially removed",
        "original_zero_policy": "preserve; never infer artificial mask from feature values",
        "text_rate_basis": "eligible non-padding, non-special tokens; half-up rounded with minimum 1",
        "text_special_protection": "first and last attention-valid positions plus token IDs 101/102",
        "text_fallback": "zero eligible tokens or no contiguous window -> no-op with recorded fallback",
        "validation_mode": "deterministic; 36 cells on all 728 valid samples",
        "train_mode": "uniform random pattern and rate, random legal start; seed=20260924+train_index",
        "interval_convention": "half-open [start,end); early=0, middle=floor((T-L)/2), late=T-L",
        "grid_file": "q2_valid_36grid.csv", "mask_runtime": "call simulate_missingness on original sample and the grid row pattern/rate/position; do not store feature copies",
        "nominal_ratio": "selected positions divided by 50 for audio/vision, selected tokens divided by eligible token count for text",
        "effective_ratio": "originally nonzero selected rows divided by 50 for audio/vision",
        "attachment3_usage": "none"}
    (OUT / "q2_missing_protocol.json").write_text(json.dumps(missing_protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tests_passed": sum(x["passed"] for x in tests), "tests_total": len(tests),
                      "grid_rows": len(rows), "train_random_patterns": dict(Counter(x["pattern"] for x in train_meta)),
                      "grid_fallbacks": sum(r["fallback"] != "none" for r in rows)}, ensure_ascii=False))


def _check_grid(rows):
    assert len(rows) == 26208
    assert len({(r["experiment_id"], r["sample_index"]) for r in rows}) == 26208
    assert all(r["audio_start"] == r["vision_start"] and r["audio_end"] == r["vision_end"]
               for r in rows if r["pattern"] == "audio+vision synchronized")
    return "26208 unique cell-sample rows; AV intervals identical"


def _check_random(meta):
    assert len(meta) == 3395
    assert all(x["fallback"] == "none" for x in meta)
    assert all(x["start"] is not None and x["end"] is not None and x["end"] > x["start"] for x in meta)
    assert all(x["audio_start"] == x["vision_start"] and x["audio_end"] == x["vision_end"]
               for x in meta if x["pattern"] == "audio+vision synchronized")
    return "3395 legal stochastic calls, no fallback, AV synchronized"


if __name__ == "__main__":
    main()
