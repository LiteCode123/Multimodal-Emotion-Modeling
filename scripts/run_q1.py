"""Rebuild the final Q1 official-transcript feature table from prepared media features."""

import csv

from _common import ROOT, cli, new_run_dir, require
from src.data.paths import resolve_recorded_path
from src.features import q1_official_text as q1


def main() -> None:
    require(q1.MANIFEST, q1.BASE, q1.OLD_SUMMARY, q1.LABEL)
    with q1.MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        require(*(resolve_recorded_path(row["video_path"]) for row in csv.DictReader(stream)))
    destination = new_run_dir(ROOT / "runs/q1/official", "outputs/q1_official_text_multimodal_features_50bin.csv")
    q1.TASK = destination
    q1.OUT = destination / "outputs"
    q1.FIG = destination / "figures"
    q1.FEATURE_OUT = q1.OUT / "q1_official_text_multimodal_features_50bin.csv"
    q1.WORD_OUT = q1.OUT / "q1_official_word_alignment_100.csv"
    q1.SUMMARY_OUT = q1.OUT / "q1_official_text_sample_summary_100.csv"
    q1.SCHEMA_OUT = q1.OUT / "q1_official_text_schema.json"
    q1.VALIDATION_OUT = q1.OUT / "q1_official_text_validation.json"
    q1.OUT.mkdir(parents=True, exist_ok=True)
    q1.FIG.mkdir(parents=True, exist_ok=True)
    q1.main()


if __name__ == "__main__":
    cli(main)
