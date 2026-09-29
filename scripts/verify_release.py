"""Read-only checks for the curated results; official raw data is not required."""

import csv
import json

import numpy as np
import torch

from _common import ROOT, require
from src.models.q2_mafmt import MAFMT
from src.models.q3_gmf import Q3Model


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    public_summary = ROOT / "results/public_summary.json"
    q2 = ROOT / "results/q2/test/outputs/q2_final_test_metrics.csv"
    q3 = ROOT / "results/q3/model/outputs/q3_final_metrics.json"
    q2_checkpoint = ROOT / "results/q2/model/models/q2_mafmt_best.pt"
    q3_checkpoint = ROOT / "results/q3/model/outputs/q3_final_model.pt"
    require(public_summary, q2, q3, q2_checkpoint, q3_checkpoint)

    summary = json.loads(public_summary.read_text(encoding="utf-8"))
    assert summary["q1"]["validation_status"] == "PASS"
    assert summary["q1"]["sample_count"] == 100
    assert (summary["q1"]["feature_rows"], summary["q1"]["feature_columns"]) == (5000, 843)
    assert summary["q2"]["attachment3_unlabeled_predictions"] == 30
    assert summary["q3"]["attachment4_unlabeled_predictions_and_explanations"] == 20
    q2_metrics = {row["model"]: row for row in rows(q2)}
    assert set(q2_metrics) == {"Baseline", "Full"}
    assert all(int(row["n_test"]) == 727 for row in q2_metrics.values())
    assert float(q2_metrics["Full"]["Macro_F1"]) > float(q2_metrics["Baseline"]["Macro_F1"])
    assert float(q2_metrics["Full"]["MAE"]) < float(q2_metrics["Baseline"]["MAE"])
    q3_metrics = json.loads(q3.read_text(encoding="utf-8"))
    assert 0 < q3_metrics["test"]["macro_f1"] < 1

    q2_saved = torch.load(q2_checkpoint, map_location="cpu", weights_only=True)
    q2_model = MAFMT().eval()
    q2_model.load_state_dict(q2_saved["state_dict"], strict=True)
    with torch.inference_mode():
        p, r, g = q2_model(torch.zeros(1, 50, 768), torch.zeros(1, 50, 74),
                           torch.zeros(1, 50, 35), torch.ones(1, 50),
                           torch.zeros(1, 50, 3), torch.zeros(1, 50, 2))
    assert p.shape == (1, 3) and r.shape == (1,) and g.shape == (1, 50, 3)
    assert np.isfinite(p.numpy()).all()

    q3_config = json.loads((ROOT / "results/q3/model/outputs/q3_model_config.json").read_text(encoding="utf-8"))
    q3_model = Q3Model(q3_config["fusion"], q3_config["hidden"], q3_config["dropout"]).eval()
    q3_model.load_state_dict(torch.load(q3_checkpoint, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        p, r = q3_model({"text": torch.zeros(1, 768), "audio": torch.zeros(1, 74),
                         "vision": torch.zeros(1, 35)})
    assert p.shape == (1, 3) and r.shape == (1,)
    print("PASS: anonymous Q1 summary; Q2 test 727; Q3 test metrics; both checkpoints load")


if __name__ == "__main__":
    main()
