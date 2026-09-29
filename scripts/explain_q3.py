"""Predict and explain the 20 unlabeled Attachment 4 samples with frozen Q3 artifacts."""

from _common import ROOT, cli, config, new_run_dir, require
from src.explainability import q3_inference as q3
import json


def main() -> None:
    cfg = config("q3")
    locked = json.loads((ROOT / "results/q3/model/outputs/q3_model_config.json").read_text(encoding="utf-8"))
    if any(cfg[key] != locked[key] for key in ("fusion", "hidden", "dropout")):
        raise ValueError("Q3 config differs from the frozen checkpoint")
    require(ROOT / cfg["checkpoint"], ROOT / cfg["preprocessing"],
            ROOT / cfg["attachment4"], q3.TEST14, q3.TEST15)
    q3.TASK = new_run_dir(ROOT / cfg["run_dir"] / "inference", "outputs/attachment4_predictions.csv")
    q3.OUT = q3.TASK / "outputs"
    q3.FIG = q3.TASK / "figures"
    q3.OUT.mkdir(parents=True, exist_ok=True)
    q3.FIG.mkdir(parents=True, exist_ok=True)
    q3.main()


if __name__ == "__main__":
    cli(main)
