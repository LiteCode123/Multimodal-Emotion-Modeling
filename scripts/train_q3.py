"""Train and select the Q3 GMF-MT model into runs/q3/model."""

import sys

from _common import ROOT, cli, config, new_run_dir, require
from src.models import q3_gmf as q3


def main() -> None:
    cfg = config("q3")
    if (cfg["seed"] != q3.SEED or cfg["fusion"] != "gated" or cfg["hidden"] != 128
            or cfg["dropout"] != 0.2):
        raise ValueError("Q3 config differs from the frozen model setting")
    require(ROOT / cfg["source"], q3.PROTOCOL)
    q3.TASK = new_run_dir(ROOT / cfg["run_dir"] / "model", "outputs/q3_final_model.pt")
    q3.OUT = q3.TASK / "outputs"
    q3.FIG = q3.TASK / "figures"
    q3.OUT.mkdir(parents=True, exist_ok=True)
    q3.FIG.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--max-candidates", "4"]
    q3.main()


if __name__ == "__main__":
    cli(main)
