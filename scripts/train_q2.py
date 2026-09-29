"""Train MAF-MT with the frozen Q2 protocol into runs/q2/model."""

from _common import ROOT, cli, config, new_run_dir, require
from src.data.paths import resolve_recorded_path
from src.models import q2_mafmt as q2


def main() -> None:
    cfg = config("q2")
    if cfg["seed"] != q2.SEED or cfg["sequence_length"] != 50 or cfg["batch_size"] != q2.BS:
        raise ValueError("Q2 config differs from the frozen training protocol")
    manifest = ROOT / "results/q2/baseline/models/baseline_model_manifest.json"
    require(ROOT / cfg["source"], manifest, q2.GRID)
    import json
    bert = resolve_recorded_path(json.loads(manifest.read_text(encoding="utf-8"))["text_encoder_local_path"])
    require(bert / "model.safetensors")
    q2.TASK = new_run_dir(ROOT / cfg["run_dir"] / "model", "models/q2_mafmt_best.pt")
    q2.OUT = q2.TASK / "outputs"
    q2.MODELS = q2.TASK / "models"
    q2.CACHE = q2.TASK / "cache"
    q2.FIG = q2.TASK / "figures"
    q2.main()


if __name__ == "__main__":
    cli(main)
