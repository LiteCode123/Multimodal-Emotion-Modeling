"""Reproduce the frozen Q2 Attachment 3 prediction and validated calibration."""

from _common import ROOT, cli, config, new_run_dir, require
from src.data.paths import resolve_recorded_path
from src.models import q2_final_calibration as final
from src.models import q2_inference as inference


def main() -> None:
    cfg = config("q2")
    lock = inference.read_json(ROOT / "results/q2/final/outputs/q2_fix01_validation.json")
    if cfg["final_neutral_logit_penalty"] != lock["selected_neutral_logit_penalty"]:
        raise ValueError("Q2 config differs from the frozen validation calibration")
    protocol = inference.read_json(inference.T04 / "q2_protocol.json")
    manifest = inference.read_json(inference.T06 / "baseline_model_manifest.json")
    require(resolve_recorded_path(protocol["attachment2_file"]),
            resolve_recorded_path(protocol["attachment3_directory"]),
            resolve_recorded_path(protocol["attachment3_directory"]).parent / "未对齐版本",
            resolve_recorded_path(manifest["text_encoder_local_path"]) / "model.safetensors",
            ROOT / cfg["checkpoint"])
    inference.TASK = new_run_dir(ROOT / cfg["run_dir"] / "inference", "submission/attachment3_predictions.csv")
    inference.OUT = inference.TASK / "outputs"
    inference.SUBMISSION = inference.TASK / "submission"
    inference.OUT.mkdir(parents=True, exist_ok=True)
    inference.SUBMISSION.mkdir(parents=True, exist_ok=True)
    inference.main()
    final.TASK = new_run_dir(ROOT / cfg["run_dir"] / "final", "outputs/attachment3_predictions_fix01_candidate.csv")
    final.OUT = final.TASK / "outputs"
    final.OUT.mkdir(parents=True, exist_ok=True)
    final.main()


if __name__ == "__main__":
    cli(main)
