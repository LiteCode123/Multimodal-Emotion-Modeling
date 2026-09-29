"""Resolve paths recorded by the original experiment without editing frozen outputs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_recorded_path(value: str) -> Path:
    raw = str(value).replace("\\", "/")
    if raw.startswith("01_data/"):
        raw = "data/" + raw.removeprefix("01_data/")
    elif raw.startswith("models/"):
        raw = "data/models/" + raw.removeprefix("models/")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe recorded path: {value}")
    return ROOT / relative
