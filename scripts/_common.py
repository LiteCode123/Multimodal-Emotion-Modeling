"""Small CLI helpers; the .yaml files use JSON syntax, which is valid YAML."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def config(question: str) -> dict:
    return json.loads((ROOT / "configs" / f"{question}.yaml").read_text(encoding="utf-8"))


def require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        shown = "\n".join(missing[:5])
        extra = f"\n... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError("Missing required input(s):\n" + shown + extra)


def new_run_dir(path: Path, output_name: str) -> Path:
    if (path / output_name).exists():
        raise FileExistsError(f"Run output already exists: {path / output_name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cli(main) -> None:
    try:
        main()
    except (FileNotFoundError, FileExistsError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
