"""Shared test fixtures and helpers."""

from pathlib import Path

_HUB_DIR = Path.home() / ".cache/huggingface/hub"
_Q8_CANDIDATES = [
    _HUB_DIR / "models--dgrauet--ltx-2.3-mlx-q8" / "snapshots",
]


def find_q8_model_dir() -> Path | None:
    """Find the latest q8 model snapshot directory."""
    for candidate in _Q8_CANDIDATES:
        if not candidate.exists():
            continue
        snapshots = sorted(candidate.iterdir())
        if not snapshots:
            continue
        model_dir = snapshots[-1]
        if (model_dir / "transformer-distilled.safetensors").exists():
            return model_dir
    return None


MODEL_DIR = find_q8_model_dir()
