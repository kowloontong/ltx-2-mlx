"""Shared test fixtures and helpers."""

from pathlib import Path

_HUB_DIR = Path.home() / ".cache/huggingface/hub"
_Q8_MODEL = _HUB_DIR / "models--dgrauet--ltx-2.3-mlx-distilled-q8" / "snapshots"


def find_q8_model_dir() -> Path | None:
    """Find the latest q8 model snapshot directory."""
    if not _Q8_MODEL.exists():
        return None
    snapshots = sorted(_Q8_MODEL.iterdir())
    if not snapshots:
        return None
    model_dir = snapshots[-1]
    # Verify the transformer weights actually exist
    if not (model_dir / "transformer-distilled.safetensors").exists():
        return None
    return model_dir


MODEL_DIR = find_q8_model_dir()
