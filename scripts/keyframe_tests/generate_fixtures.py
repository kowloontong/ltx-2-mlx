"""Generate realistic keyframe test fixture pairs for interpolation testing.

Creates 5 pairs of 480x704 PNG images from real photographs in tests/fixtures/.
Uses crops, color shifts, and transforms to create diverse interpolation scenarios.

Source images:
    - keyframe_start.png / keyframe_end.png (hedgehog 3D character, 832x1472)
    - test_i2v.png (cat in forest, 2816x1536)

Usage:
    uv run python scripts/keyframe_tests/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "keyframe_pairs"
SOURCE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
WIDTH = 704
HEIGHT = 480


def _crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop and resize to target dimensions preserving aspect ratio."""
    # Resize so the smallest dimension fits, then center crop
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _save_pair(name: str, start: Image.Image, end: Image.Image) -> None:
    """Save a start/end image pair to the fixtures directory."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    start.save(FIXTURES_DIR / f"{name}_start.png")
    end.save(FIXTURES_DIR / f"{name}_end.png")
    print(f"  {name}: start {start.size}, end {end.size}")


def generate_hedgehog() -> None:
    """Original hedgehog pair — the known-working interpolation test."""
    start = Image.open(SOURCE_DIR / "keyframe_start.png")
    end = Image.open(SOURCE_DIR / "keyframe_end.png")
    _save_pair("hedgehog", _crop_center(start, WIDTH, HEIGHT), _crop_center(end, WIDTH, HEIGHT))


def generate_forest_zoom() -> None:
    """Cat in forest: wide shot -> zoomed-in crop (simulates camera zoom)."""
    img = Image.open(SOURCE_DIR / "test_i2v.png")
    # Wide shot: full image
    start = _crop_center(img, WIDTH, HEIGHT)
    # Zoom: crop center 50% of original, then resize up
    src_w, src_h = img.size
    crop_w, crop_h = src_w // 2, src_h // 2
    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2
    zoomed = img.crop((left, top, left + crop_w, top + crop_h))
    end = _crop_center(zoomed, WIDTH, HEIGHT)
    _save_pair("forest_zoom", start, end)


def generate_day_to_night() -> None:
    """Cat in forest: bright daylight -> dark blue-shifted (simulates day-to-dusk)."""
    img = Image.open(SOURCE_DIR / "test_i2v.png")
    base = _crop_center(img, WIDTH, HEIGHT)
    # Start: bright, warm
    start = ImageEnhance.Brightness(base).enhance(1.15)
    start = ImageEnhance.Color(start).enhance(1.2)
    # End: dark, blue-shifted
    dark = ImageEnhance.Brightness(base).enhance(0.35)
    arr = np.array(dark, dtype=np.float32)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.6, 0, 255)  # boost blue
    arr[:, :, 0] = arr[:, :, 0] * 0.7  # reduce red
    arr[:, :, 1] = arr[:, :, 1] * 0.8  # reduce green
    end = Image.fromarray(arr.astype(np.uint8))
    _save_pair("day_to_night", start, end)


def generate_hedgehog_seasons() -> None:
    """Hedgehog start: warm autumn tones -> hedgehog end: cool winter tones."""
    start_src = Image.open(SOURCE_DIR / "keyframe_start.png")
    end_src = Image.open(SOURCE_DIR / "keyframe_end.png")
    start_base = _crop_center(start_src, WIDTH, HEIGHT)
    end_base = _crop_center(end_src, WIDTH, HEIGHT)
    # Start: warm autumn (boost red/yellow, increase saturation)
    arr_s = np.array(start_base, dtype=np.float32)
    arr_s[:, :, 0] = np.clip(arr_s[:, :, 0] * 1.15, 0, 255)
    arr_s[:, :, 2] = arr_s[:, :, 2] * 0.85
    start = ImageEnhance.Color(Image.fromarray(arr_s.astype(np.uint8))).enhance(1.3)
    # End: cool winter (boost blue, desaturate, brighten slightly)
    arr_e = np.array(end_base, dtype=np.float32)
    arr_e[:, :, 2] = np.clip(arr_e[:, :, 2] * 1.3, 0, 255)
    arr_e[:, :, 0] = arr_e[:, :, 0] * 0.9
    end = ImageEnhance.Color(Image.fromarray(arr_e.astype(np.uint8))).enhance(0.7)
    end = ImageEnhance.Brightness(end).enhance(1.1)
    _save_pair("hedgehog_seasons", start, end)


def generate_forest_blur() -> None:
    """Cat in forest: sharp focus -> soft bokeh blur (simulates focus pull)."""
    img = Image.open(SOURCE_DIR / "test_i2v.png")
    base = _crop_center(img, WIDTH, HEIGHT)
    start = ImageEnhance.Sharpness(base).enhance(1.5)
    end = base.filter(ImageFilter.GaussianBlur(radius=8))
    _save_pair("forest_blur", start, end)


GENERATORS = [
    generate_hedgehog,
    generate_forest_zoom,
    generate_day_to_night,
    generate_hedgehog_seasons,
    generate_forest_blur,
]


def main() -> None:
    """Generate all fixture pairs."""
    print(f"Generating keyframe test fixtures in {FIXTURES_DIR}")
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for gen in GENERATORS:
        gen()
    print(f"\nDone. {len(GENERATORS)} fixture pairs generated.")


if __name__ == "__main__":
    main()
