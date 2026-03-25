"""Generate diverse keyframe test fixture pairs for divergence testing.

Creates 5 pairs of 480x704 PNG images exercising different visual properties:
solid colors, gradients, identity (same image), text overlays, and geometric patterns.

Usage:
    uv run python scripts/keyframe_tests/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "keyframe_pairs"
WIDTH = 704
HEIGHT = 480


def _save_pair(name: str, start: Image.Image, end: Image.Image) -> None:
    """Save a start/end image pair to the fixtures directory."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    start.save(FIXTURES_DIR / f"{name}_start.png")
    end.save(FIXTURES_DIR / f"{name}_end.png")
    print(f"  {name}: {FIXTURES_DIR / name}_start.png, {FIXTURES_DIR / name}_end.png")


def generate_solid_colors() -> None:
    """Solid red -> Solid blue."""
    start = Image.new("RGB", (WIDTH, HEIGHT), (255, 0, 0))
    end = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 255))
    _save_pair("solid_colors", start, end)


def generate_gradient() -> None:
    """Horizontal gradient (black->white) -> Vertical gradient (black->white)."""
    # Horizontal gradient
    start = Image.new("RGB", (WIDTH, HEIGHT))
    for x in range(WIDTH):
        val = int(255 * x / (WIDTH - 1))
        for y in range(HEIGHT):
            start.putpixel((x, y), (val, val, val))

    # Vertical gradient
    end = Image.new("RGB", (WIDTH, HEIGHT))
    for y in range(HEIGHT):
        val = int(255 * y / (HEIGHT - 1))
        for x in range(WIDTH):
            end.putpixel((x, y), (val, val, val))

    _save_pair("gradient", start, end)


def generate_identity() -> None:
    """Same checkerboard image duplicated — should produce near-static video."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    cell_w = WIDTH // 8
    cell_h = HEIGHT // 8
    for row in range(8):
        for col in range(8):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
            x0 = col * cell_w
            y0 = row * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color)
    _save_pair("identity", img.copy(), img.copy())


def generate_text_overlay() -> None:
    """White background with large black text 'START' -> 'END'."""
    for label, suffix in [("START", "start"), ("END", "end")]:
        img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Try to use a large font; fall back to default if unavailable
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 120)
            except OSError:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (WIDTH - text_w) // 2
        y = (HEIGHT - text_h) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        img.save(FIXTURES_DIR / f"text_overlay_{suffix}.png")
    print(f"  text_overlay: {FIXTURES_DIR / 'text_overlay'}_start.png, {FIXTURES_DIR / 'text_overlay'}_end.png")


def generate_geometric() -> None:
    """8x8 checkerboard -> diagonal stripes."""
    # Checkerboard
    start = Image.new("RGB", (WIDTH, HEIGHT))
    draw_start = ImageDraw.Draw(start)
    cell_w = WIDTH // 8
    cell_h = HEIGHT // 8
    for row in range(8):
        for col in range(8):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
            x0 = col * cell_w
            y0 = row * cell_h
            draw_start.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color)

    # Diagonal stripes
    end = Image.new("RGB", (WIDTH, HEIGHT))
    stripe_width = 40
    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = 255 if ((x + y) // stripe_width) % 2 == 0 else 0
            end.putpixel((x, y), (val, val, val))

    _save_pair("geometric", start, end)


GENERATORS = [
    generate_solid_colors,
    generate_gradient,
    generate_identity,
    generate_text_overlay,
    generate_geometric,
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
