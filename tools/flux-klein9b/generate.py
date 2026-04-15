#!/usr/bin/env python3
"""
FLUX.2 klein 9B Image Generation Script
========================================
Uses Diffusers library with Apple Silicon MPS support.

Usage:
    uv run python generate.py --prompt "a cat" --output cat.png
    uv run python generate.py --prompt "landscape" --height 768 --width 1344 --steps 4
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline


def get_device() -> tuple[str, torch.dtype]:
    """Detect available device (MPS/CUDA/CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        print("[Device] Apple Silicon MPS (Metal)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("[Device] CPU (slow, not recommended)")
    return device, dtype


def load_pipeline(
    model_path: str | Path,
    device: str,
    dtype: torch.dtype,
    enable_cpu_offload: bool = True,
) -> Flux2KleinPipeline:
    """Load FLUX.2 klein pipeline."""
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"[Error] Model path not found: {model_path}")
        print(f"[Info] Please download the model and place it at this path.")
        print(f"[Info] See README.md for download instructions.")
        sys.exit(1)

    print(f"[Loading] FLUX.2 klein 9B from {model_path}...")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )

    if enable_cpu_offload:
        print("[Memory] Enabling model CPU offload (saves VRAM)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def generate_image(
    pipe: Flux2KleinPipeline,
    prompt: str,
    output_path: str | Path,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int | None = None,
) -> None:
    """Generate and save an image."""
    output_path = Path(output_path)

    print(f"[Prompt] {prompt}")
    print(f"[Config] {height}x{width}, {num_inference_steps} steps, guidance={guidance_scale}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"[Seed] {seed}")

    print("[Generating] ... (this may take a while on first run)")

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"[Done] Image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2 klein 9B Text-to-Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python generate.py --prompt "a beautiful sunset over the ocean"
  uv run python generate.py -p "cyberpunk city" -o cyberpunk.png -H 768 -W 1344
  uv run python generate.py -p "portrait" --seed 42 --steps 4
        """,
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.png",
        help="Output image path (default: output.png)",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default="./models/black-forest-labs/FLUX.2-klein-9B",
        help="Path to model directory (default: ./models/black-forest-labs/FLUX.2-klein-9B)",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4, distilled model)",
    )
    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance scale (default: 1.0, set higher for more prompt adherence)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable CPU offload (uses more VRAM, faster inference)",
    )

    args = parser.parse_args()

    device, dtype = get_device()
    pipe = load_pipeline(
        args.model_path,
        device,
        dtype,
        enable_cpu_offload=not args.no_cpu_offload,
    )

    generate_image(
        pipe,
        args.prompt,
        args.output,
        args.height,
        args.width,
        args.steps,
        args.guidance,
        args.seed,
    )


if __name__ == "__main__":
    main()
