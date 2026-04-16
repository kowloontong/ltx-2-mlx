"""Flux image generator using stable-diffusion.cpp."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .base import BaseGenerator
from ..config import FluxConfig


class FluxGenerator(BaseGenerator):
    """Flux image generator."""

    def __init__(self, config: FluxConfig):
        self.config = config

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = None,
        height: int = None,
        steps: int = None,
        cfg_scale: float = None,
    ) -> Tuple[bool, str]:
        """Generate image from text.

        Args:
            prompt: Text prompt.
            output_path: Output image path.
            width: Image width.
            height: Image height.
            steps: Denoising steps.
            cfg_scale: CFG guidance scale.

        Returns:
            Tuple of (success, error_message).
        """
        width = width or self.config.default_width
        height = height or self.config.default_height
        steps = steps or self.config.default_steps
        cfg_scale = cfg_scale or self.config.default_cfg_scale

        # Convert output path to absolute path
        output_path = str(Path(output_path).resolve())

        # Build command
        script_path = self.config.script_dir / "generate.py"
        cmd = [
            "python3",
            str(script_path),
            prompt,
            "-o",
            output_path,
            "-w",
            str(width),
            "--height",
            str(height),
            "-s",
            str(steps),
            "--cfg-scale",
            str(cfg_scale),
        ]

        # Execute
        process = self.run_command(cmd, cwd=str(self.config.script_dir))
        returncode = self.wait_for_process(process)

        # Check result
        output_exists = Path(output_path).exists()
        
        if returncode != 0:
            return False, f"Flux generation failed (return code: {returncode})"
        elif not output_exists:
            return False, f"Flux output file not created: {output_path}"
        else:
            return True, ""
