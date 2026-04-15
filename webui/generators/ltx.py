"""LTX video generator with support for T2V and I2V.

Defaults to two-stage HQ pipeline for best quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

from .base import BaseGenerator
from ..config import LTXConfig, PipelineConfig


class LTXGenerator(BaseGenerator):
    """LTX video generator supporting T2V and I2V."""

    def __init__(self, config: LTXConfig, pipeline_config: PipelineConfig):
        self.config = config
        self.pipeline_config = pipeline_config

    def generate_t2v(
        self,
        prompt: str,
        output_path: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        frames: Optional[int] = None,
        seed: int = 42,
        pipeline_type: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        stage1_steps: Optional[int] = None,
        stage2_steps: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Generate video from text (T2V).

        Args:
            prompt: Text prompt.
            output_path: Output video path.
            height: Video height.
            width: Video width.
            frames: Number of frames.
            seed: Random seed.
            pipeline_type: Pipeline type (one-stage, two-stage, two-stage-hq).
            cfg_scale: CFG guidance scale.
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.

        Returns:
            Tuple of (success, error_message).
        """
        # Use defaults from config
        pipeline_type = pipeline_type or self.pipeline_config.default_pipeline
        cfg_scale = cfg_scale or self.pipeline_config.default_cfg_scale
        stage1_steps = stage1_steps or self.pipeline_config.default_stage1_steps
        stage2_steps = stage2_steps or self.pipeline_config.default_stage2_steps

        height = height or self.config.default_height
        width = width or self.config.default_width
        frames = frames or self.config.default_frames

        # Build command
        python_path = Path(".venv/bin/python3")
        cmd = [
            str(python_path),
            "-m",
            "ltx_pipelines_mlx",
            "generate",
            "--model",
            str(self.config.model_dir),
            "--prompt",
            prompt,
            "--output",
            output_path,
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(frames),
            "--seed",
            str(seed),
        ]

        # Add pipeline-specific arguments
        if pipeline_type == "two-stage":
            cmd.extend(
                [
                    "--two-stage",
                    "--stage1-steps",
                    str(stage1_steps),
                    "--stage2-steps",
                    str(stage2_steps),
                    "--cfg-scale",
                    str(cfg_scale),
                ]
            )
        elif pipeline_type == "two-stage-hq":
            cmd.extend(
                [
                    "--hq",
                    "--stage1-steps",
                    str(stage1_steps),
                    "--stage2-steps",
                    str(stage2_steps),
                    "--cfg-scale",
                    str(cfg_scale),
                ]
            )
        else:  # one-stage
            cmd.extend(["--steps", "8"])

        cmd.append("--quiet")

        # Execute
        process = self.run_command(cmd)
        returncode = self.wait_for_process(process)

        success = returncode == 0 and Path(output_path).exists()
        return success, "" if success else "LTX T2V generation failed"

    def generate_i2v(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        frames: Optional[int] = None,
        seed: int = 42,
        pipeline_type: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        stage1_steps: Optional[int] = None,
        stage2_steps: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Generate video from image (I2V).

        Args:
            prompt: Text prompt.
            image_path: Input image path.
            output_path: Output video path.
            height: Video height.
            width: Video width.
            frames: Number of frames (defaults to i2v_frames for better detail).
            seed: Random seed.
            pipeline_type: Pipeline type.
            cfg_scale: CFG guidance scale (defaults to i2v_cfg_scale).
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.

        Returns:
            Tuple of (success, error_message).
        """
        # Use I2V-specific defaults
        pipeline_type = pipeline_type or self.pipeline_config.default_pipeline
        cfg_scale = cfg_scale or self.pipeline_config.i2v_cfg_scale
        frames = frames or self.pipeline_config.i2v_frames  # Fewer frames for I2V

        height = height or self.config.default_height
        width = width or self.config.default_width

        stage1_steps = stage1_steps or self.pipeline_config.default_stage1_steps
        stage2_steps = stage2_steps or self.pipeline_config.default_stage2_steps

        # Build command
        python_path = Path(".venv/bin/python3")
        cmd = [
            str(python_path),
            "-m",
            "ltx_pipelines_mlx",
            "generate",
            "--model",
            str(self.config.model_dir),
            "--prompt",
            prompt,
            "--output",
            output_path,
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(frames),
            "--seed",
            str(seed),
            "--image",
            image_path,
        ]

        # Add pipeline-specific arguments
        if pipeline_type == "two-stage":
            cmd.extend(
                [
                    "--two-stage",
                    "--stage1-steps",
                    str(stage1_steps),
                    "--stage2-steps",
                    str(stage2_steps),
                    "--cfg-scale",
                    str(cfg_scale),
                ]
            )
        elif pipeline_type == "two-stage-hq":
            cmd.extend(
                [
                    "--hq",
                    "--stage1-steps",
                    str(stage1_steps),
                    "--stage2-steps",
                    str(stage2_steps),
                    "--cfg-scale",
                    str(cfg_scale),
                ]
            )
        else:  # one-stage
            cmd.extend(["--steps", "8"])

        cmd.append("--quiet")

        # Execute
        process = self.run_command(cmd)
        returncode = self.wait_for_process(process)

        success = returncode == 0 and Path(output_path).exists()
        return success, "" if success else "LTX I2V generation failed"

    def generate(
        self,
        prompt: str,
        output_path: str,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Generate video (T2V or I2V depending on image_path).

        Args:
            prompt: Text prompt.
            output_path: Output video path.
            image_path: Optional input image path (for I2V).
            **kwargs: Additional arguments.

        Returns:
            Tuple of (success, error_message).
        """
        if image_path:
            return self.generate_i2v(
                prompt=prompt,
                image_path=image_path,
                output_path=output_path,
                **kwargs,
            )
        else:
            return self.generate_t2v(
                prompt=prompt,
                output_path=output_path,
                **kwargs,
            )
