"""Configuration management for Web UI.

Supports loading from:
1. Environment variables
2. YAML configuration file
3. Default values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os


@dataclass
class FluxConfig:
    """Flux image generator configuration."""

    script_dir: Path
    model_dir: Path
    output_dir: Path
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 4
    default_cfg_scale: float = 1.0


@dataclass
class LTXConfig:
    """LTX video generator configuration."""

    model_dir: Path
    output_dir: Path
    default_height: int = 576
    default_width: int = 832
    default_frames: int = 97


@dataclass
class PipelineConfig:
    """Pipeline configuration - defaults to two-stage HQ."""

    default_pipeline: str = "two-stage-hq"  # Default to HQ
    default_cfg_scale: float = 3.5
    default_stage1_steps: int = 20
    default_stage2_steps: int = 5

    # I2V-specific configuration
    i2v_cfg_scale: float = 3.0
    i2v_frames: int = 65  # I2V recommends fewer frames


@dataclass
class WebUIConfig:
    """Main Web UI configuration."""

    flux: FluxConfig
    ltx: LTXConfig
    pipeline: PipelineConfig
    qwen_model_path: Path

    @classmethod
    def from_env(cls) -> WebUIConfig:
        """Load configuration from environment variables."""
        home = Path.home()

        return cls(
            flux=FluxConfig(
                script_dir=Path(os.getenv("FLUX_SCRIPT_DIR", str(home / "work/flux-gguf-image-gen"))),
                model_dir=Path(os.getenv("FLUX_MODEL_DIR", str(home / "work/flux-gguf-image-gen/models"))),
                output_dir=Path(os.getenv("FLUX_OUTPUT_DIR", "outputs/webui/flux")),
            ),
            ltx=LTXConfig(
                model_dir=Path(os.getenv("LTX_MODEL_DIR", "models/ltx-2.3-mlx-q8")),
                output_dir=Path(os.getenv("LTX_OUTPUT_DIR", "outputs/webui")),
            ),
            pipeline=PipelineConfig(),
            qwen_model_path=Path(
                os.getenv("QWEN_MODEL_PATH", str(home / "models/Qwen3.5-27B.Q4_K_M.gguf"))
            ),
        )

    @classmethod
    def from_yaml(cls, path: str = "configs/webui.yaml") -> WebUIConfig:
        """Load configuration from YAML file.

        Falls back to environment variables if file doesn't exist.
        """
        yaml_path = Path(path)

        if not yaml_path.exists():
            print(f"Config file not found: {path}, using environment variables")
            return cls.from_env()

        try:
            import yaml

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            return cls(
                flux=FluxConfig(
                    script_dir=Path(data.get("flux", {}).get("script_dir", "~/work/flux-gguf-image-gen")).expanduser(),
                    model_dir=Path(data.get("flux", {}).get("model_dir", "~/work/flux-gguf-image-gen/models")).expanduser(),
                    output_dir=Path(data.get("flux", {}).get("output_dir", "outputs/webui/flux")),
                    default_width=data.get("flux", {}).get("default_width", 1024),
                    default_height=data.get("flux", {}).get("default_height", 1024),
                ),
                ltx=LTXConfig(
                    model_dir=Path(data.get("ltx", {}).get("model_dir", "models/ltx-2.3-mlx-q8")),
                    output_dir=Path(data.get("ltx", {}).get("output_dir", "outputs/webui")),
                    default_height=data.get("ltx", {}).get("default_height", 576),
                    default_width=data.get("ltx", {}).get("default_width", 832),
                    default_frames=data.get("ltx", {}).get("default_frames", 97),
                ),
                pipeline=PipelineConfig(
                    default_pipeline=data.get("pipeline", {}).get("default_pipeline", "two-stage-hq"),
                    default_cfg_scale=data.get("pipeline", {}).get("default_cfg_scale", 3.5),
                    default_stage1_steps=data.get("pipeline", {}).get("default_stage1_steps", 20),
                    default_stage2_steps=data.get("pipeline", {}).get("default_stage2_steps", 5),
                    i2v_cfg_scale=data.get("pipeline", {}).get("i2v_cfg_scale", 3.0),
                    i2v_frames=data.get("pipeline", {}).get("i2v_frames", 65),
                ),
                qwen_model_path=Path(data.get("qwen_model_path", "~/models/Qwen3.5-27B.Q4_K_M.gguf")).expanduser(),
            )
        except Exception as e:
            print(f"Error loading config from YAML: {e}, using environment variables")
            return cls.from_env()

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> WebUIConfig:
        """Load configuration with fallback chain.

        Priority:
        1. YAML file (if config_path provided or configs/webui.yaml exists)
        2. Environment variables
        3. Defaults
        """
        if config_path:
            return cls.from_yaml(config_path)

        # Try default config path
        default_config = Path("configs/webui.yaml")
        if default_config.exists():
            return cls.from_yaml(str(default_config))

        # Fall back to environment variables
        return cls.from_env()
