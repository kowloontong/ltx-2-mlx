#!/usr/bin/env python3
"""
环境验证脚本 - 检查依赖是否正确安装
使用方法: uv run python test_env.py
"""

import sys


def check_torch():
    """检查 PyTorch 和 MPS 支持."""
    import torch

    print(f"[PyTorch] {torch.__version__}")

    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name()}")
        return True
    elif torch.backends.mps.is_available():
        print(f"[GPU] MPS (Apple Silicon): {torch.backends.mps.is_available()}")
        print(f"[Info] Metal GPU available: {torch.backends.mps.is_built()}")
        return True
    else:
        print("[GPU] No GPU acceleration detected (CPU mode)")
        return False


def check_diffusers():
    """检查 Diffusers 版本."""
    import diffusers

    print(f"[Diffusers] {diffusers.__version__}")
    return True


def check_transformers():
    """检查 Transformers."""
    import transformers

    print(f"[Transformers] {transformers.__version__}")
    return True


def check_model_path(path: str = "./models/black-forest-labs/FLUX.2-klein-9B"):
    """检查模型是否存在."""
    import os
    from pathlib import Path

    model_path = Path(path)

    if not model_path.exists():
        print(f"[Model] ❌ Not found at {model_path}")
        print(f"[Info] Expected structure:")
        print(f"       {model_path}/")
        print(f"       ├── flux-2-klein-9b.safetensors")
        print(f"       ├── model_index.json")
        print(f"       ├── scheduler/")
        print(f"       ├── text_encoder/")
        print(f"       ├── tokenizer/")
        print(f"       └── vae/")
        return False

    main_file = model_path / "flux-2-klein-9b.safetensors"
    if not main_file.exists():
        print(f"[Model] ❌ Main file not found: {main_file}")
        return False

    size = main_file.stat().st_size / (1024**3)
    print(f"[Model] ✅ Found at {model_path}")
    print(f"[Model] File size: {size:.1f} GB")

    if size < 15:
        print(f"[Warning] File seems too small, may be corrupted")
        return False

    return True


def main():
    print("=" * 50)
    print("FLUX.2 klein 9B 环境验证")
    print("=" * 50)
    print()

    checks = []

    print("[1/4] 检查 PyTorch...")
    checks.append(check_torch())
    print()

    print("[2/4] 检查 Diffusers...")
    checks.append(check_diffusers())
    print()

    print("[3/4] 检查 Transformers...")
    checks.append(check_transformers())
    print()

    print("[4/4] 检查模型文件...")
    import os

    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "black-forest-labs", "FLUX.2-klein-9B"
    )
    model_found = check_model_path(default_path)
    checks.append(model_found)
    print()

    print("=" * 50)
    if all(checks):
        print("✅ 所有检查通过! 可以开始生成图像")
        print()
        print("下一步:")
        print(f"  uv run python generate.py -p 'hello world' -o test.png")
        return 0
    else:
        print("❌ 部分检查失败，请修复后再试")
        if not model_found:
            print()
            print("模型下载:")
            print("  1. 创建目录: mkdir -p " + os.path.dirname(default_path))
            print("  2. 下载模型: huggingface-cli download black-forest-labs/FLUX.2-klein-9B")
            print("  或运行: ./download_model.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
