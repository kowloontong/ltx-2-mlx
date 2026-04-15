# FLUX.2 klein 9B Image Generation

本地运行 FLUX.2 klein 9B 图像生成，使用 Diffusers 库，支持 Apple Silicon MPS。

## 模型下载

⚠️ **需要 HuggingFace 账号**：首次使用需要同意 FLUX Non-Commercial License。

### 1. 注册并获取 Token

1. 注册 HuggingFace 账号: https://huggingface.co/join
2. 登录后获取 Token: https://huggingface.co/settings/tokens
3. 在终端设置环境变量 (可选):
   ```bash
   export HF_TOKEN="your_token_here"
   ```

### 2. 下载模型文件

模型总大小: ~18.2 GB

```bash
# 创建模型目录
mkdir -p models/black-forest-labs/FLUX.2-klein-9B

cd models/black-forest-labs/FLUX.2-klein-9B

# 下载主模型文件 (18.2 GB)
wget https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors

# 或使用 huggingface-cli
# huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir .
```

### 3. 目录结构

下载完成后，目录应该如下:

```
models/black-forest-labs/FLUX.2-klein-9B/
├── flux-2-klein-9b.safetensors    # 主模型 (18.2 GB)
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
└── vae/
```

### 4. 验证下载

```bash
# 检查文件大小
ls -lh flux-2-klein-9b.safetensors
# 应该显示: 18.2G

# MD5 校验 (可选)
md5sum flux-2-klein-9b.safetensors
```

## 安装依赖

```bash
# 使用 uv 安装依赖
cd tools/flux-klein9b
uv sync

# 或手动安装
uv pip install torch torchvision diffusers transformers accelerate safetensors pillow
```

## 使用方法

### 基本用法

```bash
# 进入工具目录
cd tools/flux-klein9b

# 生成图像 (1024x1024)
uv run python generate.py --prompt "a beautiful landscape with mountains" -o landscape.png

# 指定尺寸
uv run python generate.py -p "a cyberpunk city" -o cyberpunk.png -H 768 -W 1344

# 使用随机种子 (可复现)
uv run python generate.py -p "portrait" -o portrait.png --seed 42

# 调整推理步数 (蒸馏模型建议 4 步)
uv run python generate.py -p "still life" -o still_life.png --steps 4
```

### 参数说明

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt` | `-p` | (必填) | 图像描述文本 |
| `--output` | `-o` | `output.png` | 输出路径 |
| `--model-path` | `-m` | `./models/...` | 模型目录路径 |
| `--height` | `-H` | 1024 | 图像高度 |
| `--width` | `-W` | 1024 | 图像宽度 |
| `--steps` | `-s` | 4 | 推理步数 (蒸馏模型用 4) |
| `--guidance` | `-g` | 1.0 | 引导强度 |
| `--seed` | | None | 随机种子 |
| `--no-cpu-offload` | | False | 禁用 CPU 卸载 (更耗显存但更快) |

### 分辨率建议

推荐分辨率 (与训练数据分布一致):
- 1:1  (方形): 1024x1024
- 16:9 (宽屏): 1344x768, 1152x672
- 9:16 (竖屏): 768x1344, 672x1152
- 3:4  (竖向): 1024x1360
- 4:3  (横向): 1360x1024

## Apple Silicon 内存说明

| 模式 | 内存占用 | 说明 |
|------|---------|------|
| CPU Offload (默认) | ~12-16GB unified | 推荐，流畅运行 |
| 无 Offload | ~29GB+ unified | 更快但可能 OOM |

首次运行时，模型加载较慢 (需要编译 MPS 内核)。后续运行会更快。

## License

⚠️ **重要**: FLUX.2 klein 9B 采用 **Non-Commercial License**。

- 不可用于商业用途
- 仅供研究和个人使用
- 详细条款: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md

如需商用，请联系 Black Forest Labs 或使用 FLUX.2 klein 4B (Apache 2.0)。

## 常见问题

### Q: 下载太慢怎么办？
A: 使用 huggingface-cli 或 hf_transfer 加速:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
wget ...
```

### Q: 提示 "You need to agree to..."？
A: 需要在 HuggingFace 网页上同意 License 才能下载 9B 模型。

### Q: MPS 太慢？
A: 可以尝试使用 `--no-cpu-offload`，但需要更多内存。

### Q: 图像生成质量不好？
A: 尝试:
- 增加分辨率
- 调整 prompt (更详细)
- 使用 --steps 8 或更高

## 参考链接

- [FLUX.2 GitHub](https://github.com/black-forest-labs/flux2)
- [Diffusers 文档](https://huggingface.co/docs/diffusers)
- [HuggingFace 模型页](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
