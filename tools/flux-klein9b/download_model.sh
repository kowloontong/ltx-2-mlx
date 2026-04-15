#!/bin/bash
#
# FLUX.2 klein 9B 模型下载脚本
# 使用方法: ./download_model.sh
#

set -e

MODEL_DIR="${HOME}/Models/flux-klein-9b"
MODEL_URL="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors"
FILENAME="flux-2-klein-9b.safetensors"
EXPECTED_SIZE="18.2G"

echo "============================================"
echo "FLUX.2 klein 9B 模型下载脚本"
echo "============================================"
echo ""
echo "模型: FLUX.2-klein-9B"
echo "URL:  $MODEL_URL"
echo "目标: $MODEL_DIR/$FILENAME"
echo "大小: $EXPECTED_SIZE"
echo ""

# 检查 wget 或 curl
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
    echo "[错误] 需要 wget 或 curl"
    exit 1
fi

# 创建目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# 检查是否已有文件
if [ -f "$FILENAME" ]; then
    echo "[检测] 文件已存在: $FILENAME"
    ls -lh "$FILENAME"
    read -p "是否重新下载? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "跳过下载"
        exit 0
    fi
    rm "$FILENAME"
fi

# 检查 HuggingFace token
echo ""
echo "============================================"
echo "需要 HuggingFace Token"
echo "============================================"
echo "1. 注册账号: https://huggingface.co/join"
echo "2. 同意协议: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
echo "3. 获取Token: https://huggingface.co/settings/tokens"
echo ""

read -p "请输入 HuggingFace Token (或直接回车跳过，下载可能失败): " HF_TOKEN

# 开始下载
echo ""
echo "[下载] 开始下载..."
echo ""

if [ -n "$HF_TOKEN" ]; then
    echo "[信息] 使用 Token 下载"
    if command -v wget &> /dev/null; then
        wget --header="Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -O "$FILENAME"
    else
        curl -L -H "Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -o "$FILENAME"
    fi
else
    echo "[警告] 未提供 Token，下载可能会失败"
    if command -v wget &> /dev/null; then
        wget "$MODEL_URL" -O "$FILENAME"
    else
        curl -L "$MODEL_URL" -o "$FILENAME"
    fi
fi

# 验证
if [ -f "$FILENAME" ]; then
    echo ""
    echo "[成功] 下载完成!"
    ls -lh "$FILENAME"
    echo ""
    echo "============================================"
    echo "下一步"
    echo "============================================"
    echo "1. 将模型目录添加到项目:"
    echo "   ln -s $MODEL_DIR ./tools/flux-klein9b/models/black-forest-labs/FLUX.2-klein-9B"
    echo ""
    echo "2. 运行生成脚本:"
    echo "   cd tools/flux-klein9b"
    echo "   uv sync"
    echo "   uv run python generate.py -p 'a cat' -o cat.png"
else
    echo ""
    echo "[失败] 下载未完成"
    exit 1
fi
