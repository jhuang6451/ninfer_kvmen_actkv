#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${1:-${MODELS_DIR:-$HOME/Models/Qwen3.8-27B}}"
GGUF_FILE="${MODELS_DIR}/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
MMPROJ_FILE="${MODELS_DIR}/mmproj-F16.gguf"
OUTPUT_NINFER="${MODELS_DIR}/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer"

if [ ! -f "${GGUF_FILE}" ]; then
    echo "❌ 错误: 找不到 GGUF 模型文件: ${GGUF_FILE}"
    exit 1
fi

echo "=========================================================="
echo "🔄 启动 GSQ-RCO GGUF -> .ninfer 无损封装转换"
echo "源模型: ${GGUF_FILE}"
echo "视觉投影: ${MMPROJ_FILE}"
echo "输出目标: ${OUTPUT_NINFER}"
echo "=========================================================="

podman run --rm -it \
    --security-opt label=disable \
    -v "${ROOT_DIR}/../backends/ninfer:/workspace:Z" \
    -v "${MODELS_DIR}:/models:Z" \
    -w /workspace \
    localhost/ninfer-builder:cuda13.3 \
    bash -c "pip install --no-cache-dir gguf numpy torch --break-system-packages 2>/dev/null || pip install gguf numpy torch && \
    python3 -m tools.convert \
      --recipe qwen3_8_27b_gguf \
      --source gguf=/models/$(basename "${GGUF_FILE}") \
      --source vision=/models/$(basename "${MMPROJ_FILE}") \
      --components text,vision,mtp \
      --device cpu \
      --out /models/$(basename "${OUTPUT_NINFER}")"

echo "✅ 模型无损封装完成: ${OUTPUT_NINFER}"
