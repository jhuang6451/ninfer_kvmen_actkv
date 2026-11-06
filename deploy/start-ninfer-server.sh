#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-$HOME/Models/Qwen3.8-27B/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer}}"
PORT="${2:-18200}"

echo "=========================================================="
echo "🌟 启动 KVMem + ActKV NInfer 推理服务 (RTX 5080 sm_120a)"
echo "模型路径: ${MODEL_PATH}"
echo "监听端口: ${PORT}"
echo "=========================================================="

if [ ! -f "${MODEL_PATH}" ]; then
    echo "⚠️ 警告: 模型文件不存在: ${MODEL_PATH}"
    echo "请先运行 ./convert-model.sh 进行模型无损封装转换。"
    exit 1
fi

MODEL_DIR="$(dirname "${MODEL_PATH}")"
MODEL_NAME="$(basename "${MODEL_PATH}")"

podman run --name ninfer-actkv-server --rm -it \
    --network host \
    --security-opt label=disable \
    --device nvidia.com/gpu=all \
    --cap-add IPC_LOCK \
    -v "${MODEL_DIR}:/models:Z" \
    localhost/ninfer-kvmem-actkv:rtx50 \
    "/models/${MODEL_NAME}" \
    --port "${PORT}" \
    --model-id ninfer-27b \
    --max-context 262144 \
    --kvmem-recent-tokens 1024 \
    --prefill-chunk 256 \
    --kvmem-budget 36864 \
    --kvmem-gen-reserve 16384 \
    --kvmem-host-mib 12288 \
    --kvmem-action-lrfu \
    --kvmem-lrfu-lambda 0.50 \
    --kvmem-lrfu-top-frac 0.90 \
    --kvmem-score-agg mean \
    --kv-dtype rk8v4 \
    --spec mtp --draft-tokens 5 --ngram-draft-tokens 31 --adaptive-mtp \
    --vision --vision-residency cpu --vision-max-merged 256 \
    --default-reasoning-effort xhigh --default-thinking-budget 53248 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
