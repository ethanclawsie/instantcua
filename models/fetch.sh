#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/hf models/gguf models/mlx

echo "==> Downloading Qwen/Qwen3-VL-2B-Instruct (HF bf16 safetensors + tokenizer)..."
for f in config.json generation_config.json preprocessor_config.json video_preprocessor_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json chat_template.json model.safetensors; do
    if [ ! -s "models/hf/$f" ]; then
        echo "  fetching $f..."
        curl -fL --retry 3 -o "models/hf/$f.part" "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/$f"
        mv "models/hf/$f.part" "models/hf/$f"
    fi
done

if [ "${FETCH_COMPETITORS:-0}" = "1" ]; then
    echo "==> Downloading GGUF weights + mmproj for llama.cpp competitor benchmark..."
    for f in Qwen3VL-2B-Instruct-Q4_K_M.gguf Qwen3VL-2B-Instruct-Q8_0.gguf Qwen3VL-2B-Instruct-F16.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf; do
        if [ ! -s "models/gguf/$f" ]; then
            echo "  fetching $f..."
            curl -fL --retry 3 -o "models/gguf/$f.part" "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/$f"
            mv "models/gguf/$f.part" "models/gguf/$f"
        fi
    done
fi

echo "==> Done fetching models."
