#!/bin/sh
# Prepare an A100 Colab VM for the pipeline. Expects the repo at /content/repo.tar.gz.
set -e
REV=57c125dfa29f2d39157e280b1d1f7480b8736d8d

rm -rf /content/lfm3
mkdir -p /content/lfm3
tar xzf /content/repo.tar.gz -C /content/lfm3
pip install -q transformers jinja2 onnx onnxruntime-gpu

mkdir -p /content/lfm3/graph
curl -sL -o /content/lfm3/graph/model_q4f16.onnx \
  "https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/resolve/${REV}/onnx/model_q4f16.onnx"
curl -sL -o /content/lfm3/graph/model_q4f16.onnx_data \
  "https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/resolve/${REV}/onnx/model_q4f16.onnx_data"

cd /content/lfm3
python tools/build_steer_graph.py --input graph/model_q4f16.onnx --output graph/lfm3-steer.onnx
echo DONE > /content/setup.done
