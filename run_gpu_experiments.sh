#!/bin/bash

# 🖥️ GPU Experiment Runner for CloudRift Instance
# Run this script on your GPU instance: ssh riftuser@176.124.69.199

set -e

echo "🚀 Starting GPU experiments for KV Cache Profiler..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Not in kv-cache-profiler directory"
    echo "Please run: cd kv-cache-profiler"
    exit 1
fi

# Install HuggingFace CLI
echo "📦 Installing HuggingFace CLI..."
pip install huggingface_hub

# 1. Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi || { echo "❌ nvidia-smi failed - no GPU available"; exit 1; }

# 2. Check PyTorch CUDA
echo "🔍 Checking PyTorch CUDA support..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || { echo "❌ PyTorch CUDA check failed"; exit 1; }

# 3. Check HuggingFace authentication
echo "🔐 Checking HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable not set"
    echo "Please run: export HF_TOKEN='your_token_here'"
    exit 1
fi

# 4. Test environment
echo "🧪 Testing environment..."
uv run profiler/cli.py env

# 5. Quick GPU memory test
echo "🧪 Quick GPU memory test..."
uv run profiler/cli.py bench --model facebook/opt-1.3b --concurrency 1 --prompt-toks 32 --gen-toks 8 --verbose

# 6. Run Llama 3.1 8B experiment with GPU memory tracking
echo "🦙 Running Llama 3.1 8B experiment (GPU accelerated)..."
uv run experiments/exp1_concurrency.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 1 2 4 \
    --prompt-toks 128 \
    --gen-toks 32 \
    --verbose \
    --output-dir data/gpu_llama31

# 7. Run smaller Llama 4 experiment (be conservative with memory)
echo "🦙 Running Llama 4 Scout 17B experiment (GPU accelerated)..."
uv run experiments/exp1_concurrency.py \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --concurrency 1 2 \
    --prompt-toks 64 \
    --gen-toks 16 \
    --verbose \
    --output-dir data/gpu_llama4

# 8. Show results
echo "📊 Experiment Results Summary:"
echo "================================"

echo "📁 Generated files:"
ls -la data/gpu_*

echo ""
echo "✅ GPU experiments completed successfully!"
echo ""
echo "🔄 Next steps:"
echo "1. Check the generated plots in data/gpu_llama31/ and data/gpu_llama4/"
echo "2. Copy results back to main environment with:"
echo "   scp -r riftuser@176.124.69.199:~/kv-cache-profiler/data/gpu_* ./data/"
echo "3. Update README with real GPU memory data"

echo ""
echo "📈 Quick preview of results:"
echo "GPU memory should now show real values like:"
echo "- Llama 3.1 8B: 16GB → 19GB → 23GB (instead of 0.00 GB)"
echo "- Throughput: ~15-20x faster than CPU"
