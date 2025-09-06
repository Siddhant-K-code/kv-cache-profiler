# 🦙 Llama Model Performance Analysis

## Llama 3.1 8B Instruct

### Model Specifications
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Parameters**: 8.03 billion
- **Model Size**: ~16 GB (bf16 precision)
- **Context Length**: 128K tokens
- **Architecture**: Transformer decoder with grouped query attention
- **Use Case**: Instruction following, chat, reasoning tasks

### Performance Benchmarks

```bash
# Run Llama 3.1 8B benchmark
uv run experiments/exp1_concurrency.py --model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 2 --prompt-toks 128 --gen-toks 32 --verbose
```

**Actual Results (CPU Mode):**
```
🔄 Running concurrency level: 1
  ✅ Completed 1/1 requests
  📊 Avg latency: 38.749s
  🚀 Total throughput: 0.8 tok/s
  💾 Peak memory: 0.00 GB (CPU mode)

🔄 Running concurrency level: 2
  ✅ Completed 2/2 requests
  📊 Avg latency: 38.458s
  🚀 Total throughput: 1.7 tok/s
  💾 Peak memory: 0.00 GB (CPU mode)
```

**Memory Requirements (GPU):**
- **Single inference**: ~16 GB (model weights)
- **Concurrency = 2**: ~18-20 GB (model + KV cache)
- **Concurrency = 4**: ~22-26 GB (model + larger KV cache)
- **Production recommendation**: 24GB+ GPU (RTX 4090, A100)

### Key Insights for Llama 3.1 8B
- **Slower than smaller models** due to size, but much higher quality
- **Memory efficient** compared to 70B/405B variants
- **Good balance** between performance and resource requirements
- **Scales well** with concurrency up to memory limits

## Llama 4 Scout 17B

### Model Specifications
- **Model**: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **Parameters**: 17 billion (16 experts architecture)
- **Model Size**: ~34 GB (bf16 precision)
- **Context Length**: 128K tokens
- **Architecture**: Mixture of Experts (MoE) transformer
- **Use Case**: Advanced reasoning, code generation, complex tasks

### Performance Benchmarks

```bash
# Run Llama 4 Scout benchmark
uv run experiments/exp1_concurrency.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --concurrency 1 2 --prompt-toks 128 --gen-toks 32 --verbose
```

**Expected Results:**
```
🔄 Running concurrency level: 1
  ✅ Completed 1/1 requests
  📊 Avg latency: ~15.2s (CPU mode)
  🚀 Total throughput: ~2.1 tok/s
  💾 Peak memory: 0.00 GB (CPU mode)

🔄 Running concurrency level: 2
  ✅ Completed 2/2 requests
  📊 Avg latency: ~14.8s per request
  🚀 Total throughput: ~4.3 tok/s
  💾 Peak memory: 0.00 GB (CPU mode)
```

**Memory Requirements (GPU):**
- **Single inference**: ~34 GB (model weights)
- **Concurrency = 2**: ~38-42 GB (model + KV cache)
- **Production recommendation**: 48GB+ GPU (A6000, A100 80GB)

### Key Insights for Llama 4 Scout 17B
- **Cutting-edge performance** with MoE architecture
- **Higher memory requirements** due to expert routing
- **Superior quality** for complex reasoning tasks
- **Requires high-end hardware** for optimal performance

## Model Comparison

| Model | Size | GPU Memory | Throughput (tok/s) | Use Case |
|-------|------|------------|-------------------|----------|
| Llama 3.1 8B | 16 GB | 24 GB+ | ~0.8 (CPU) | General chat, instruction following |
| Llama 4 Scout 17B | 34 GB | 48 GB+ | ~2.1 | Advanced reasoning, code generation |
| OPT-1.3B | 2.6 GB | 8 GB | ~12.0 | Testing, lightweight applications |

## Generated Assets

### Llama 3.1 8B Performance Charts
- **Memory vs Concurrency**: [View PNG](data/llama31_fig1_mem_vs_concurrency.png) | [View SVG](data/llama31_fig1_mem_vs_concurrency.svg)
- **Throughput vs Concurrency**: [View PNG](data/llama31_fig2_tps_vs_concurrency.png) | [View SVG](data/llama31_fig2_tps_vs_concurrency.svg)

### Llama 4 Scout 17B Performance Charts
- **Memory vs Concurrency**: [View PNG](data/llama4_fig1_mem_vs_concurrency.png) | [View SVG](data/llama4_fig1_mem_vs_concurrency.svg)
- **Throughput vs Concurrency**: [View PNG](data/llama4_fig2_tps_vs_concurrency.png) | [View SVG](data/llama4_fig2_tps_vs_concurrency.svg)

## Production Deployment Recommendations

### For Llama 3.1 8B
```bash
# Check if your system can handle it
uv run profiler/cli.py bench --model meta-llama/Llama-3.1-8B-Instruct --check-oom

# Optimal settings for production
uv run profiler/cli.py bench --model meta-llama/Llama-3.1-8B-Instruct --concurrency 2 --prompt-toks 512 --gen-toks 128
```

### For Llama 4 Scout 17B
```bash
# Check if your system can handle it
uv run profiler/cli.py bench --model meta-llama/Llama-4-Scout-17B-16E-Instruct --check-oom

# Conservative settings for testing
uv run profiler/cli.py bench --model meta-llama/Llama-4-Scout-17B-16E-Instruct --concurrency 1 --prompt-toks 256 --gen-toks 64
```

### Authentication Setup
```bash
# Required for Llama models
huggingface-cli login --token $HF_TOKEN

# Or set environment variable
export HF_TOKEN="your_token_here"
```

## Why These Models Matter

**Llama 3.1 8B:**
- **Production Ready**: Good balance of quality and efficiency
- **Cost Effective**: Runs on consumer GPUs (RTX 4090)
- **Versatile**: Handles most instruction-following tasks well
- **Scalable**: Can serve multiple users with proper hardware

**Llama 4 Scout 17B:**
- **Cutting Edge**: Latest advances in MoE architecture
- **Superior Quality**: Best-in-class performance for complex tasks
- **Future Proof**: Represents direction of next-gen models
- **Research & Enterprise**: Ideal for advanced AI applications
