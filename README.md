# 🔧 KV Cache Profiler

A zero-setup tool to **simulate**, **measure**, and **visualize** KV-driven bottlenecks in LLM inference workloads. Profile memory usage and performance across **concurrency** and **sequence length** with optional comparisons to **paged KV** engines (vLLM) and **local LLMs** (Ollama).

> **"wrk for LLM inference (KV edition)"** — one command, clean plots.

## 🚀 Quick Start

```bash
# Setup (requires uv - fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync   # creates .venv and installs deps from pyproject/uv.lock

# Exp1: concurrency sweep
uv run experiments/exp1_concurrency.py \
  --model facebook/opt-1.3b --prompt-toks 512 --gen-toks 64 --concurrency 1 2 4 8

# Exp2: seqlen sweep
uv run experiments/exp2_seqlen.py \
  --model facebook/opt-1.3b --seqlens 128 512 1024 2048 4096 --gen-toks 64

# Exp3: mixed lengths (tail latency)
uv run experiments/exp3_mixed_batch.py \
  --model facebook/opt-1.3b --short 128 --long 4096 --gen-toks 64
```

## 📊 What You Get

**Publication-grade plots** (PNG + SVG) showing:

- **fig1_mem_vs_concurrency**: Peak memory usage vs concurrency levels
- **fig2_tps_vs_concurrency**: Aggregate throughput vs concurrency
- **fig3_latency_vs_seqlen**: Latency scaling with sequence length
- **fig4_mem_vs_seqlen**: Memory scaling with sequence length
- **fig5_mixed_lengths_latency**: Tail latency pathology in mixed batches

**JSON/CSV data** for further analysis and custom plotting.

## 🧠 Key Insights

### Memory & Concurrency
- Memory grows ~linearly with concurrency
- Throughput plateaus/degrades as KV cache dominates
- OOM risk increases dramatically with concurrent long sequences

### Sequence Length Impact
- Latency grows superlinearly (O(n²) attention complexity)
- KV cache memory grows ~linearly with sequence length
- Longer sequences amplify memory pressure

### Mixed Batch Pathology
- Short requests get "punished" when batched with long ones
- Demonstrates tail latency issues in production serving
- Solution: batch by similar sequence lengths

## 🛠️ CLI Usage

### Basic Benchmarking
```bash
# Quick benchmark
uv run profiler/cli.py bench --model facebook/opt-1.3b --concurrency 4

# With custom parameters
uv run profiler/cli.py bench \
  --model facebook/opt-1.3b \
  --prompt-toks 1024 \
  --gen-toks 128 \
  --concurrency 8 \
  --dtype bf16 \
  --output results

# Get model info
uv run profiler/cli.py info facebook/opt-1.3b

# Check environment
uv run profiler/cli.py env
```

### Advanced Experiments
```bash
# Concurrency sweep with custom levels
uv run experiments/exp1_concurrency.py \
  --model facebook/opt-2.7b \
  --concurrency 1 2 4 8 16 \
  --prompt-toks 1024 \
  --gen-toks 64

# Sequence length sweep
uv run experiments/exp2_seqlen.py \
  --model facebook/opt-1.3b \
  --seqlens 64 128 256 512 1024 2048 4096 8192 \
  --gen-toks 64

# Mixed batch analysis
uv run experiments/exp3_mixed_batch.py \
  --model facebook/opt-1.3b \
  --short 64 \
  --long 2048 \
  --gen-toks 32
```

## 🏗️ Architecture

```
kv-cache-profiler/
├─ profiler/
│  ├─ __init__.py
│  ├─ env.py               # GPU/env introspection (CUDA, VRAM, libs)
│  ├─ core.py              # RequestSpec, RunTrace, async harness, metrics
│  ├─ backends/
│  │  ├─ hf.py             # Hugging Face (baseline)
│  │  ├─ vllm.py           # vLLM (optional)
│  │  └─ ollama.py         # Ollama (optional)
│  ├─ plots.py             # matplotlib helpers, style presets
│  └─ cli.py               # Typer CLI (kvprof …)
├─ experiments/
│  ├─ exp1_concurrency.py
│  ├─ exp2_seqlen.py
│  ├─ exp3_mixed_batch.py
│  └─ exp4_vllm_compare.py     # optional
├─ modal/
│  └─ run_remote.py            # Minimal Modal 1.0 wrapper
├─ data/                        # outputs (gitignored if large)
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ Makefile
└─ LICENSE
```

## 🔬 Core Components

### Data Structures
- **`RequestSpec`**: Defines prompt/generate tokens, temperature
- **`RunTrace`**: Captures latency, throughput, memory usage per request
- **`MemoryTracker`**: Context manager for GPU memory profiling

### Backends
- **HuggingFace**: Baseline implementation using `transformers`
- **vLLM**: Optional paged-KV comparison (install with `uv add vllm`)
- **Ollama**: Optional local LLM testing (install with `uv add ollama`)

### Async Harness
- Concurrent request execution with proper error handling
- Memory tracking during inference
- Configurable delays between request starts

## 📈 Plotting Features

- **Publication-grade styling** with matplotlib
- **Dual format output**: PNG (300 DPI) + SVG for crisp scaling
- **Automatic titles** with model name and dtype
- **Grid styling** and consistent color schemes
- **Error handling** for missing data scenarios

## 🚨 Safety Features

### OOM Risk Assessment
```python
from profiler.env import check_oom_risk

risk = check_oom_risk("facebook/opt-1.3b", concurrency=8, seq_len=2048)
if risk["risk_level"] == "high":
    print(f"⚠️ Estimated memory: {risk['total_estimated_gb']:.1f} GB")
    print("Recommendations:", risk["recommendations"])
```

### Environment Introspection
```python
from profiler.env import print_environment_banner

print_environment_banner()
# Outputs:
# ============================================================
# KV Cache Profiler - Environment Information
# ============================================================
# Platform: Linux-6.14.0-x86_64-with-glibc2.35
# Python: 3.13.7
# CUDA: 12.6 (1 device(s))
#   - NVIDIA GeForce RTX 4090: 24.0 GB
# PyTorch: 2.7.1
# Transformers: 4.56.0
# ============================================================
```

## 🎯 Use Cases

### Development & Optimization
- **Profile before scaling**: Understand memory/latency characteristics
- **Compare models**: Memory efficiency across different architectures
- **Batch size tuning**: Find optimal concurrency levels
- **Sequence length limits**: Identify practical context windows

### Production Planning
- **Capacity planning**: Estimate GPU requirements for target workloads
- **SLA validation**: Measure tail latencies under realistic conditions
- **Cost optimization**: Balance throughput vs memory usage
- **Architecture decisions**: Compare inference engines (HF vs vLLM)

### Research & Analysis
- **KV cache studies**: Visualize memory scaling patterns
- **Attention complexity**: Measure quadratic scaling effects
- **Batching strategies**: Quantify mixed-length penalties
- **Hardware utilization**: GPU memory vs compute trade-offs

## 🔧 Advanced Configuration

### Custom Models
```bash
# Local model
uv run profiler/cli.py bench --model /path/to/local/model

# Custom HuggingFace model
uv run profiler/cli.py bench --model microsoft/DialoGPT-large

# Different precision
uv run profiler/cli.py bench --model facebook/opt-1.3b --dtype fp16
```

### Output Control
```bash
# Custom output directory
uv run experiments/exp1_concurrency.py --output-dir ./results

# Verbose logging
uv run experiments/exp1_concurrency.py --verbose

# Skip OOM checks (dangerous!)
uv run profiler/cli.py bench --no-check-oom
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce concurrency or sequence length
uv run profiler/cli.py bench --concurrency 2 --prompt-toks 256

# Check available memory first
uv run profiler/cli.py env
```

**Model Loading Errors**
```bash
# Verify model exists
uv run profiler/cli.py info facebook/opt-1.3b

# Try different dtype
uv run profiler/cli.py bench --model facebook/opt-1.3b --dtype fp32
```

**Import Errors**
```bash
# Reinstall dependencies
uv sync --reinstall

# Check Python version (requires 3.8+)
python --version
```

### Performance Tips

1. **Start small**: Begin with small models and low concurrency
2. **Monitor memory**: Use `nvidia-smi` to watch GPU utilization
3. **Batch intelligently**: Group similar sequence lengths
4. **Use appropriate dtype**: bf16 for modern GPUs, fp16 for older ones

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone https://github.com/your-org/kv-cache-profiler.git
cd kv-cache-profiler
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run ruff format .
uv run ruff check . --fix

# Type checking
uv run mypy profiler/
```

### Adding New Backends
1. Create `profiler/backends/your_backend.py`
2. Implement `YourBackend` class with `run()` method
3. Add to `profiler/backends/__init__.py`
4. Update CLI and experiments as needed

### Adding New Experiments
1. Create `experiments/exp_your_experiment.py`
2. Follow existing patterns for argument parsing
3. Use `ProfilingSession` for automatic data saving
4. Add corresponding plot functions to `profiler/plots.py`

## 📚 References

- **uv**: [Fast Python package manager](https://docs.astral.sh/uv/)
- **Modal**: [Serverless GPU compute](https://modal.com/docs/)
- **vLLM**: [PagedAttention for efficient serving](https://github.com/vllm-project/vllm)
- **Ollama**: [Local LLM runtime](https://ollama.com/)
- **Transformers**: [HuggingFace model library](https://huggingface.co/docs/transformers/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with modern Python tooling:
- **uv** for blazing-fast package management
- **Typer** for beautiful CLIs
- **Rich** for terminal formatting
- **Matplotlib** for publication-grade plots
- **PyTorch** for GPU acceleration
- **Transformers** for model loading

---

**Happy profiling!** 🚀 Found a bottleneck? Now you can see it, measure it, and fix it.
