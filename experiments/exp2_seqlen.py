"""
Experiment 2: Sequence length sweep (latency & memory)

This experiment tests how latency and memory usage scale with sequence length.
Expected behavior: latency grows superlinearly (attention), KV grows ~linearly with seq_len.
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.core import RequestSpec, ProfilingSession, run_concurrent
from profiler.backends.hf import HFBackend
from profiler.env import print_environment_banner, check_oom_risk
from profiler.plots import plot_latency_vs_seqlen, plot_memory_vs_seqlen


async def run_seqlen_experiment(
    model: str,
    sequence_lengths: list,
    generate_tokens: int,
    concurrency: int = 1,
    dtype: str = "bf16",
    output_dir: str = "data"
):
    """Run sequence length sweep experiment."""

    print("=" * 60)
    print("Experiment 2: Sequence Length Sweep")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Generate tokens: {generate_tokens}")
    print(f"Concurrency: {concurrency}")
    print(f"Data type: {dtype}")
    print()

    # Create backend
    backend = HFBackend(model=model, dtype=dtype)

    try:
        # Store results for each sequence length
        seqlen_data = {}

        with ProfilingSession("exp2_seqlen", output_dir) as session:
            for seq_len in sequence_lengths:
                print(f"\n🔄 Running sequence length: {seq_len}")

                # Check OOM risk for this sequence length
                risk = check_oom_risk(model, concurrency, seq_len + generate_tokens)
                if risk["risk_level"] == "high":
                    print(f"⚠️  HIGH OOM RISK for sequence length {seq_len}")
                    print(f"Estimated memory: {risk['total_estimated_gb']:.1f} GB")
                    continue

                # Create request spec
                spec = RequestSpec(
                    prompt_tokens=seq_len,
                    generate_tokens=generate_tokens,
                )

                # Create async task function
                async def run_single(request_id: int):
                    return await backend.run(request_id, spec)

                # Run concurrent requests
                traces = await run_concurrent(concurrency, run_single)

                # Store traces
                seqlen_data[seq_len] = traces
                session.add_traces(traces)

                # Print summary for this sequence length
                valid_traces = [t for t in traces if t.error is None]
                if valid_traces:
                    avg_latency = sum(t.total_latency_s for t in valid_traces) / len(valid_traces)
                    avg_throughput = sum(t.tokens_per_s for t in valid_traces) / len(valid_traces)
                    max_memory = max(t.peak_alloc_gb for t in valid_traces)

                    print(f"  ✅ Completed {len(valid_traces)}/{len(traces)} requests")
                    print(f"  📊 Avg latency: {avg_latency:.3f}s")
                    print(f"  🚀 Avg throughput: {avg_throughput:.1f} tok/s")
                    print(f"  💾 Peak memory: {max_memory:.2f} GB")
                else:
                    print(f"  ❌ All requests failed")

        # Generate plots
        print(f"\n📈 Generating plots...")

        # Plot latency vs sequence length
        plot_latency_vs_seqlen(
            seqlen_data,
            output_path=f"{output_dir}/fig3_latency_vs_seqlen",
            model_name=model,
            dtype=dtype
        )

        # Plot memory vs sequence length
        plot_memory_vs_seqlen(
            seqlen_data,
            output_path=f"{output_dir}/fig4_mem_vs_seqlen",
            model_name=model,
            dtype=dtype
        )

        print("✅ Experiment 2 completed successfully!")

        # Print summary insights
        print(f"\n📋 Key Insights:")
        print("- Latency should grow superlinearly with sequence length (O(n²) attention)")
        print("- Memory usage should grow ~linearly with sequence length")
        print("- Longer sequences have higher KV cache overhead")

    finally:
        backend.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run sequence length sweep experiment")
    parser.add_argument("--model", default="facebook/opt-1.3b", help="Model name")
    parser.add_argument("--seqlens", nargs="+", type=int, default=[128, 512, 1024, 2048, 4096],
                       help="Sequence lengths to test")
    parser.add_argument("--gen-toks", type=int, default=64, help="Generate tokens")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level")
    parser.add_argument("--dtype", default="bf16", help="Model dtype")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print_environment_banner()

    # Run the experiment
    asyncio.run(run_seqlen_experiment(
        model=args.model,
        sequence_lengths=args.seqlens,
        generate_tokens=args.gen_toks,
        concurrency=args.concurrency,
        dtype=args.dtype,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
