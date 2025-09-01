"""
Experiment 1: Concurrency sweep (memory & throughput)

This experiment tests how memory usage and throughput scale with concurrency levels.
Expected behavior: memory grows ~linearly with N; throughput plateaus/regresses as KV cache dominates.
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
from profiler.plots import plot_memory_vs_concurrency, plot_throughput_vs_concurrency


async def run_concurrency_experiment(
    model: str,
    prompt_tokens: int,
    generate_tokens: int,
    concurrency_levels: list,
    dtype: str = "bf16",
    output_dir: str = "data"
):
    """Run concurrency sweep experiment."""

    print("=" * 60)
    print("Experiment 1: Concurrency Sweep")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generate tokens: {generate_tokens}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Data type: {dtype}")
    print()

    # Create backend
    backend = HFBackend(model=model, dtype=dtype)

    try:
        # Create request spec
        spec = RequestSpec(
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
        )

        # Store results for each concurrency level
        concurrency_data = {}

        with ProfilingSession("exp1_concurrency", output_dir) as session:
            for concurrency in concurrency_levels:
                print(f"\n🔄 Running concurrency level: {concurrency}")

                # Check OOM risk for this concurrency level
                risk = check_oom_risk(model, concurrency, prompt_tokens + generate_tokens)
                if risk["risk_level"] == "high":
                    print(f"⚠️  HIGH OOM RISK for concurrency {concurrency}")
                    print(f"Estimated memory: {risk['total_estimated_gb']:.1f} GB")
                    continue

                # Create async task function
                async def run_single(request_id: int):
                    return await backend.run(request_id, spec)

                # Run concurrent requests
                traces = await run_concurrent(concurrency, run_single)

                # Store traces
                concurrency_data[concurrency] = traces
                session.add_traces(traces)

                # Print summary for this concurrency level
                valid_traces = [t for t in traces if t.error is None]
                if valid_traces:
                    avg_latency = sum(t.total_latency_s for t in valid_traces) / len(valid_traces)
                    total_throughput = sum(t.tokens_per_s for t in valid_traces)
                    max_memory = max(t.peak_alloc_gb for t in valid_traces)

                    print(f"  ✅ Completed {len(valid_traces)}/{len(traces)} requests")
                    print(f"  📊 Avg latency: {avg_latency:.3f}s")
                    print(f"  🚀 Total throughput: {total_throughput:.1f} tok/s")
                    print(f"  💾 Peak memory: {max_memory:.2f} GB")
                else:
                    print(f"  ❌ All requests failed")

        # Generate plots
        print(f"\n📈 Generating plots...")

        # Plot memory vs concurrency
        plot_memory_vs_concurrency(
            concurrency_data,
            output_path=f"{output_dir}/fig1_mem_vs_concurrency",
            model_name=model,
            dtype=dtype
        )

        # Plot throughput vs concurrency
        plot_throughput_vs_concurrency(
            concurrency_data,
            output_path=f"{output_dir}/fig2_tps_vs_concurrency",
            model_name=model,
            dtype=dtype
        )

        print("✅ Experiment 1 completed successfully!")

        # Print summary insights
        print(f"\n📋 Key Insights:")
        print("- Memory usage should grow ~linearly with concurrency")
        print("- Throughput may plateau or degrade due to KV cache overhead")
        print("- Watch for OOM at higher concurrency levels")

    finally:
        backend.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run concurrency sweep experiment")
    parser.add_argument("--model", default="facebook/opt-1.3b", help="Model name")
    parser.add_argument("--prompt-toks", type=int, default=512, help="Prompt tokens")
    parser.add_argument("--gen-toks", type=int, default=64, help="Generate tokens")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8],
                       help="Concurrency levels to test")
    parser.add_argument("--dtype", default="bf16", help="Model dtype")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print_environment_banner()

    # Run the experiment
    asyncio.run(run_concurrency_experiment(
        model=args.model,
        prompt_tokens=args.prompt_toks,
        generate_tokens=args.gen_toks,
        concurrency_levels=args.concurrency,
        dtype=args.dtype,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
