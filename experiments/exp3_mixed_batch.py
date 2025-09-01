"""
Experiment 3: Mixed lengths (tail latency pathology)

This experiment demonstrates how short requests are "punished" when batched with long ones.
Expected behavior: short requests have higher latency when run concurrently with long requests.
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
from profiler.plots import plot_mixed_lengths_comparison


async def run_mixed_batch_experiment(
    model: str,
    short_tokens: int,
    long_tokens: int,
    generate_tokens: int,
    dtype: str = "bf16",
    output_dir: str = "data"
):
    """Run mixed batch experiment."""

    print("=" * 60)
    print("Experiment 3: Mixed Batch Lengths")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Short sequence: {short_tokens} tokens")
    print(f"Long sequence: {long_tokens} tokens")
    print(f"Generate tokens: {generate_tokens}")
    print(f"Data type: {dtype}")
    print()

    # Create backend
    backend = HFBackend(model=model, dtype=dtype)

    try:
        # Store results for each scenario
        short_traces = []
        long_traces = []
        mixed_traces = []

        with ProfilingSession("exp3_mixed_batch", output_dir) as session:

            # Scenario 1: Run short requests separately
            print(f"\n🔄 Running short requests separately ({short_tokens} tokens)")

            short_spec = RequestSpec(
                prompt_tokens=short_tokens,
                generate_tokens=generate_tokens,
            )

            async def run_short(request_id: int):
                return await backend.run(request_id, short_spec)

            short_traces = await run_concurrent(2, run_short)  # Run 2 short requests
            session.add_traces(short_traces)

            # Print summary
            valid_short = [t for t in short_traces if t.error is None]
            if valid_short:
                avg_latency = sum(t.total_latency_s for t in valid_short) / len(valid_short)
                print(f"  ✅ Short requests avg latency: {avg_latency:.3f}s")

            # Scenario 2: Run long requests separately
            print(f"\n🔄 Running long requests separately ({long_tokens} tokens)")

            # Check OOM risk for long requests
            risk = check_oom_risk(model, 2, long_tokens + generate_tokens)
            if risk["risk_level"] == "high":
                print(f"⚠️  HIGH OOM RISK for long requests")
                print(f"Estimated memory: {risk['total_estimated_gb']:.1f} GB")
                print("Skipping long request scenario")
                long_traces = []
            else:
                long_spec = RequestSpec(
                    prompt_tokens=long_tokens,
                    generate_tokens=generate_tokens,
                )

                async def run_long(request_id: int):
                    return await backend.run(request_id, long_spec)

                long_traces = await run_concurrent(2, run_long)  # Run 2 long requests
                session.add_traces(long_traces)

                # Print summary
                valid_long = [t for t in long_traces if t.error is None]
                if valid_long:
                    avg_latency = sum(t.total_latency_s for t in valid_long) / len(valid_long)
                    print(f"  ✅ Long requests avg latency: {avg_latency:.3f}s")

            # Scenario 3: Run mixed batch (1 short + 1 long concurrently)
            print(f"\n🔄 Running mixed batch (1 short + 1 long concurrently)")

            if not long_traces:  # Skip if long requests failed
                print("Skipping mixed batch due to long request OOM risk")
                mixed_traces = []
            else:
                async def run_mixed_batch():
                    # Create tasks for both short and long requests
                    short_task = asyncio.create_task(backend.run(0, short_spec))
                    long_task = asyncio.create_task(backend.run(1, long_spec))

                    # Wait for both to complete
                    results = await asyncio.gather(short_task, long_task, return_exceptions=True)
                    return results

                mixed_results = await run_mixed_batch()

                # Convert results to traces
                for i, result in enumerate(mixed_results):
                    if isinstance(result, Exception):
                        # Create error trace
                        from profiler.core import RunTrace
                        import time
                        error_trace = RunTrace(
                            request_id=i,
                            seq_prompt=0,
                            seq_generate=0,
                            total_latency_s=0.0,
                            tokens_per_s=0.0,
                            peak_alloc_bytes=0,
                            peak_reserved_bytes=0,
                            backend="huggingface",
                            model=model,
                            timestamp=time.time(),
                            error=str(result)
                        )
                        mixed_traces.append(error_trace)
                    else:
                        mixed_traces.append(result)

                session.add_traces(mixed_traces)

                # Print summary
                valid_mixed = [t for t in mixed_traces if t.error is None]
                if valid_mixed:
                    # Separate short and long from mixed batch
                    mixed_short = [t for t in valid_mixed if t.seq_prompt <= short_tokens * 1.1]  # Allow some tolerance
                    mixed_long = [t for t in valid_mixed if t.seq_prompt > short_tokens * 1.1]

                    if mixed_short:
                        avg_short_mixed = sum(t.total_latency_s for t in mixed_short) / len(mixed_short)
                        print(f"  ✅ Short in mixed batch avg latency: {avg_short_mixed:.3f}s")

                    if mixed_long:
                        avg_long_mixed = sum(t.total_latency_s for t in mixed_long) / len(mixed_long)
                        print(f"  ✅ Long in mixed batch avg latency: {avg_long_mixed:.3f}s")

        # Generate plots
        print(f"\n📈 Generating plots...")

        if short_traces and long_traces and mixed_traces:
            plot_mixed_lengths_comparison(
                short_traces,
                long_traces,
                mixed_traces,
                output_path=f"{output_dir}/fig5_mixed_lengths_latency",
                model_name=model,
                dtype=dtype
            )
        else:
            print("⚠️  Insufficient data for plotting (some scenarios failed)")

        print("✅ Experiment 3 completed successfully!")

        # Print summary insights
        print(f"\n📋 Key Insights:")
        print("- Short requests should have higher latency when batched with long ones")
        print("- This demonstrates the 'tail latency pathology' in mixed batches")
        print("- Consider batching requests by similar lengths to avoid this issue")

        # Print detailed comparison if data is available
        if short_traces and mixed_traces:
            valid_short_separate = [t for t in short_traces if t.error is None and t.seq_prompt <= short_tokens * 1.1]
            valid_short_mixed = [t for t in mixed_traces if t.error is None and t.seq_prompt <= short_tokens * 1.1]

            if valid_short_separate and valid_short_mixed:
                separate_latency = sum(t.total_latency_s for t in valid_short_separate) / len(valid_short_separate)
                mixed_latency = sum(t.total_latency_s for t in valid_short_mixed) / len(valid_short_mixed)

                latency_increase = ((mixed_latency - separate_latency) / separate_latency) * 100
                print(f"\n📊 Short request latency increase in mixed batch: {latency_increase:.1f}%")

    finally:
        backend.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run mixed batch experiment")
    parser.add_argument("--model", default="facebook/opt-1.3b", help="Model name")
    parser.add_argument("--short", type=int, default=128, help="Short sequence length")
    parser.add_argument("--long", type=int, default=4096, help="Long sequence length")
    parser.add_argument("--gen-toks", type=int, default=64, help="Generate tokens")
    parser.add_argument("--dtype", default="bf16", help="Model dtype")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print_environment_banner()

    # Run the experiment
    asyncio.run(run_mixed_batch_experiment(
        model=args.model,
        short_tokens=args.short,
        long_tokens=args.long,
        generate_tokens=args.gen_toks,
        dtype=args.dtype,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
