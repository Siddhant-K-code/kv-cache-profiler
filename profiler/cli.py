"""
Command-line interface for KV Cache Profiler using Typer.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

try:
    from .core import RequestSpec, ProfilingSession
    from .backends.hf import HFBackend
    from .env import print_environment_banner, check_oom_risk
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from profiler.core import RequestSpec, ProfilingSession
    from profiler.backends.hf import HFBackend
    from profiler.env import print_environment_banner, check_oom_risk

app = typer.Typer(
    name="kvprof",
    help="KV Cache Profiler - Profile LLM inference bottlenecks",
    add_completion=False,
)
console = Console()


@app.command()
def bench(
    model: str = typer.Option("facebook/opt-1.3b", "--model", "-m", help="Model name or path"),
    prompt_toks: int = typer.Option(512, "--prompt-toks", "-p", help="Number of prompt tokens"),
    gen_toks: int = typer.Option(64, "--gen-toks", "-g", help="Number of tokens to generate"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Number of concurrent requests"),
    dtype: str = typer.Option("bf16", "--dtype", "-d", help="Model dtype (bf16, fp16, fp32)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file prefix"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    check_oom: bool = typer.Option(True, "--check-oom/--no-check-oom", help="Check OOM risk"),
):
    """Run a basic benchmark with the specified parameters."""

    if verbose:
        print_environment_banner()

    # Check OOM risk
    if check_oom:
        risk = check_oom_risk(model, concurrency, prompt_toks + gen_toks)
        if risk["risk_level"] == "high":
            console.print(f"[red]⚠️  HIGH OOM RISK detected![/red]")
            console.print(f"Estimated memory usage: {risk['total_estimated_gb']:.1f} GB")
            if "available_gpu_memory_gb" in risk:
                console.print(f"Available GPU memory: {risk['available_gpu_memory_gb']:.1f} GB")
            console.print("Recommendations:")
            for rec in risk["recommendations"]:
                console.print(f"  • {rec}")

            if not typer.confirm("Continue anyway?"):
                raise typer.Abort()
        elif risk["risk_level"] == "medium":
            console.print(f"[yellow]⚠️  Medium OOM risk detected[/yellow]")
            console.print(f"Estimated memory usage: {risk['total_estimated_gb']:.1f} GB")

    async def run_benchmark():
        # Create backend
        backend = HFBackend(model=model, dtype=dtype)

        try:
            # Create request spec
            spec = RequestSpec(
                prompt_tokens=prompt_toks,
                generate_tokens=gen_toks,
            )

            console.print(f"[blue]Running benchmark...[/blue]")
            console.print(f"Model: {model}")
            console.print(f"Prompt tokens: {prompt_toks}, Generate tokens: {gen_toks}")
            console.print(f"Concurrency: {concurrency}")

            # Create async task function
            async def run_single(request_id: int):
                return await backend.run(request_id, spec)

            # Run concurrent requests
            try:
                from .core import run_concurrent, calculate_metrics
            except ImportError:
                from profiler.core import run_concurrent, calculate_metrics

            traces = await run_concurrent(concurrency, run_single)

            # Calculate and display results
            metrics = calculate_metrics(traces)

            # Display results table
            display_results(traces, metrics)

            # Save results if output specified
            if output:
                session_name = output or f"bench_{model.replace('/', '_')}"
                with ProfilingSession(session_name) as session:
                    session.add_traces(traces)

        finally:
            backend.cleanup()

    # Run the async benchmark
    asyncio.run(run_benchmark())


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name or path"),
    dtype: str = typer.Option("bf16", "--dtype", "-d", help="Model dtype"),
):
    """Get information about a model without running inference."""

    print_environment_banner()

    async def get_model_info():
        backend = HFBackend(model=model, dtype=dtype)
        try:
            backend.load_model()
            info = backend.get_model_info()

            console.print(f"\n[bold]Model Information: {model}[/bold]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in info.items():
                if value is not None:
                    if key == "total_parameters" and isinstance(value, int):
                        value = f"{value:,} ({value/1e6:.1f}M)"
                    elif key == "trainable_parameters" and isinstance(value, int):
                        value = f"{value:,} ({value/1e6:.1f}M)"
                    table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)

        finally:
            backend.cleanup()

    asyncio.run(get_model_info())


@app.command()
def env():
    """Display environment information."""
    print_environment_banner()


def display_results(traces, metrics):
    """Display benchmark results in a formatted table."""

    # Summary table
    console.print(f"\n[bold]Benchmark Results Summary[/bold]")

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    if metrics:
        summary_table.add_row("Total Requests", str(metrics["total_requests"]))
        summary_table.add_row("Successful", str(metrics["successful_requests"]))
        summary_table.add_row("Failed", str(metrics["failed_requests"]))
        summary_table.add_row("Error Rate", f"{metrics['error_rate']:.1%}")

        if metrics["successful_requests"] > 0:
            summary_table.add_row("Mean Latency", f"{metrics['mean_latency_s']:.3f}s")
            summary_table.add_row("P95 Latency", f"{metrics['p95_latency_s']:.3f}s")
            summary_table.add_row("Mean Throughput", f"{metrics['mean_tokens_per_s']:.1f} tok/s")
            summary_table.add_row("Total Throughput", f"{metrics['total_tokens_per_s']:.1f} tok/s")
            summary_table.add_row("Peak Memory (Alloc)", f"{metrics['peak_alloc_gb']:.2f} GB")
            summary_table.add_row("Peak Memory (Reserved)", f"{metrics['peak_reserved_gb']:.2f} GB")

    console.print(summary_table)

    # Individual traces table (if not too many)
    if len(traces) <= 10:
        console.print(f"\n[bold]Individual Request Details[/bold]")

        traces_table = Table(show_header=True, header_style="bold magenta")
        traces_table.add_column("ID", style="cyan")
        traces_table.add_column("Prompt", style="green")
        traces_table.add_column("Generated", style="green")
        traces_table.add_column("Latency (s)", style="yellow")
        traces_table.add_column("Throughput", style="blue")
        traces_table.add_column("Memory (GB)", style="red")
        traces_table.add_column("Status", style="white")

        for trace in traces:
            status = "✅ OK" if trace.error is None else f"❌ {trace.error[:20]}..."
            memory_str = f"{trace.peak_alloc_gb:.2f}"
            throughput_str = f"{trace.tokens_per_s:.1f}" if trace.error is None else "N/A"

            traces_table.add_row(
                str(trace.request_id),
                str(trace.seq_prompt),
                str(trace.seq_generate),
                f"{trace.total_latency_s:.3f}",
                throughput_str,
                memory_str,
                status,
            )

        console.print(traces_table)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
