"""
Plotting functionality for KV cache profiling results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from .core import RunTrace, calculate_metrics


# Set publication-grade style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


def setup_plot_style():
    """Set up publication-grade plot styling."""
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True


def save_figure(fig, filepath: str, title_suffix: str = ""):
    """Save figure in both PNG and SVG formats."""
    base_path = Path(filepath).with_suffix('')

    # Add title suffix if provided
    if title_suffix:
        fig.suptitle(f"{fig._suptitle.get_text()} ({title_suffix})" if fig._suptitle else title_suffix)

    # Save PNG
    png_path = base_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Save SVG
    svg_path = base_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')

    print(f"Saved plots: {png_path} and {svg_path}")
    return png_path, svg_path


def plot_memory_vs_concurrency(
    concurrency_data: Dict[int, List[RunTrace]],
    output_path: str = "fig1_mem_vs_concurrency",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """
    Plot peak memory usage vs concurrency.

    Args:
        concurrency_data: Dict mapping concurrency levels to traces
        output_path: Base output path for files
        model_name: Model name for title
        dtype: Data type for title

    Returns:
        Tuple of (PNG path, SVG path)
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    concurrency_levels = sorted(concurrency_data.keys())
    peak_alloc_gb = []
    peak_reserved_gb = []

    for conc in concurrency_levels:
        traces = concurrency_data[conc]
        valid_traces = [t for t in traces if t.error is None]

        if valid_traces:
            max_alloc = max(t.peak_alloc_gb for t in valid_traces)
            max_reserved = max(t.peak_reserved_gb for t in valid_traces)
        else:
            max_alloc = 0
            max_reserved = 0

        peak_alloc_gb.append(max_alloc)
        peak_reserved_gb.append(max_reserved)

    # Plot allocated memory
    ax1.plot(concurrency_levels, peak_alloc_gb, 'o-', color='#2E86AB', linewidth=3, markersize=8)
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Peak Allocated Memory (GB)')
    ax1.set_title('Peak Allocated Memory vs Concurrency')
    ax1.grid(True, alpha=0.3)

    # Plot reserved memory
    ax2.plot(concurrency_levels, peak_reserved_gb, 'o-', color='#A23B72', linewidth=3, markersize=8)
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Peak Reserved Memory (GB)')
    ax2.set_title('Peak Reserved Memory vs Concurrency')
    ax2.grid(True, alpha=0.3)

    # Overall title
    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    fig.suptitle('Memory Usage vs Concurrency', fontsize=18, fontweight='bold')

    plt.tight_layout()
    return save_figure(fig, output_path, title_suffix)


def plot_throughput_vs_concurrency(
    concurrency_data: Dict[int, List[RunTrace]],
    output_path: str = "fig2_tps_vs_concurrency",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Plot aggregate throughput vs concurrency."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    concurrency_levels = sorted(concurrency_data.keys())
    total_throughput = []
    mean_throughput = []

    for conc in concurrency_levels:
        traces = concurrency_data[conc]
        valid_traces = [t for t in traces if t.error is None]

        if valid_traces:
            total_tps = sum(t.tokens_per_s for t in valid_traces)
            mean_tps = total_tps / len(valid_traces)
        else:
            total_tps = 0
            mean_tps = 0

        total_throughput.append(total_tps)
        mean_throughput.append(mean_tps)

    # Plot both total and mean throughput
    ax.plot(concurrency_levels, total_throughput, 'o-', label='Total Throughput',
            color='#F18F01', linewidth=3, markersize=8)
    ax.plot(concurrency_levels, mean_throughput, 's--', label='Mean Throughput',
            color='#C73E1D', linewidth=2, markersize=6)

    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput vs Concurrency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)


def plot_latency_vs_seqlen(
    seqlen_data: Dict[int, List[RunTrace]],
    output_path: str = "fig3_latency_vs_seqlen",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Plot latency vs sequence length."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    seq_lengths = sorted(seqlen_data.keys())
    mean_latencies = []
    p95_latencies = []

    for seq_len in seq_lengths:
        traces = seqlen_data[seq_len]
        valid_traces = [t for t in traces if t.error is None]

        if valid_traces:
            latencies = [t.total_latency_s for t in valid_traces]
            mean_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
        else:
            mean_lat = 0
            p95_lat = 0

        mean_latencies.append(mean_lat)
        p95_latencies.append(p95_lat)

    ax.plot(seq_lengths, mean_latencies, 'o-', label='Mean Latency',
            color='#2E86AB', linewidth=3, markersize=8)
    ax.plot(seq_lengths, p95_latencies, 's--', label='P95 Latency',
            color='#A23B72', linewidth=2, markersize=6)

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)


def plot_memory_vs_seqlen(
    seqlen_data: Dict[int, List[RunTrace]],
    output_path: str = "fig4_mem_vs_seqlen",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Plot memory usage vs sequence length."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    seq_lengths = sorted(seqlen_data.keys())
    peak_alloc_gb = []
    peak_reserved_gb = []

    for seq_len in seq_lengths:
        traces = seqlen_data[seq_len]
        valid_traces = [t for t in traces if t.error is None]

        if valid_traces:
            max_alloc = max(t.peak_alloc_gb for t in valid_traces)
            max_reserved = max(t.peak_reserved_gb for t in valid_traces)
        else:
            max_alloc = 0
            max_reserved = 0

        peak_alloc_gb.append(max_alloc)
        peak_reserved_gb.append(max_reserved)

    ax.plot(seq_lengths, peak_alloc_gb, 'o-', label='Peak Allocated',
            color='#2E86AB', linewidth=3, markersize=8)
    ax.plot(seq_lengths, peak_reserved_gb, 's--', label='Peak Reserved',
            color='#A23B72', linewidth=2, markersize=6)

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Memory Usage vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)


def plot_mixed_lengths_comparison(
    short_traces: List[RunTrace],
    long_traces: List[RunTrace],
    mixed_traces: List[RunTrace],
    output_path: str = "fig5_mixed_lengths_latency",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Plot comparison of separate vs concurrent mixed length requests."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate metrics for each scenario
    scenarios = ['Short (Separate)', 'Long (Separate)', 'Mixed (Concurrent)']
    mean_latencies = []

    for traces in [short_traces, long_traces, mixed_traces]:
        valid_traces = [t for t in traces if t.error is None]
        if valid_traces:
            mean_lat = np.mean([t.total_latency_s for t in valid_traces])
        else:
            mean_lat = 0
        mean_latencies.append(mean_lat)

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(scenarios, mean_latencies, color=colors, alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar, value in zip(bars, mean_latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Mean Latency (seconds)')
    ax.set_title('Latency Comparison: Separate vs Mixed Batch')
    ax.grid(True, alpha=0.3, axis='y')

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)


def plot_backend_comparison(
    backend_data: Dict[str, List[RunTrace]],
    output_path: str = "fig6_mem_vs_conc_hf_vs_vllm",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Plot memory usage comparison between backends."""
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    backends = list(backend_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Extract concurrency levels (assuming all backends have same levels)
    first_backend = backends[0]
    concurrency_levels = sorted(set(t.request_id for t in backend_data[first_backend]))

    # Plot memory usage
    for i, backend in enumerate(backends):
        traces = backend_data[backend]

        # Group by concurrency (using request_id as proxy)
        conc_memory = {}
        for trace in traces:
            conc = trace.request_id  # Assuming request_id represents concurrency level
            if conc not in conc_memory:
                conc_memory[conc] = []
            conc_memory[conc].append(trace)

        conc_levels = sorted(conc_memory.keys())
        peak_alloc = []
        peak_reserved = []

        for conc in conc_levels:
            valid_traces = [t for t in conc_memory[conc] if t.error is None]
            if valid_traces:
                max_alloc = max(t.peak_alloc_gb for t in valid_traces)
                max_reserved = max(t.peak_reserved_gb for t in valid_traces)
            else:
                max_alloc = 0
                max_reserved = 0

            peak_alloc.append(max_alloc)
            peak_reserved.append(max_reserved)

        # Plot allocated memory
        ax1.plot(conc_levels, peak_alloc, 'o-', label=f'{backend} (Allocated)',
                color=colors[i % len(colors)], linewidth=2, markersize=6)

        # Plot reserved memory
        ax2.plot(conc_levels, peak_reserved, 's--', label=f'{backend} (Reserved)',
                color=colors[i % len(colors)], linewidth=2, markersize=6)

    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Peak Allocated Memory (GB)')
    ax1.set_title('Peak Allocated Memory by Backend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Peak Reserved Memory (GB)')
    ax2.set_title('Peak Reserved Memory by Backend')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Backend Memory Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)


def load_traces_from_json(filepath: str) -> List[RunTrace]:
    """Load traces from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    traces = []
    for trace_dict in data.get('traces', []):
        trace = RunTrace(
            request_id=trace_dict['request_id'],
            seq_prompt=trace_dict['seq_prompt'],
            seq_generate=trace_dict['seq_generate'],
            total_latency_s=trace_dict['total_latency_s'],
            tokens_per_s=trace_dict['tokens_per_s'],
            peak_alloc_bytes=trace_dict['peak_alloc_bytes'],
            peak_reserved_bytes=trace_dict['peak_reserved_bytes'],
            backend=trace_dict['backend'],
            model=trace_dict['model'],
            timestamp=trace_dict['timestamp'],
            error=trace_dict.get('error'),
        )
        traces.append(trace)

    return traces


def create_summary_plot(
    traces: List[RunTrace],
    output_path: str = "summary_plot",
    model_name: str = "",
    dtype: str = "bf16"
) -> Tuple[str, str]:
    """Create a comprehensive summary plot with multiple metrics."""
    setup_plot_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    valid_traces = [t for t in traces if t.error is None]

    if not valid_traces:
        fig.suptitle('No valid traces to plot')
        return save_figure(fig, output_path)

    # Plot 1: Latency distribution
    latencies = [t.total_latency_s for t in valid_traces]
    ax1.hist(latencies, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.set_xlabel('Latency (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Latency Distribution')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Throughput distribution
    throughputs = [t.tokens_per_s for t in valid_traces]
    ax2.hist(throughputs, bins=20, alpha=0.7, color='#A23B72', edgecolor='black')
    ax2.set_xlabel('Throughput (tokens/sec)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Throughput Distribution')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Memory usage scatter
    memory_alloc = [t.peak_alloc_gb for t in valid_traces]
    memory_reserved = [t.peak_reserved_gb for t in valid_traces]
    ax3.scatter(memory_alloc, memory_reserved, alpha=0.6, color='#F18F01', s=50)
    ax3.set_xlabel('Peak Allocated Memory (GB)')
    ax3.set_ylabel('Peak Reserved Memory (GB)')
    ax3.set_title('Memory Usage Correlation')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Latency vs Memory
    ax4.scatter(memory_alloc, latencies, alpha=0.6, color='#C73E1D', s=50)
    ax4.set_xlabel('Peak Allocated Memory (GB)')
    ax4.set_ylabel('Latency (seconds)')
    ax4.set_title('Latency vs Memory Usage')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Profiling Summary', fontsize=18, fontweight='bold')
    plt.tight_layout()

    title_suffix = f"{model_name}, {dtype}" if model_name else dtype
    return save_figure(fig, output_path, title_suffix)
