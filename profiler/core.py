"""
Core data structures and async concurrency harness for KV cache profiling.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Callable, Any, Optional, Dict
import torch
from pydantic import BaseModel


@dataclass
class RequestSpec:
    """Specification for a profiling request."""
    prompt_tokens: int
    generate_tokens: int
    temperature: float = 0.0

    def __post_init__(self):
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if self.generate_tokens <= 0:
            raise ValueError("generate_tokens must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")


@dataclass
class RunTrace:
    """Trace data from a single profiling run."""
    request_id: int
    seq_prompt: int
    seq_generate: int
    total_latency_s: float
    tokens_per_s: float
    peak_alloc_bytes: int
    peak_reserved_bytes: int
    backend: str
    model: str
    timestamp: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "seq_prompt": self.seq_prompt,
            "seq_generate": self.seq_generate,
            "total_latency_s": self.total_latency_s,
            "tokens_per_s": self.tokens_per_s,
            "peak_alloc_bytes": self.peak_alloc_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "backend": self.backend,
            "model": self.model,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @property
    def peak_alloc_gb(self) -> float:
        """Peak allocated memory in GB."""
        return self.peak_alloc_bytes / (1024**3)

    @property
    def peak_reserved_gb(self) -> float:
        """Peak reserved memory in GB."""
        return self.peak_reserved_bytes / (1024**3)


class MemoryTracker:
    """Context manager for tracking GPU memory usage."""

    def __init__(self, device: int = 0):
        self.device = device
        self.start_alloc = 0
        self.start_reserved = 0
        self.peak_alloc = 0
        self.peak_reserved = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_alloc = torch.cuda.memory_allocated(self.device)
            self.start_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_alloc = torch.cuda.max_memory_allocated(self.device)
            self.peak_reserved = torch.cuda.max_memory_reserved(self.device)

    def get_peak_memory(self) -> tuple[int, int]:
        """Get peak allocated and reserved memory in bytes."""
        return self.peak_alloc, self.peak_reserved


async def run_concurrent(
    concurrency: int,
    task_func: Callable[[int], Any],
    delay_between_starts: float = 0.0
) -> List[RunTrace]:
    """
    Run concurrent tasks and collect traces.

    Args:
        concurrency: Number of concurrent tasks
        task_func: Async function that takes request_id and returns RunTrace
        delay_between_starts: Delay between starting each task (seconds)

    Returns:
        List of RunTrace objects
    """
    tasks = []

    for i in range(concurrency):
        if delay_between_starts > 0 and i > 0:
            await asyncio.sleep(delay_between_starts)
        task = asyncio.create_task(task_func(i))
        tasks.append(task)

    # Wait for all tasks to complete
    traces = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions and convert to RunTrace objects
    results = []
    for i, trace in enumerate(traces):
        if isinstance(trace, Exception):
            # Create error trace
            error_trace = RunTrace(
                request_id=i,
                seq_prompt=0,
                seq_generate=0,
                total_latency_s=0.0,
                tokens_per_s=0.0,
                peak_alloc_bytes=0,
                peak_reserved_bytes=0,
                backend="unknown",
                model="unknown",
                timestamp=time.time(),
                error=str(trace)
            )
            results.append(error_trace)
        else:
            results.append(trace)

    return results


def create_synthetic_prompt(num_tokens: int, tokenizer=None) -> str:
    """
    Create a synthetic prompt with approximately the specified number of tokens.

    Args:
        num_tokens: Target number of tokens
        tokenizer: Optional tokenizer for accurate token counting

    Returns:
        Synthetic prompt string
    """
    if tokenizer is not None:
        # Use tokenizer for accurate token counting
        base_text = "The quick brown fox jumps over the lazy dog. "
        current_text = ""
        current_tokens = 0

        while current_tokens < num_tokens:
            current_text += base_text
            current_tokens = len(tokenizer.encode(current_text))

        # Trim to exact token count if needed
        if current_tokens > num_tokens:
            tokens = tokenizer.encode(current_text)[:num_tokens]
            current_text = tokenizer.decode(tokens)

        return current_text
    else:
        # Simple approximation: ~4 characters per token
        chars_per_token = 4
        target_chars = num_tokens * chars_per_token
        base_text = "The quick brown fox jumps over the lazy dog. "
        repetitions = max(1, target_chars // len(base_text))
        return base_text * repetitions


def calculate_metrics(traces: List[RunTrace]) -> Dict[str, Any]:
    """Calculate aggregate metrics from traces."""
    if not traces:
        return {}

    valid_traces = [t for t in traces if t.error is None]
    error_traces = [t for t in traces if t.error is not None]

    if not valid_traces:
        return {
            "total_requests": len(traces),
            "successful_requests": 0,
            "failed_requests": len(error_traces),
            "error_rate": 1.0,
        }

    latencies = [t.total_latency_s for t in valid_traces]
    throughputs = [t.tokens_per_s for t in valid_traces]
    peak_allocs = [t.peak_alloc_bytes for t in valid_traces]
    peak_reserved = [t.peak_reserved_bytes for t in valid_traces]

    metrics = {
        "total_requests": len(traces),
        "successful_requests": len(valid_traces),
        "failed_requests": len(error_traces),
        "error_rate": len(error_traces) / len(traces),

        # Latency metrics
        "mean_latency_s": sum(latencies) / len(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "p50_latency_s": sorted(latencies)[len(latencies) // 2],
        "p95_latency_s": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_latency_s": sorted(latencies)[int(len(latencies) * 0.99)],

        # Throughput metrics
        "mean_tokens_per_s": sum(throughputs) / len(throughputs),
        "total_tokens_per_s": sum(throughputs),
        "min_tokens_per_s": min(throughputs),
        "max_tokens_per_s": max(throughputs),

        # Memory metrics
        "peak_alloc_bytes": max(peak_allocs),
        "peak_reserved_bytes": max(peak_reserved),
        "peak_alloc_gb": max(peak_allocs) / (1024**3),
        "peak_reserved_gb": max(peak_reserved) / (1024**3),
        "mean_alloc_bytes": sum(peak_allocs) / len(peak_allocs),
        "mean_reserved_bytes": sum(peak_reserved) / len(peak_reserved),
    }

    return metrics


def save_traces_json(traces: List[RunTrace], filepath: str):
    """Save traces to JSON file."""
    import json

    data = {
        "traces": [trace.to_dict() for trace in traces],
        "metrics": calculate_metrics(traces),
        "timestamp": time.time(),
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_traces_csv(traces: List[RunTrace], filepath: str):
    """Save traces to CSV file."""
    import csv

    if not traces:
        return

    fieldnames = [
        "request_id", "seq_prompt", "seq_generate", "total_latency_s",
        "tokens_per_s", "peak_alloc_bytes", "peak_reserved_bytes",
        "peak_alloc_gb", "peak_reserved_gb", "backend", "model",
        "timestamp", "error"
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trace in traces:
            row = trace.to_dict()
            row["peak_alloc_gb"] = trace.peak_alloc_gb
            row["peak_reserved_gb"] = trace.peak_reserved_gb
            writer.writerow(row)


class ProfilingSession:
    """Context manager for a profiling session with automatic cleanup."""

    def __init__(self, session_name: str, output_dir: str = "data"):
        self.session_name = session_name
        self.output_dir = output_dir
        self.start_time = None
        self.traces: List[RunTrace] = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.traces:
            import os
            os.makedirs(self.output_dir, exist_ok=True)

            timestamp = int(self.start_time)
            base_filename = f"{self.session_name}_{timestamp}"

            json_path = os.path.join(self.output_dir, f"{base_filename}.json")
            csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")

            save_traces_json(self.traces, json_path)
            save_traces_csv(self.traces, csv_path)

            print(f"Saved {len(self.traces)} traces to:")
            print(f"  JSON: {json_path}")
            print(f"  CSV: {csv_path}")

    def add_traces(self, traces: List[RunTrace]):
        """Add traces to the session."""
        self.traces.extend(traces)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current session metrics."""
        return calculate_metrics(self.traces)
