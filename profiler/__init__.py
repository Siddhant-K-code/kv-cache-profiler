"""
KV Cache Profiler - A tool for profiling LLM inference bottlenecks with KV cache analysis

This package provides tools to profile and visualize KV cache memory usage and performance
bottlenecks in LLM inference workloads across different backends (HuggingFace, vLLM, Ollama).
"""

from .core import RequestSpec, RunTrace
from .env import get_environment_info

__version__ = "0.1.0"
__all__ = ["RequestSpec", "RunTrace", "get_environment_info"]
