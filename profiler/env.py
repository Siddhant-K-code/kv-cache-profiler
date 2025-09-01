"""
Environment introspection module for GPU/CUDA/library detection and reporting.
"""

import platform
import subprocess
import sys
from typing import Dict, Optional, Any
import torch


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA information including version and device details."""
    cuda_info = {
        "available": torch.cuda.is_available(),
        "version": None,
        "device_count": 0,
        "devices": [],
        "current_device": None,
    }

    if torch.cuda.is_available():
        cuda_info["version"] = torch.version.cuda
        cuda_info["device_count"] = torch.cuda.device_count()
        cuda_info["current_device"] = torch.cuda.current_device()

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "major": device_props.major,
                "minor": device_props.minor,
                "multi_processor_count": device_props.multi_processor_count,
            }
            cuda_info["devices"].append(device_info)

    return cuda_info


def get_library_versions() -> Dict[str, Optional[str]]:
    """Get versions of key libraries."""
    versions = {}

    # Core libraries
    versions["torch"] = torch.__version__

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = None

    try:
        import accelerate
        versions["accelerate"] = accelerate.__version__
    except ImportError:
        versions["accelerate"] = None

    try:
        import tokenizers
        versions["tokenizers"] = tokenizers.__version__
    except ImportError:
        versions["tokenizers"] = None

    # Optional libraries
    try:
        import vllm
        versions["vllm"] = vllm.__version__
    except ImportError:
        versions["vllm"] = None

    try:
        import ollama
        # Ollama doesn't have __version__, try to get it from package metadata
        try:
            versions["ollama"] = ollama.__version__
        except AttributeError:
            # Fallback to package metadata
            try:
                import importlib.metadata
                versions["ollama"] = importlib.metadata.version("ollama")
            except Exception:
                versions["ollama"] = "installed"
    except ImportError:
        versions["ollama"] = None

    return versions


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    memory_info = {
        "cuda_memory": {},
        "system_memory": None,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_memory = {
                "allocated": torch.cuda.memory_allocated(i),
                "reserved": torch.cuda.memory_reserved(i),
                "max_allocated": torch.cuda.max_memory_allocated(i),
                "max_reserved": torch.cuda.max_memory_reserved(i),
            }
            memory_info["cuda_memory"][f"device_{i}"] = device_memory

    try:
        import psutil
        memory_info["system_memory"] = {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        }
    except ImportError:
        pass

    return memory_info


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information."""
    return {
        "system": get_system_info(),
        "cuda": get_cuda_info(),
        "libraries": get_library_versions(),
        "memory": get_memory_info(),
    }


def print_environment_banner():
    """Print a formatted environment banner."""
    env_info = get_environment_info()

    print("=" * 60)
    print("KV Cache Profiler - Environment Information")
    print("=" * 60)

    # System info
    print(f"Platform: {env_info['system']['platform']}")
    print(f"Python: {env_info['system']['python_version'].split()[0]}")

    # CUDA info
    cuda = env_info['cuda']
    if cuda['available']:
        print(f"CUDA: {cuda['version']} ({cuda['device_count']} device(s))")
        for device in cuda['devices']:
            memory_gb = device['total_memory'] / (1024**3)
            print(f"  - {device['name']}: {memory_gb:.1f} GB")
    else:
        print("CUDA: Not available (CPU mode)")

    # Library versions
    libs = env_info['libraries']
    print(f"PyTorch: {libs['torch']}")
    print(f"Transformers: {libs['transformers'] or 'Not installed'}")
    if libs['vllm']:
        print(f"vLLM: {libs['vllm']}")
    if libs['ollama']:
        print(f"Ollama: {libs['ollama']}")

    print("=" * 60)


def check_oom_risk(model_name: str, concurrency: int, seq_len: int) -> Dict[str, Any]:
    """Estimate OOM risk based on model size and parameters."""
    # Rough estimates for common models (in GB)
    model_sizes = {
        "facebook/opt-1.3b": 2.6,
        "facebook/opt-2.7b": 5.4,
        "facebook/opt-6.7b": 13.4,
        "microsoft/DialoGPT-medium": 0.7,
        "microsoft/DialoGPT-large": 1.4,
        "gpt2": 0.5,
        "gpt2-medium": 1.4,
        "gpt2-large": 3.0,
        "gpt2-xl": 6.0,
        # Meta Llama 3.1 models
        "meta-llama/Llama-3.1-8B": 16.0,
        "meta-llama/Llama-3.1-8B-Instruct": 16.0,
        "meta-llama/Llama-3.1-70B": 140.0,
        "meta-llama/Llama-3.1-70B-Instruct": 140.0,
        "meta-llama/Llama-3.1-405B": 810.0,
        "meta-llama/Llama-3.1-405B-Instruct": 810.0,
        # Meta Llama 3 models (legacy)
        "meta-llama/Meta-Llama-3-8B": 16.0,
        "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,
        "meta-llama/Meta-Llama-3-70B": 140.0,
        "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
    }

    base_model_size = model_sizes.get(model_name, 5.0)  # Default estimate

    # Estimate KV cache size (rough approximation)
    # KV cache size ≈ 2 * num_layers * hidden_size * seq_len * batch_size * 2 bytes (fp16)
    # For simplicity, use a heuristic based on model size
    kv_cache_per_token_gb = base_model_size * 0.001  # Very rough estimate
    total_kv_cache_gb = kv_cache_per_token_gb * seq_len * concurrency

    total_estimated_gb = base_model_size + total_kv_cache_gb

    risk_assessment = {
        "model_size_gb": base_model_size,
        "kv_cache_gb": total_kv_cache_gb,
        "total_estimated_gb": total_estimated_gb,
        "risk_level": "low",
        "recommendations": [],
    }

    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        risk_assessment["available_gpu_memory_gb"] = available_memory

        if total_estimated_gb > available_memory * 0.9:
            risk_assessment["risk_level"] = "high"
            risk_assessment["recommendations"].extend([
                "Reduce concurrency",
                "Reduce sequence length",
                "Consider using a smaller model",
                "Enable gradient checkpointing if available"
            ])
        elif total_estimated_gb > available_memory * 0.7:
            risk_assessment["risk_level"] = "medium"
            risk_assessment["recommendations"].extend([
                "Monitor memory usage closely",
                "Consider reducing batch size if OOM occurs"
            ])
    else:
        risk_assessment["recommendations"].append("Running on CPU - expect slower performance")

    return risk_assessment
