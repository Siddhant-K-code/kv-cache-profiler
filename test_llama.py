e#!/usr/bin/env python3
"""
Test script for Llama 3.1 model support in KV Cache Profiler.
"""

import sys
from pathlib import Path

# Add profiler to path
sys.path.insert(0, str(Path(__file__).parent))

from profiler.env import check_oom_risk


def test_llama_models():
    """Test OOM risk assessment for various Llama 3.1 models."""

    models_to_test = [
        ("meta-llama/Llama-3.1-8B-Instruct", 2, 2048),
        ("meta-llama/Llama-3.1-70B-Instruct", 1, 2048),
        ("meta-llama/Llama-3.1-405B-Instruct", 1, 1024),
    ]

    print("🦙 Testing Llama 3.1 Model Support")
    print("=" * 50)

    for model, concurrency, seq_len in models_to_test:
        print(f"\n📊 Model: {model}")
        print(f"   Concurrency: {concurrency}, Sequence Length: {seq_len}")

        try:
            risk = check_oom_risk(model, concurrency, seq_len)

            print(f"   Risk level: {risk['risk_level']}")
            print(f"   Estimated memory: {risk['total_estimated_gb']:.1f} GB")
            print(f"   Model size: {risk['model_size_gb']} GB")
            print(f"   KV cache: {risk['kv_cache_gb']:.3f} GB")

            if risk['recommendations']:
                print("   Recommendations:")
                for rec in risk['recommendations']:
                    print(f"     - {rec}")
            else:
                print("   ✅ No special recommendations")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n✅ Llama 3.1 support test completed!")


if __name__ == "__main__":
    test_llama_models()
