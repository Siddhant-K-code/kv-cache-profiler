"""
Basic test to verify the KV Cache Profiler implementation works.
"""

import asyncio
import sys
from pathlib import Path

# Add profiler to path
sys.path.insert(0, str(Path(__file__).parent))

from profiler.core import RequestSpec, RunTrace
from profiler.env import get_environment_info, check_oom_risk
from profiler.backends.hf import HFBackend


async def test_basic_functionality():
    """Test basic functionality without requiring GPU."""

    print("🧪 Testing KV Cache Profiler...")

    # Test 1: Environment info
    print("\n1️⃣ Testing environment detection...")
    env_info = get_environment_info()
    assert "system" in env_info
    assert "cuda" in env_info
    assert "libraries" in env_info
    print("✅ Environment detection works")

    # Test 2: OOM risk assessment
    print("\n2️⃣ Testing OOM risk assessment...")
    risk = check_oom_risk("facebook/opt-1.3b", concurrency=4, seq_len=512)
    assert "risk_level" in risk
    assert "recommendations" in risk
    assert risk["risk_level"] in ["low", "medium", "high"]
    print(f"✅ OOM risk assessment works (risk: {risk['risk_level']})")

    # Test 3: RequestSpec validation
    print("\n3️⃣ Testing RequestSpec...")
    spec = RequestSpec(prompt_tokens=100, generate_tokens=50, temperature=0.0)
    assert spec.prompt_tokens == 100
    assert spec.generate_tokens == 50
    assert spec.temperature == 0.0

    # Test invalid specs
    try:
        RequestSpec(prompt_tokens=-1, generate_tokens=50)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✅ RequestSpec validation works")

    # Test 4: RunTrace creation
    print("\n4️⃣ Testing RunTrace...")
    import time
    trace = RunTrace(
        request_id=0,
        seq_prompt=100,
        seq_generate=50,
        total_latency_s=1.5,
        tokens_per_s=33.3,
        peak_alloc_bytes=1024*1024*1024,  # 1GB
        peak_reserved_bytes=2*1024*1024*1024,  # 2GB
        backend="test",
        model="test-model",
        timestamp=time.time()
    )

    assert trace.peak_alloc_gb == 1.0
    assert trace.peak_reserved_gb == 2.0

    trace_dict = trace.to_dict()
    assert "request_id" in trace_dict
    assert "total_latency_s" in trace_dict

    print("✅ RunTrace creation works")

    # Test 5: Backend creation (without loading model)
    print("\n5️⃣ Testing HFBackend creation...")
    backend = HFBackend(model="facebook/opt-1.3b", dtype="bf16")
    assert backend.model_name == "facebook/opt-1.3b"
    assert not backend._loaded

    info = backend.get_model_info()
    assert info["loaded"] == False

    print("✅ HFBackend creation works")

    print("\n🎉 All basic tests passed!")
    print("\nTo run a full test with model loading:")
    print("  uv run profiler/cli.py env")
    print("  uv run profiler/cli.py bench --model facebook/opt-1.3b --concurrency 1")


def main():
    """Run basic tests."""
    asyncio.run(test_basic_functionality())


if __name__ == "__main__":
    main()
