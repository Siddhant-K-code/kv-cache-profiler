"""
HuggingFace backend implementation for KV cache profiling.
"""

import asyncio
import time
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from ..core import RequestSpec, RunTrace, MemoryTracker, create_synthetic_prompt


class HFBackend:
    """HuggingFace backend for LLM inference profiling."""

    def __init__(
        self,
        model: str,
        dtype: str = "bf16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model
        self.dtype = self._parse_dtype(dtype)
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _parse_dtype(self, dtype: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}")

        return dtype_map[dtype]

    def load_model(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading model: {self.model_name}")
        print(f"  dtype: {self.dtype}")
        print(f"  device_map: {self.device_map}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            ).eval()

            self._loaded = True
            print(f"Model loaded successfully!")

            # Print model info
            if hasattr(self.model, 'config'):
                config = self.model.config
                print(f"  Model config: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                if hasattr(config, 'hidden_size'):
                    print(f"  Hidden size: {config.hidden_size}")
                if hasattr(config, 'num_hidden_layers'):
                    print(f"  Layers: {config.num_hidden_layers}")
                if hasattr(config, 'num_attention_heads'):
                    print(f"  Attention heads: {config.num_attention_heads}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    async def run(self, request_id: int, spec: RequestSpec) -> RunTrace:
        """Run a single inference request and return trace."""
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Create synthetic prompt
            prompt_text = create_synthetic_prompt(spec.prompt_tokens, self.tokenizer)

            # Tokenize input
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Move to device if CUDA is available
            if torch.cuda.is_available() and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            actual_prompt_tokens = inputs['input_ids'].shape[1]

            # Set up generation config
            generation_config = GenerationConfig(
                max_new_tokens=spec.generate_tokens,
                do_sample=spec.temperature > 0.0,
                temperature=spec.temperature if spec.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

            # Track memory usage during generation
            with MemoryTracker() as memory_tracker:
                # Run generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )

            # Calculate metrics
            end_time = time.time()
            total_latency = end_time - start_time

            # Get actual generated tokens
            generated_tokens = outputs.sequences.shape[1] - actual_prompt_tokens
            tokens_per_s = generated_tokens / total_latency if total_latency > 0 else 0.0

            # Get memory usage
            peak_alloc, peak_reserved = memory_tracker.get_peak_memory()

            return RunTrace(
                request_id=request_id,
                seq_prompt=actual_prompt_tokens,
                seq_generate=generated_tokens,
                total_latency_s=total_latency,
                tokens_per_s=tokens_per_s,
                peak_alloc_bytes=peak_alloc,
                peak_reserved_bytes=peak_reserved,
                backend="huggingface",
                model=self.model_name,
                timestamp=start_time,
            )

        except Exception as e:
            end_time = time.time()
            return RunTrace(
                request_id=request_id,
                seq_prompt=0,
                seq_generate=0,
                total_latency_s=end_time - start_time,
                tokens_per_s=0.0,
                peak_alloc_bytes=0,
                peak_reserved_bytes=0,
                backend="huggingface",
                model=self.model_name,
                timestamp=start_time,
                error=str(e),
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._loaded:
            return {"loaded": False}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "device_map": self.device_map,
        }

        if hasattr(self.model, 'config'):
            config = self.model.config
            info.update({
                "model_type": getattr(config, 'model_type', 'unknown'),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "vocab_size": getattr(config, 'vocab_size', None),
            })

        # Get device info
        if hasattr(self.model, 'device'):
            info["device"] = str(self.model.device)

        # Get parameter count
        if self.model is not None:
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                })
            except Exception:
                pass

        return info

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._loaded = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


async def create_hf_backend(
    model: str,
    dtype: str = "bf16",
    device_map: str = "auto",
    **kwargs
) -> HFBackend:
    """Async factory function to create and load HF backend."""
    backend = HFBackend(
        model=model,
        dtype=dtype,
        device_map=device_map,
        **kwargs
    )

    # Load model in executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model)

    return backend
