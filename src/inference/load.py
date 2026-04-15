"""Model loading utilities for inference backends."""

from typing import Any

import torch
from vllm import LLM

from src.utils.log import log


# default loading parameters
_DEFAULT_MAX_MODEL_LEN = 32768
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.82
_DEFAULT_MAX_NUM_SEQS = 32

# olmo-3 think models use sliding window attention, incompatible with prefix caching in vllm 0.7.x
_NO_PREFIX_CACHE_PREFIXES = ("allenai/olmo-3",)


def _patch_olmo3_config() -> None:
    """Add missing head_dim property to Olmo3Config for vLLM 0.7.x compatibility.

    vLLM's Transformers backend accesses config.head_dim, but Olmo3Config does
    not define it. It can be derived from hidden_size // num_attention_heads.
    """
    try:
        from transformers import Olmo3Config
    except ImportError:
        return  # older transformers without olmo3 support

    if not hasattr(Olmo3Config, "head_dim"):
        Olmo3Config.head_dim = property(  # type: ignore[attr-defined]
            lambda self: self.hidden_size // self.num_attention_heads,
        )


def _patch_vllm_scheduler() -> None:
    """Monkey-patch the vLLM v0 scheduler to fix an async output processing bug.

    In vLLM 0.7.x, finished sequences are placed in _async_stopped for
    deferred cleanup. A bug causes the cleanup to sometimes not run before the
    next scheduling step, triggering:
        assert len(self._async_stopped) == 0
    The fix: drain _async_stopped at the start of _schedule_running by calling
    free_finished_seq_groups() (which frees KV cache blocks and clears the list).
    """
    try:
        import vllm.core.scheduler as _sched  # only exists in vllm v0 (0.7.x)
    except ModuleNotFoundError:
        return  # vllm v1 (0.8+) does not have this bug

    _orig = _sched.Scheduler._schedule_running

    def _patched(self: _sched.Scheduler, *args: Any, **kwargs: Any) -> Any:
        # drain any leftover async-stopped sequences before the assertion fires
        if getattr(self, "_async_stopped", None):
            if hasattr(self, "free_finished_seq_groups"):
                self.free_finished_seq_groups()
            else:
                self._async_stopped.clear()
        return _orig(self, *args, **kwargs)

    _sched.Scheduler._schedule_running = _patched


def load_llm(
    model_path: str,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
) -> LLM:
    """Load a model into vLLM for fast batched inference.

    Returns the vLLM LLM client ready for generation.
    """
    # detect gpu count based on available backend
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        gpu_count = 1  # mps only supports single device
    else:
        gpu_count = 1

    _patch_olmo3_config()
    _patch_vllm_scheduler()

    # olmo-3 uses sliding window attention which is incompatible with prefix caching
    path_lower = model_path.lower()
    use_prefix_caching = not any(
        path_lower.startswith(p) for p in _NO_PREFIX_CACHE_PREFIXES
    )

    log(f"  Prefix caching: {use_prefix_caching}")
    log(f"  GPU count: {gpu_count}")

    llm_client = LLM(
        model=model_path,
        trust_remote_code=True,
        enable_prefix_caching=use_prefix_caching,
        tensor_parallel_size=gpu_count,
        max_num_seqs=max_num_seqs or _DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization=gpu_memory_utilization
        or _DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len or _DEFAULT_MAX_MODEL_LEN,
    )

    return llm_client
