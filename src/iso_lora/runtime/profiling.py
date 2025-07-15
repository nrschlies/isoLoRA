"""
CUDA GPU‑time profiler — returns *GPU‑hours* and peak memory.

Usage
-----
from iso_lora.runtime.profiling import gpu_profiler
with gpu_profiler(logger, tag="cifar10_iso"):
    train()
"""
from __future__ import annotations
import time, contextlib, logging

import torch


@contextlib.contextmanager
def gpu_profiler(
    logger: logging.Logger | None = None,
    tag: str = "run",
    enabled: bool | None = None,
):
    if enabled is False or not torch.cuda.is_available():
        # CPU or MPS — just measure wall + skip CUDA events
        t0 = time.time()
        yield
        dt = time.time() - t0
        if logger:
            logger.info(f"[gpu‑profiler] {tag} wall={dt:.2f}s (no CUDA)")
        return

    # --- CUDA timing ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    yield                          # <‑‑ code block runs here
    end_evt.record()
    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt)
    gpu_hours = ms / 1000 / 3600
    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
    if logger:
        logger.info(
            f"[gpu‑profiler] {tag} gpu_time={gpu_hours:.4f} h "
            f"peakMem={peak_mem:.1f} MB"
        )
