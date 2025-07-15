"""
Edge‑device monitor: wall‑clock duration + RAM usage.

Usage
-----
from iso_lora.runtime.edge_device import edge_monitor
with edge_monitor(logger, tag="cifar10_debug"):
    ... your training / inference code ...
"""
from __future__ import annotations
import os, time, contextlib, logging

try:
    import psutil
except ImportError:  # fallback — only wall‑clock if psutil missing
    psutil = None


@contextlib.contextmanager
def edge_monitor(logger: logging.Logger | None = None, tag: str = "run"):
    proc = psutil.Process(os.getpid()) if psutil else None
    mem0 = proc.memory_info().rss if proc else 0
    t0 = time.time()
    yield
    dt = time.time() - t0
    mem1 = proc.memory_info().rss if proc else 0
    peak = (
        max(mem0, mem1, getattr(proc, "memory_info", lambda: (0, 0))().rss)
        if proc
        else None
    )

    if logger:
        msg = f"[edge‑monitor] {tag} wall={dt:.2f}s"
        if peak is not None:
            msg += f"  peakRAM={peak/1e6:.1f} MB"
        logger.info(msg)
