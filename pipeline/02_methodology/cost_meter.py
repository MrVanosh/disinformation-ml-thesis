"""Context manager mierzący koszt obliczeniowy: czas, peak memory, trainable params.

Użycie:
    with CostMeter() as cm:
        # ... trening / inferencja ...
        cm.set_trainable_params(model)

    print(cm.report())
    # {"train_s": 1340.5, "infer_ms_per_sample": 38.4, "peak_ram_mb": 22840,
    #  "peak_vram_mb": ..., "trainable_params": 28311552}
"""

from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from typing import Any


class CostMeter:
    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.peak_ram_mb: float = 0.0
        self.peak_vram_mb: float = 0.0
        self.trainable_params: int = 0
        self.n_samples_processed: int = 0
        self._psutil = None

    def __enter__(self):
        gc.collect()
        try:
            import psutil
            self._psutil = psutil
            proc = psutil.Process(os.getpid())
            self.peak_ram_mb = proc.memory_info().rss / (1024 ** 2)
        except ImportError:
            self._psutil = None

        # CUDA peak reset
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            elif torch.backends.mps.is_available():
                # MPS nie ma reset; mierzymy "after"
                pass
        except ImportError:
            pass

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self._psutil:
            proc = self._psutil.Process(os.getpid())
            self.peak_ram_mb = max(self.peak_ram_mb, proc.memory_info().rss / (1024 ** 2))

        try:
            import torch
            if torch.cuda.is_available():
                self.peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            elif torch.backends.mps.is_available():
                # current_allocated_memory dla MPS dostępne od torch ~2.0
                try:
                    self.peak_vram_mb = float(torch.mps.current_allocated_memory() / (1024 ** 2))
                except Exception:
                    pass
        except ImportError:
            pass

    def set_trainable_params(self, model) -> None:
        try:
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            self.trainable_params = 0

    def set_n_samples(self, n: int) -> None:
        self.n_samples_processed = n

    @property
    def elapsed_s(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def report(self) -> dict[str, Any]:
        ms_per_sample = (
            (self.elapsed_s * 1000.0) / self.n_samples_processed
            if self.n_samples_processed
            else None
        )
        return {
            "elapsed_s": round(self.elapsed_s, 3),
            "ms_per_sample": round(ms_per_sample, 3) if ms_per_sample is not None else None,
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "trainable_params": int(self.trainable_params),
            "n_samples_processed": int(self.n_samples_processed),
        }
