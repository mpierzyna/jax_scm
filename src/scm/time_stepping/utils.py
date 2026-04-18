from __future__ import annotations

import dataclasses
import time

from jax import numpy as jnp

from scm import consts
from scm.interfaces import ProgVarsMYNN


class IterationTimer:
    """JAX callback to print timing information during iterations."""

    def __init__(self, n_total: int):
        self.last_time = None
        self.start_time = None
        self.n_total = n_total
        self.i = 0

    @staticmethod
    def _format_long_duration(d: float) -> str:
        unit = "s"
        if d > 120:
            d /= 60
            unit = "min"
        if d > 120:
            d /= 60
            unit = "h"
        return f"{d:.1f}{unit}"

    def callback(self, t: int):
        current_time = time.time()
        perc_done = self.i / (self.n_total - 1) * 100

        if self.last_time is None:
            self.start_time = current_time
            print(f"t={t} ({perc_done:.0f}%)")
        else:
            duration = current_time - self.last_time
            eta = duration * (self.n_total - self.i)
            eta_f = self._format_long_duration(eta)
            print(f"t={t} ({perc_done:.0f}%), this iter: {duration:.2f}s, ETA: {eta_f}")

        self.last_time = current_time
        self.i += 1

    def finalize(self):
        current_time = time.time()
        print(f"Total elapsed time: {self._format_long_duration(current_time - self.start_time)}")


def clip_state(y: ProgVarsMYNN) -> ProgVarsMYNN:
    """Clip state variables to physically meaningful ranges."""
    if hasattr(y, "qke"):
        y = dataclasses.replace(y, qke=jnp.clip(y.qke, min=consts.qke_min))
    if hasattr(y, "qv"):
        y = dataclasses.replace(y, qv=jnp.clip(y.qv, min=0))  # no negative humidity
    return y
