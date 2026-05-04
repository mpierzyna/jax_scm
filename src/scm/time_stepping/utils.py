from __future__ import annotations

import dataclasses
import time

import jax
from jax import numpy as jnp

from scm import consts
from scm.mo import MOResult
from scm.mynn.interfaces import DiagVarsMYNN, ProgVarsMYNN


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepCarry:
    """Unified carry for all time steppers (Euler, AB2, CN).

    y and diag/mo are the current state and diagnostics; prev_tends and prev_mo
    hold the previous-step values used for AB2 extrapolation of explicit sources
    and surface fluxes in the CN scheme.  On warmup both prev_* fields are set
    equal to the current-step values so AB2 degenerates to first-order Euler.
    """

    y: ProgVarsMYNN
    prev_tends: ProgVarsMYNN  # explicit tendencies at t-1 (AB2 history)
    prev_mo: MOResult  # MO result at t-1 (AB2 history for CN surface fluxes)
    diag: DiagVarsMYNN  # diagnostics at t (for K in CN and output collection)
    mo: MOResult  # MO result at t (output collection)


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
    """Clip state variables to physical floors after each time step.

    This is a numerical floor, not a physical correction — it does not conserve
    moisture or TKE budgets.  It is intentionally non-differentiable: the zero
    gradient below the floor is the correct inductive bias for AD-based parameter
    optimization (parameters that drive the state negative should be penalized, not
    rewarded).  Differentiability inside the closure is maintained by point-of-use
    smooth_eps guards, not by softening these clips.
    """
    if hasattr(y, "qke"):
        y = dataclasses.replace(y, qke=jnp.clip(y.qke, min=consts.qke_min))
    if hasattr(y, "qv"):
        y = dataclasses.replace(y, qv=jnp.clip(y.qv, min=0))
    return y
