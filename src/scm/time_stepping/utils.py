from __future__ import annotations

import dataclasses

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
