from __future__ import annotations

from typing import Callable

import jax

from scm.interfaces import ModelFn
from scm.time_stepping.utils import StepCarry, clip_state


def get_euler_step_fn(model: ModelFn) -> Callable:
    """Euler warmup step factory.  Returns a raw-state → StepCarry function."""

    def _euler(t_s, dt_s, y0, params) -> StepCarry:
        dydt0, diag0, mo0 = model(t_s, y0, params)
        y1 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y0, dydt0)
        y1 = clip_state(y1)
        return StepCarry(y=y1, prev_tends=dydt0, prev_mo=mo0, diag=diag0, mo=mo0)

    return _euler


def get_ab2_step_fn(model: ModelFn) -> Callable:
    """Two-step Adams-Bashforth step factory.  Returns a StepCarry → StepCarry function."""

    def _ab2(carry: StepCarry, t_s, dt_s, params) -> StepCarry:
        dydt1, diag1, mo1 = model(t_s, carry.y, params)
        dydt_ab = jax.tree_util.tree_map(lambda d1, d0: (3 / 2) * d1 - (1 / 2) * d0, dydt1, carry.prev_tends)
        y2 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, carry.y, dydt_ab)
        y2 = clip_state(y2)
        return StepCarry(y=y2, prev_tends=dydt1, prev_mo=mo1, diag=diag1, mo=mo1)

    return _ab2
