from __future__ import annotations

from typing import Callable

import jax

from scm.interfaces import ModelFn
from scm.time_stepping.utils import clip_state


def get_euler_step_fn(model: ModelFn) -> Callable:
    """Euler integrator factory."""

    def _euler(t_s, dt_s, y0):
        """Euler integration. y0 is model state, dydt0 are ODE tendencies"""
        dydt0, diag0, mo_res0 = model(t_s, y0)
        y1 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y0, dydt0)
        y1 = clip_state(y1)
        return y1, dydt0, diag0, mo_res0

    return _euler


def get_ab2_step_fn(model: ModelFn) -> Callable:
    """Two-step Adams-Bashforth integrator factory."""

    def _ab2(t_s, dt_s, y1, dydt0):
        """Two-step Adams-Bashforth integration. y1 is state (i-1), dydt0 are tendencies (i-2)."""
        dydt1, diag1, mo_res1 = model(t_s, y1)
        dydt_ab = jax.tree_util.tree_map(lambda d1, d0: (3 / 2) * d1 - (1 / 2) * d0, dydt1, dydt0)
        y2 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y1, dydt_ab)
        y2 = clip_state(y2)
        return y2, dydt1, diag1, mo_res1

    return _ab2
