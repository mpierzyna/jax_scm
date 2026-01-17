from __future__ import annotations

import dataclasses
import time
from typing import Tuple, Callable

import jax
from jax import numpy as jnp

from scm.interfaces import Simulation, ProgVarsT, DiagVarsT, ModelFn
from scm.mo import MOResult

Q_SQ_MIN = 1e-10  # clipping to avoid negative TKE


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


def get_euler_step_fn(model: ModelFn) -> Callable:
    """Euler integrator factory."""

    @jax.jit
    def _euler(dt_s: float, y0, **kwargs):
        """Euler integration. y0 is model state, dydt0 are ODE tendencies"""
        dydt0, diag0, mo_res0 = model(y0, **kwargs)
        y1 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y0, dydt0)
        if hasattr(y1, "q_sq"):
            y1 = dataclasses.replace(y1, q_sq=jnp.clip(y1.q_sq, min=Q_SQ_MIN))
        return y1, dydt0, diag0, mo_res0

    return _euler


def get_ab2_step_fn(model: ModelFn) -> Callable:
    """Two-step Adams-Bashforth integrator factory."""

    @jax.jit
    def _ab2(dt_s: float, y1, dydt0, **kwargs):
        """Two-step Adams-Bashforth integration. y1 is state (i-1), dydt0 are tendencies (i-2)."""
        dydt1, diag1, mo_res1 = model(y1, **kwargs)
        dydt_ab = jax.tree_util.tree_map(lambda d1, d0: (3 / 2) * d1 - (1 / 2) * d0, dydt1, dydt0)
        y2 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y1, dydt_ab)
        if hasattr(y2, "q_sq"):
            y2 = dataclasses.replace(y2, q_sq=jnp.clip(y2.q_sq, min=Q_SQ_MIN))
        return y2, dydt1, diag1, mo_res1

    return _ab2


def simulate(
    model: ModelFn,
    sim: Simulation,
    dt_s: float,
    dt_s_out: float,
) -> Tuple[ProgVarsT, DiagVarsT, MOResult, jnp.ndarray]:
    """Simulate model with AB2 and constant timestep."""
    # Setup time integration
    _euler = get_euler_step_fn(model)
    _ab2 = get_ab2_step_fn(model)

    # Create forcing evaluation function
    get_forcing = sim.forcing.get_eval_fn()

    # Setup timestep arrays
    # Inner steps shifted by dt because initial Euler step is taken outside loop
    t_outer = jnp.arange(sim.t_start_s, sim.t_end_s, dt_s_out)
    rel_t_inner = jnp.arange(0, dt_s_out, dt_s) + dt_s  # relative to outer step
    print(
        f"Inner steps: {len(rel_t_inner)}, "
        f"Outer steps: {len(t_outer)}, "
        f"Total steps: {len(t_outer) * len(rel_t_inner)}"
    )
    timer = IterationTimer(n_total=len(t_outer))

    @jax.jit
    def _scan_inner(carry, t):
        """Advance model by one step but don't accumulate outputs"""
        y1, dydt0, _, _ = carry
        y2, dydt1, diag1, mo_res1 = _ab2(dt_s, y1, dydt0, forcing=get_forcing(t))
        return (y2, dydt1, diag1, mo_res1), None

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate outputs"""
        carry_new, _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
        jax.debug.callback(timer.callback, t + dt_s_out)
        return carry_new, carry_new

    jax.debug.print("Begin simulation...")
    y0 = sim.init
    y1, dydt0, diag0, mo_res0 = _euler(dt_s, y0, forcing=get_forcing(t_outer[0]))  # Warmup: one Euler step
    _, (y_hist, _, diag_hist, mo_hist) = jax.lax.scan(_scan_outer, init=(y1, dydt0, diag0, mo_res0), xs=t_outer)
    timer.finalize()

    return y_hist, diag_hist, mo_hist, t_outer


def simulate_adaptive_dt(
    model: ModelFn,
    sim: Simulation,
    cfl_max: float,
    dt_s_init: float,
    dt_s_max: float,
    dt_s_out: float,
) -> Tuple[ProgVarsT, DiagVarsT, MOResult, jnp.ndarray]:
    """Simulate model with AB2 and adaptive time stepping based on CFL condition for diffusion."""
    # Setup time integration
    _euler = get_euler_step_fn(model)
    _ab2 = get_ab2_step_fn(model)

    # Create forcing evaluation function
    get_forcing = sim.forcing.get_eval_fn()

    # Setup outer loop and timer
    t_outer = jnp.arange(sim.t_start_s, sim.t_end_s, dt_s_out)
    timer = IterationTimer(n_total=len(t_outer))

    @jax.jit
    def _get_dt(Km: jnp.ndarray, Kh: jnp.ndarray) -> jnp.ndarray:
        """Compute adaptive timestep based on CFL condition for diffusion."""
        Km_max = jnp.max(Km)
        Kh_max = jnp.max(Kh)
        K_max = jnp.clip(jnp.maximum(Km_max, Kh_max), min=1e-6)  # avoid zero division
        dt = cfl_max * sim.grid.dz**2 / K_max

        return jnp.minimum(dt, dt_s_max)

    @jax.jit
    def _while_body(carry):
        """Advance model by one adaptive step"""
        # Unpack previous state
        y1, dydt0, diag0, _, i, t, t_left = carry

        # Compute adaptive timestep. Make sure we always finish for dt_out_s.
        dt_s = jnp.minimum(_get_dt(Km=diag0.Km, Kh=diag0.Kh), t_left)

        # Integrate one step
        y2, dydt1, diag1, mo_res1 = _ab2(dt_s, y1, dydt0, forcing=get_forcing(t))

        # Advance time
        t_left = t_left - dt_s
        t = t + dt_s
        i = i + 1

        return y2, dydt1, diag1, mo_res1, i, t, t_left

    @jax.jit
    def _while_cond(carry):
        """Condition for adaptive stepping loop"""
        *_, t_left = carry
        return t_left > 0

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate outputs"""
        carry = (*carry, 0, t.astype(float), dt_s_out)  # Expand carry for while loop
        carry_new = jax.lax.while_loop(_while_cond, _while_body, carry)
        *carry_new, i, _, _ = carry_new  # unpack carry
        jax.debug.callback(timer.callback, t + dt_s_out)
        jax.debug.print("Took {i} steps with average dt={dt_s:.4f}s", i=i, dt_s=dt_s_out / i)
        return tuple(carry_new), tuple(carry_new)

    jax.debug.print("Begin simulation...")
    y0 = sim.init
    y1, dydt0, diag0, mo_res0 = _euler(dt_s_init, y0, forcing=get_forcing(t_outer[0]))  # Warmup: one Euler step
    _, (y_hist, _, diag_hist, mo_hist) = jax.lax.scan(_scan_outer, init=(y1, dydt0, diag0, mo_res0), xs=t_outer)
    timer.finalize()

    return y_hist, diag_hist, mo_hist, t_outer
