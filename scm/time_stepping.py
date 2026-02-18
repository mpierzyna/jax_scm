from __future__ import annotations

import dataclasses
import time
from typing import Tuple, Callable

import jax
from jax import numpy as jnp

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, ProgVarsT, DiagVarsT, ModelFn
from scm.mo import MOResult


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


def clip_state(y: ProgVarsT) -> ProgVarsT:
    """Clip state variables to physically meaningful ranges."""
    if hasattr(y, "qke"):
        y = dataclasses.replace(y, qke=jnp.clip(y.qke, min=consts.qke_min))
    if hasattr(y, "qv"):
        y = dataclasses.replace(y, qv=jnp.clip(y.qv, min=0))  # no negative humidity
    return y


def get_euler_step_fn(model: ModelFn) -> Callable:
    """Euler integrator factory."""

    @jax.jit
    def _euler(t_s, dt_s, y0):
        """Euler integration. y0 is model state, dydt0 are ODE tendencies"""
        dydt0, diag0, mo_res0 = model(t_s, y0)
        y1 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y0, dydt0)
        y1 = clip_state(y1)
        return y1, dydt0, diag0, mo_res0

    return _euler


def get_ab2_step_fn(model: ModelFn) -> Callable:
    """Two-step Adams-Bashforth integrator factory."""

    @jax.jit
    def _ab2(t_s, dt_s, y1, dydt0):
        """Two-step Adams-Bashforth integration. y1 is state (i-1), dydt0 are tendencies (i-2)."""
        dydt1, diag1, mo_res1 = model(t_s, y1)
        dydt_ab = jax.tree_util.tree_map(lambda d1, d0: (3 / 2) * d1 - (1 / 2) * d0, dydt1, dydt0)
        y2 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y1, dydt_ab)
        y2 = clip_state(y2)
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
        y2, dydt1, diag1, mo_res1 = _ab2(t, dt_s, y1, dydt0)
        return (y2, dydt1, diag1, mo_res1), None

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate outputs"""
        carry_new, _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
        jax.debug.callback(timer.callback, t + dt_s_out)
        return carry_new, carry_new

    jax.debug.print("Begin simulation...")
    y0 = sim.init
    y1, dydt0, diag0, mo_res0 = _euler(t_outer[0], dt_s, y0)  # Warmup: one Euler step
    _, (y_hist, _, diag_hist, mo_hist) = jax.lax.scan(_scan_outer, init=(y1, dydt0, diag0, mo_res0), xs=t_outer)
    timer.finalize()

    return y_hist, diag_hist, mo_hist, t_outer


def get_cn_step_fn(
    model: ModelFn,
    grid: StaggeredGrid,
) -> Tuple[Callable, Callable]:
    """Factory for semi-implicit Crank-Nicolson stepper (MYNN-specific).

    Diffusion terms are solved implicitly with K fixed at the current time step
    (K diagnosed from the closure and held constant within each CN step).
    Non-diffusive explicit sources (Coriolis, QKE production/dissipation,
    large-scale advection) use AB2 extrapolation after the warmup step.

    The ``model`` **must** be initialized with ``implicit=True`` so that it
    returns only non-diffusive tendencies as the forcing vector S.

    Parameters
    ----------
    model : ModelFn
        MYNN model function (initialized with ``implicit=True``).
    grid : StaggeredGrid
        Vertical grid (provides ``dz``).

    Returns
    -------
    _cn_warmup : Callable
        ``(t_s, dt_s, y0) -> (y1, S0, diag0, mo_res0)``
        Warmup step: CN with S^{prev} = S^0 (first-order accurate in time).
    _cn : Callable
        ``(t_s, dt_s, y1, S_prev) -> (y2, S1, diag1, mo_res1)``
        Regular CN step with AB2-extrapolated explicit sources.
    """
    from scm.implicit import cn_solve
    from scm.mynn.interfaces import ProgVarsMYNN

    def _apply_cn(
        y: ProgVarsMYNN,
        S: ProgVarsMYNN,
        diag,
        mo_res,
        dt_s: float,
    ) -> ProgVarsMYNN:
        """Apply one CN diffusion solve for every prognostic variable.

        K values come from the diagnosed diffusivities (fixed at current step).
        Surface BCs come from MO (treated explicitly as a source term).
        """
        dz = grid.dz
        u_new = cn_solve(y.u, diag.Km, dt_s, dz, S.u, mo_res.u_w)
        v_new = cn_solve(y.v, diag.Km, dt_s, dz, S.v, mo_res.v_w)
        th_new = cn_solve(y.th, diag.Kh, dt_s, dz, S.th, mo_res.w_th)
        qv_new = cn_solve(y.qv, diag.Kh, dt_s, dz, S.qv, mo_res.w_qv)
        # QKE surface flux is always zero (surface BC set in closure)
        qke_new = cn_solve(y.qke, diag.Kq, dt_s, dz, S.qke, 0.0)
        return ProgVarsMYNN(u=u_new, v=v_new, th=th_new, qv=qv_new, qke=qke_new)

    @jax.jit
    def _cn_warmup(t_s, dt_s, y0):
        """Warmup step: CN with S_prev = S^0 (no previous tendency stored yet).

        AB2 extrapolation degenerates to pure S^0, making this equivalent
        to first-order-accurate forward Euler for the explicit sources.
        """
        S0, diag0, mo_res0 = model(t_s, y0)
        y1 = _apply_cn(y0, S0, diag0, mo_res0, dt_s)
        y1 = clip_state(y1)
        return y1, S0, diag0, mo_res0

    @jax.jit
    def _cn(t_s, dt_s, y1, S_prev):
        """CN step with AB2-extrapolated non-diffusive explicit sources.

        Explicit sources (Coriolis, QKE budget, advection) are extrapolated
        as S_ab2 = (3/2)*S^n - (1/2)*S^{n-1} before solving the CN system.
        Diffusivities K are taken from the current-step closure diagnostics.
        """
        S1, diag1, mo_res1 = model(t_s, y1)
        S_ab2 = jax.tree_util.tree_map(lambda s1, s0: (3 / 2) * s1 - (1 / 2) * s0, S1, S_prev)
        y2 = _apply_cn(y1, S_ab2, diag1, mo_res1, dt_s)
        y2 = clip_state(y2)
        return y2, S1, diag1, mo_res1

    return _cn_warmup, _cn


def simulate_cn(
    model: ModelFn,
    sim: Simulation,
    dt_s: float,
    dt_s_out: float,
) -> Tuple[ProgVarsT, DiagVarsT, MOResult, jnp.ndarray]:
    """Simulate model with semi-implicit Crank-Nicolson diffusion and AB2 explicit sources.

    Diffusion terms are solved implicitly (K fixed per step); non-diffusive
    terms (Coriolis, QKE production/dissipation, large-scale advection) use
    AB2 extrapolation. The ``model`` must be initialized with ``implicit=True``.

    Parameters
    ----------
    model : ModelFn
        MYNN model function (initialized with ``implicit=True``).
    sim : Simulation
        Simulation container (provides initial state and time bounds).
    dt_s : float
        Inner time step [s].
    dt_s_out : float
        Output interval [s]; must be a multiple of ``dt_s``.

    Returns
    -------
    y_hist, diag_hist, mo_hist, t_outer
        History arrays sampled at ``dt_s_out`` intervals.
    """
    _cn_warmup, _cn = get_cn_step_fn(model, sim.grid)

    # Setup timestep arrays (same layout as simulate())
    t_outer = jnp.arange(sim.t_start_s, sim.t_end_s, dt_s_out)
    rel_t_inner = jnp.arange(0, dt_s_out, dt_s) + dt_s  # relative to outer step, shifted by dt
    print(
        f"Inner steps: {len(rel_t_inner)}, "
        f"Outer steps: {len(t_outer)}, "
        f"Total steps: {len(t_outer) * len(rel_t_inner)}"
    )
    timer = IterationTimer(n_total=len(t_outer))

    @jax.jit
    def _scan_inner(carry, t):
        """Advance model by one CN step without accumulating output."""
        y1, S_prev, _, _ = carry
        y2, S1, diag1, mo_res1 = _cn(t, dt_s, y1, S_prev)
        return (y2, S1, diag1, mo_res1), None

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate output."""
        carry_new, _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
        jax.debug.callback(timer.callback, t + dt_s_out)
        return carry_new, carry_new

    jax.debug.print("Begin simulation (CN)...")
    y0 = sim.init
    # Warmup: one CN step with S_prev = S^0 (first-order accurate at startup)
    y1, S0, diag0, mo_res0 = _cn_warmup(t_outer[0], dt_s, y0)
    _, (y_hist, _, diag_hist, mo_hist) = jax.lax.scan(_scan_outer, init=(y1, S0, diag0, mo_res0), xs=t_outer)
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
        y2, dydt1, diag1, mo_res1 = _ab2(t, dt_s, y1, dydt0)

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
    y1, dydt0, diag0, mo_res0 = _euler(t_outer[0], dt_s_init, y0)  # Warmup: one Euler step
    _, (y_hist, _, diag_hist, mo_hist) = jax.lax.scan(_scan_outer, init=(y1, dydt0, diag0, mo_res0), xs=t_outer)
    timer.finalize()

    return y_hist, diag_hist, mo_hist, t_outer
