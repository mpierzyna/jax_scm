from __future__ import annotations

from typing import Tuple

import jax
from jax import numpy as jnp

from scm.interfaces import Simulation, ProgVarsT, DiagVarsT, ModelFn
from scm.mo import MOResult
from scm.config import Namelist
from scm.time_stepping.explicit import get_euler_step_fn, get_ab2_step_fn
from scm.time_stepping.implicit import get_cn_step_fn
from scm.time_stepping.utils import IterationTimer


def simulate(
    model: ModelFn,
    sim: Simulation,
    cfg: Namelist,
) -> Tuple[ProgVarsT, DiagVarsT, MOResult, jnp.ndarray]:
    print("Config:", cfg)

    if cfg.time_int == "explicit":
        if cfg.adaptive_timestep is not None:
            # AB2 with adaptive time stepping
            return simulate_ab2_adaptive(
                model=model,
                sim=sim,
                cfl_max=cfg.adaptive_timestep.cfl_max,
                dt_s_init=cfg.dt_s,
                dt_s_max=cfg.adaptive_timestep.dt_s_max,
                dt_s_out=cfg.dt_s_out,
            )
        else:
            # AB2 with fixed time stepping
            return simulate_ab2_fixed(
                model=model,
                sim=sim,
                dt_s=cfg.dt_s,
                dt_s_out=cfg.dt_s_out,
            )
    elif cfg.time_int == "implicit":
        # Semi-implicit CN
        return simulate_cn(
            model=model,
            sim=sim,
            dt_s=cfg.dt_s,
            dt_s_out=cfg.dt_s_out,
        )
    else:
        raise ValueError(f"Invalid time_int: {cfg.time_int}")


def simulate_ab2_fixed(
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

    def _scan_inner(carry, t):
        """Advance model by one step but don't accumulate outputs"""
        y1, dydt0, _, _ = carry
        y2, dydt1, diag1, mo_res1 = _ab2(t, dt_s, y1, dydt0)
        return (y2, dydt1, diag1, mo_res1), None

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


def simulate_ab2_adaptive(
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

    def _get_dt(Km: jnp.ndarray, Kh: jnp.ndarray) -> jnp.ndarray:
        """Compute adaptive timestep based on CFL condition for diffusion."""
        Km_max = jnp.max(Km)
        Kh_max = jnp.max(Kh)
        K_max = jnp.clip(jnp.maximum(Km_max, Kh_max), min=1e-6)  # avoid zero division
        dt = cfl_max * sim.grid.dz**2 / K_max

        return jnp.minimum(dt, dt_s_max)

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

    def _while_cond(carry):
        """Condition for adaptive stepping loop"""
        *_, t_left = carry
        return t_left > 0

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

    def _scan_inner(carry, t):
        """Advance model by one CN step without accumulating output."""
        y1, S_prev, _, _ = carry
        y2, S1, diag1, mo_res1 = _cn(t, dt_s, y1, S_prev)
        return (y2, S1, diag1, mo_res1), None

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
