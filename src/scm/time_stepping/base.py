"""Unified simulation entry point for JAX-SCM time integration."""

from __future__ import annotations

import jax
from jax import numpy as jnp

from scm import consts
from scm.config import Namelist
from scm.interfaces import ModelFn, Output, Simulation
from scm.mynn.closure import MYNNParams
from scm.mynn.interfaces import TendsVarsMYNN
from scm.time_stepping.explicit import get_ab2_step_fn, get_euler_step_fn
from scm.time_stepping.implicit import get_cn_step_fn
from scm.time_stepping.logging import get_logger_from_cfg
from scm.time_stepping.utils import StepCarry


def simulate(model: ModelFn, sim: Simulation, cfg: Namelist, params=None) -> Output:
    """Run a full SCM simulation and return the state trajectory.

    An Euler warmup step initialises the AB2 history, then the chosen scheme
    (explicit AB2 or implicit Crank-Nicolson) advances the state.  The outer
    loop saves one snapshot per ``cfg.dt_s_out`` interval; inner physics steps
    run at ``cfg.dt_s`` (or adaptively when ``cfg.adaptive_timestep`` is set).

    Parameters
    ----------
    model : ModelFn
        Right-hand-side function with signature
        ``(t_s, state, params) -> (tendencies, DiagVarsMYNN, MOResult)``.
    sim : Simulation
        Grid, initial conditions, forcing, and time bounds.
    cfg : Namelist
        Solver configuration (time step, integration method, logging, etc.).
    params : MYNNParams, optional
        Closure parameters.  Defaults to ``MYNNParams()`` when ``None``.

    Returns
    -------
    scm.interfaces.Output
        Stacked trajectory of model state, diagnosed variables,
        surface layer results, and state tendencies. Shape is `(N_out+1, ...)`.
        Note: state tendencies are only calculated correctly when using an explicit time-stepping method.

    Notes
    -----
    The initial state is prepended to the trajectory so that ``t_s[0] == sim.t_start_s``.
    """
    if params is None:
        params = MYNNParams()

    # Prepare time coordinates
    t_outer = jnp.arange(sim.t_start_s, sim.t_end_s, cfg.dt_s_out)
    logger = get_logger_from_cfg(cfg=cfg.logging, n_total=len(t_outer))

    # Configure the time integration stepper.  Each `_step_fn` returns
    # (new_carry, output, info) where `info` is a dict of scalar JAX arrays
    # that the logger may surface (empty for fixed-timestep schemes).
    if cfg.time_int == "explicit" and cfg.adaptive_timestep is not None:
        _warmup = get_euler_step_fn(model)
        _ab2 = get_ab2_step_fn(model)

        def _get_dt(diag):
            K_max = jnp.clip(jnp.maximum(jnp.max(diag.Km), jnp.max(diag.Kh)), min=consts.K_min)
            dt = cfg.adaptive_timestep.cfl_max * sim.grid.dz**2 / K_max
            return jnp.minimum(dt, cfg.adaptive_timestep.dt_s_max)

        def _step_fn(carry: StepCarry, t, dt_out):
            def _while_body(c):
                step_carry, i, t_curr, t_left, dt_min = c
                dt = jnp.minimum(_get_dt(step_carry.diag), t_left)
                new_carry = _ab2(step_carry, t_curr, dt, params)
                return (new_carry, i + 1, t_curr + dt, t_left - dt, jnp.minimum(dt_min, dt))

            loop_init = (carry, 0, t.astype(float), dt_out, jnp.array(jnp.inf))
            new_carry, n_inner, _, _, dt_min = jax.lax.while_loop(lambda c: c[3] > 0, _while_body, loop_init)
            info = {"n_inner": n_inner, "dt_min": dt_min}
            return new_carry, new_carry, info

    else:
        if cfg.time_int == "explicit":
            _warmup = get_euler_step_fn(model)
            _step = get_ab2_step_fn(model)
        else:
            _warmup, _step = get_cn_step_fn(model, sim.grid)

        rel_t_inner = jnp.arange(0, cfg.dt_s_out, cfg.dt_s) + cfg.dt_s

        def _step_fn(carry: StepCarry, t, dt_out):
            def _scan_inner(c, t_in):
                return _step(c, t_in, cfg.dt_s, params), None

            new_carry, _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
            return new_carry, new_carry, {}

    # Run the simulation loop
    logger.on_start()
    init_carry: StepCarry = _warmup(t_outer[0], cfg.dt_s, sim.init, params)

    def _outer_body(carry, t):
        # todo: out is just a copy of new_carry. Thus, it contains tendencies which cause memory overhead. fix this.
        new_carry, out, info = _step_fn(carry, t, cfg.dt_s_out)
        logger.on_outer_step(t + cfg.dt_s_out, info)
        return new_carry, out

    _, history = jax.lax.scan(_outer_body, init=init_carry, xs=t_outer)
    logger.on_end()

    # Set up tendencies as separate object type when outputting
    tends0 = TendsVarsMYNN(dudt=0.*sim.init.u, dvdt=0.*sim.init.v, dthdt=0.*sim.init.th,
                           dqvdt=0.*sim.init.qv, dqkedt=0.*sim.init.qke)    # Initial state has no tendencies
    tends_h = TendsVarsMYNN(dudt=history.prev_tends.u, dvdt=history.prev_tends.v, dthdt=history.prev_tends.th,
                            dqvdt=history.prev_tends.qv, dqkedt=history.prev_tends.qke)

    # Assemble Output by merging the initial state with the trajectory
    out0 = Output(
        state_traj=jax.tree_util.tree_map(lambda x: x[None], sim.init),
        diag_traj=jax.tree_util.tree_map(lambda x: x[None], init_carry.diag),
        mo_traj=jax.tree_util.tree_map(lambda x: x[None], init_carry.mo),
        t_s=jnp.array([sim.t_start_s]),
        tends_traj=jax.tree_util.tree_map(lambda x: x[None], tends0),
    )

    out_h = Output(
        state_traj=history.y,
        diag_traj=history.diag,
        mo_traj=history.mo,
        t_s=t_outer + cfg.dt_s_out,
        tends_traj=tends_h,
    )

    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), out0, out_h)
