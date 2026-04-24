from __future__ import annotations

import jax
from jax import numpy as jnp

from scm.config import Namelist
from scm.interfaces import Simulation, ModelFn, Output, MYNNParams
from scm.time_stepping.explicit import get_euler_step_fn, get_ab2_step_fn
from scm.time_stepping.implicit import get_cn_step_fn
from scm.time_stepping.utils import IterationTimer, StepCarry


def simulate(model: ModelFn, sim: Simulation, cfg: Namelist, params=None) -> Output:
    """Unified simulation entry point with a single outer loop."""
    if params is None:
        params = MYNNParams()

    if cfg.print_advanced_status:
        print("Config:", cfg)

    # Prepare time coordinates
    t_outer = jnp.arange(sim.t_start_s, sim.t_end_s, cfg.dt_s_out)
    if cfg.print_advanced_status:
        # todo: refactor timing/status printing. Pass only callback, so simulate can be jitted
        timer = IterationTimer(n_total=len(t_outer))

    # Configure the time integration stepper
    if cfg.time_int == "explicit" and cfg.adaptive_timestep is not None:
        # Adaptive AB2 stepper
        _warmup = get_euler_step_fn(model)
        _ab2 = get_ab2_step_fn(model)

        def _get_dt(diag):
            K_max = jnp.clip(jnp.maximum(jnp.max(diag.Km), jnp.max(diag.Kh)), min=1e-6)
            dt = cfg.adaptive_timestep.cfl_max * sim.grid.dz**2 / K_max
            return jnp.minimum(dt, cfg.adaptive_timestep.dt_s_max)

        def _step_fn(carry: StepCarry, t, dt_out):
            def _while_body(c):
                step_carry, i, t_curr, t_left = c
                dt = jnp.minimum(_get_dt(step_carry.diag), t_left)
                new_carry = _ab2(step_carry, t_curr, dt, params)
                return (new_carry, i + 1, t_curr + dt, t_left - dt)

            loop_init = (carry, 0, t.astype(float), dt_out)
            loop_final = jax.lax.while_loop(lambda c: c[-1] > 0, _while_body, loop_init)
            new_carry, i, _, _ = loop_final
            if cfg.print_advanced_status:
                jax.debug.print("Took {i} steps", i=i)
            return new_carry, new_carry

    else:
        # Fixed-timestep (AB2 or CN)
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
            return new_carry, new_carry

    # Run the simulation loop
    jax.debug.print("Begin simulation...")
    init_carry: StepCarry = _warmup(t_outer[0], cfg.dt_s, sim.init, params)

    def _outer_body(carry, t):
        new_carry, out = _step_fn(carry, t, cfg.dt_s_out)
        if cfg.print_advanced_status:
            jax.debug.callback(timer.callback, t + cfg.dt_s_out)
        return new_carry, out

    _, history = jax.lax.scan(_outer_body, init=init_carry, xs=t_outer)
    if cfg.print_advanced_status:
        timer.finalize()
    jax.debug.print("Simulation complete.")

    # Assemble Output by merging the initial state with the trajectory
    # Initial state (t=0) as an Output object
    out0 = Output(
        state_traj=jax.tree_util.tree_map(lambda x: x[None], sim.init),
        diag_traj=jax.tree_util.tree_map(lambda x: x[None], init_carry.diag),
        mo_traj=jax.tree_util.tree_map(lambda x: x[None], init_carry.mo),
        t_s=jnp.array([sim.t_start_s]),
    )

    # Simulation results (t > 0) as an Output object
    out_h = Output(
        state_traj=history.y,
        diag_traj=history.diag,
        mo_traj=history.mo,
        t_s=t_outer + cfg.dt_s_out,
    )

    # Merge initial state with trajectory
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), out0, out_h)
