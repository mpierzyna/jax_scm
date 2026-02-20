from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from scm.grad import d_dz
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, ModelFn, Forcing
from scm.mo import init_mo_sfc, MOResult
from scm.mynn.closure import init_closure, get_qke_sfc
from scm.mynn.interfaces import ProgVarsMYNN, DiagVarsMYNN


def init_model(sim: Simulation[ProgVarsMYNN, DiagVarsMYNN], implicit: bool) -> ModelFn[ProgVarsMYNN, DiagVarsMYNN]:
    """Initialize MYNN model function for time stepper."""
    # Make grid and forcing available locally
    grid: StaggeredGrid = sim.grid
    forcing: Forcing = sim.forcing

    # Create MO model
    z_mo = float(grid.z[0])
    eval_mo = init_mo_sfc(
        z0m=sim.mo_settings.z0m,
        z0h=sim.mo_settings.z0h,
        z=z_mo,
        sim_funcs=sim.mo_settings.sim_funcs,
        prescribe="th_s" if sim.forcing.w_th_s is None else "w_th_s",
    )

    # Init MYNN scheme
    closure_fn = init_closure(grid=grid)

    def _model(t_s: jnp.ndarray, state: ProgVarsMYNN) -> Tuple[ProgVarsMYNN, DiagVarsMYNN, MOResult]:
        """Model function takes state and forcing and returns tendencies and diagnostics."""
        # Unpack state
        u, v, th, qke, qv = state.u, state.v, state.th, state.qke, state.qv

        # Forcing for this step
        f_c = forcing.f_c
        u_geo = forcing.u_geo(t_s)
        v_geo = forcing.v_geo(t_s)
        w_th_s = forcing.w_th_s(t_s) if forcing.w_th_s is not None else None
        th_s = forcing.th_s(t_s) if forcing.th_s is not None else None
        w_qv_s = forcing.w_qv_s(t_s)

        # Run MO for surface coupling
        mo_res: MOResult = eval_mo(u_0=u[0], v_0=v[0], th_0=th[0], qv_0=qv[0], w_th_s=w_th_s, th_s=th_s, w_qv_s=w_qv_s)

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = d_dz(u, dz=grid.dz, bot="edge", top=0.0)
        dv_dz = d_dz(v, dz=grid.dz, bot="edge", top=0.0)
        dth_dz = d_dz(th, dz=grid.dz, bot="edge", top=forcing.dth_dz_top)
        dqv_dz = d_dz(qv, dz=grid.dz, bot="edge", top=0.0)  # todo: upper BC?

        qke_sfc = get_qke_sfc(u_st=mo_res.u_st)  # surface BC for qke
        dqke_dz = d_dz(qke, dz=grid.dz, bot=(state.qke[0] - qke_sfc) / grid.z[0], top=0.0)

        grads = ProgVarsMYNN(u=du_dz, v=dv_dz, th=dth_dz, qke=dqke_dz, qv=dqv_dz)

        # Execute closure to get fluxes
        # Lower boundary conditions for fluxes are applied INSIDE closure!
        diag = closure_fn(state, grads, mo_res)

        if implicit:
            # In implicit mode, divergence solved in time stepper.
            # All zero here, so only tendencies/forcing forwarded.
            div_u_w = jnp.zeros_like(u)
            div_v_w = jnp.zeros_like(v)
            div_w_th = jnp.zeros_like(th)
            div_w_qv = jnp.zeros_like(qv)
            div_w_qke = jnp.zeros_like(qke)
        else:
            # In explicit mode, compute flux divergence directly (half levels -> full levels)
            div_u_w = (diag.u_w[1:] - diag.u_w[:-1]) / grid.dz
            div_v_w = (diag.v_w[1:] - diag.v_w[:-1]) / grid.dz
            div_w_th = (diag.w_th[1:] - diag.w_th[:-1]) / grid.dz
            div_w_qv = (diag.w_qv[1:] - diag.w_qv[:-1]) / grid.dz
            div_w_qke = (diag.w_qke[1:] - diag.w_qke[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f_c * v - f_c * v_geo - div_u_w
        v_tend = -f_c * u + f_c * u_geo - div_v_w
        th_tends = -div_w_th  # todo: some geostrophic wind term?
        qv_tends = -div_w_qv
        qke_tends = diag.qke_P_S + diag.qke_P_B - diag.qke_eps + div_w_qke

        # Gather tendencies
        tends = ProgVarsMYNN(u=u_tend, v=v_tend, th=th_tends, qv=qv_tends, qke=qke_tends)

        # Add large scale tendencies if provided as tendencies
        if forcing.ls_tends is not None:
            ls_tends = forcing.ls_tends(t_s, state, grads, diag)
            tends = jax.tree_util.tree_map(lambda x, ls_x: x + ls_x, tends, ls_tends)

        return tends, diag, mo_res

    return _model
