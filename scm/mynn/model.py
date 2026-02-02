from __future__ import annotations

import dataclasses
from typing import Tuple

import jax

from scm.grad import d_dz
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, ModelFn, StaticForcing
from scm.mo import init_mo_sfc, MOResult
from scm.mynn.closure import init_closure
from scm.mynn.interfaces import ProgVarsMYNN, DiagVarsMYNN


def init_model(sim: Simulation[ProgVarsMYNN]) -> ModelFn[ProgVarsMYNN, DiagVarsMYNN]:
    """Initialize MYNN model function for time stepper."""
    # Make grid available locally
    grid: StaggeredGrid = sim.grid

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

    @jax.jit
    def _model(state: ProgVarsMYNN, forcing: StaticForcing) -> Tuple[ProgVarsMYNN, DiagVarsMYNN, MOResult]:
        """Model function takes state and forcing and returns tendencies and diagnostics."""
        # Unpack state
        u, v, th, qke, qv = state.u, state.v, state.th, state.qke, state.qv

        # Unpack forcing
        f_c = forcing.f_c
        u_geo, v_geo = forcing.u_geo, forcing.v_geo
        w_th_s, th_s, w_qv_s = forcing.w_th_s, forcing.th_s, forcing.w_qv_s

        # Run MO for surface coupling
        mo_res: MOResult = eval_mo(u_0=u[0], v_0=v[0], th_0=th[0], qv_0=qv[0], w_th_s=w_th_s, th_s=th_s, w_qv_s=w_qv_s)

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = d_dz(u, dz=grid.dz, bot="edge", top=0.0)
        dv_dz = d_dz(v, dz=grid.dz, bot="edge", top=0.0)
        dth_dz = d_dz(th, dz=grid.dz, bot="edge", top=forcing.dth_dz_top)
        dqv_dz = d_dz(qv, dz=grid.dz, bot="edge", top=0.0)  # todo: upper BC?
        dqke_dz = d_dz(qke, dz=grid.dz, bot="edge", top=0.0)  # todo: lower BC = 0 ok?
        grads = ProgVarsMYNN(u=du_dz, v=dv_dz, th=dth_dz, qke=dqke_dz, qv=dqv_dz)

        # Execute closure to get fluxes
        diag = closure_fn(state, grads, mo_res)

        # Update fluxes with MO results
        u_w = diag.u_w.at[0].set(mo_res.u_w)
        v_w = diag.v_w.at[0].set(mo_res.v_w)
        w_th = diag.w_th.at[0].set(mo_res.w_th)
        w_qv = diag.w_qv.at[0].set(mo_res.w_qv)
        w_qke = diag.w_qke  # todo: any update for TKE needed?

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_th = (w_th[1:] - w_th[:-1]) / grid.dz
        div_w_qv = (w_qv[1:] - w_qv[:-1]) / grid.dz
        div_w_qke = (w_qke[1:] - w_qke[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f_c * v - f_c * v_geo - div_u_w
        v_tend = -f_c * u + f_c * u_geo - div_v_w
        th_tends = -div_w_th  # todo: some geostrophic wind term?
        qv_tends = -div_w_qv
        qke_tends = diag.qke_P_S + diag.qke_P_B - diag.qke_eps + div_w_qke

        # Gather tendencies and updated diagnostics (because MOST values added!)
        tends = ProgVarsMYNN(u=u_tend, v=v_tend, th=th_tends, qv=qv_tends, qke=qke_tends)
        diag = dataclasses.replace(diag, u_w=u_w, v_w=v_w, w_th=w_th, w_qv=w_qv)
        return tends, diag, mo_res

    return _model
