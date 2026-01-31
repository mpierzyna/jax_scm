from __future__ import annotations

import dataclasses
import logging
from typing import Tuple, Literal, TypeVar

import jax
import jax.experimental.checkify
import jax.numpy as jnp
import numpy as np

from scm.closures.mynn import init_mynn, ProgVarsMYNN, DiagVarsMYNN
from scm.forcing.era5 import get_era5_sim
from scm.forcing.interp import interp_dtindex
from scm.grad import d_dz
from scm.grid import StaggeredGrid
from scm.interfaces import ModelFn, StaticForcing
from scm.io.local import make_dataset
from scm.mo import init_mo_sfc, MOResult, BusingerDyerSimFuncs, SurfaceProperties
from scm.time_stepping import simulate_adaptive_dt
import scm.conversions as conv
import cases

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("scm")

T = TypeVar("T")


def init_model(
    grid: StaggeredGrid,
    sfc: SurfaceProperties,
    prescribe_sfc_heat: Literal["w_th_s", "th_s"],
) -> ModelFn[ProgVarsMYNN, DiagVarsMYNN]:
    # Create MO model
    z_mo = float(grid.z[0])
    eval_mo = init_mo_sfc(
        z0m=sfc.z0m,
        z0h=sfc.z0h,
        z=z_mo,
        sim_funcs=sfc.sim_funcs,
        prescribe=prescribe_sfc_heat,
    )

    # Init MYNN scheme
    closure_fn = init_mynn(grid=grid)

    @jax.jit
    def _model(state: ProgVarsMYNN, forcing: StaticForcing) -> Tuple[ProgVarsMYNN, DiagVarsMYNN, MOResult]:
        # Unpack state
        u, v, thv, q_sq, qv = state.u, state.v, state.thv, state.q_sq, state.qv
        th_0 = conv.thv_to_th(thv=thv[0], qv=qv[0])

        # Unpack forcing
        f_c = forcing.f_c
        u_geo, v_geo = forcing.u_geo, forcing.v_geo
        w_th_s, th_s, w_qv_s = forcing.w_th_s, forcing.th_s, forcing.w_qv_s

        # Run MO for surface coupling
        mo_res: MOResult = eval_mo(u_0=u[0], v_0=v[0], th_0=th_0, qv_0=qv[0], w_th_s=w_th_s, th_s=th_s, w_qv_s=w_qv_s)

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = d_dz(u, dz=grid.dz, bot="edge", top=0.0)
        dv_dz = d_dz(v, dz=grid.dz, bot="edge", top=0.0)
        dthv_dz = d_dz(thv, dz=grid.dz, bot="edge", top=forcing.dth_dz_top)  # todo: can I assume dry pot temp here?
        dqv_dz = d_dz(qv, dz=grid.dz, bot="edge", top=0.0)  # todo: upper BC?
        dqsq_dz = d_dz(q_sq, dz=grid.dz, bot="edge", top=0.0)  # todo: lower BC = 0 ok?
        grads = ProgVarsMYNN(u=du_dz, v=dv_dz, thv=dthv_dz, q_sq=dqsq_dz, qv=dqv_dz)

        # Execute closure to get fluxes
        diag = closure_fn(state, grads, mo_res)

        # Update fluxes with MO results
        u_w = diag.u_w.at[0].set(mo_res.u_w)
        v_w = diag.v_w.at[0].set(mo_res.v_w)
        w_thv = diag.w_thv.at[0].set(mo_res.w_thv)
        w_th = diag.w_th.at[0].set(mo_res.w_th)  # this is just for diagnostics
        w_qv = diag.w_qv.at[0].set(mo_res.w_qv)
        w_qke = diag.w_qke  # todo: any update for TKE needed?

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_thv = (w_thv[1:] - w_thv[:-1]) / grid.dz
        div_w_qv = (w_qv[1:] - w_qv[:-1]) / grid.dz
        div_w_qke = (w_qke[1:] - w_qke[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f_c * v - f_c * v_geo - div_u_w
        v_tend = -f_c * u + f_c * u_geo - div_v_w
        thv_tends = -div_w_thv  # todo: some geostrophic wind term?
        qv_tends = -div_w_qv
        q_sq_tend = diag.q_sq_P_S + diag.q_sq_P_B - diag.q_sq_eps + div_w_qke

        # Gather tendencies and updated diagnostics (because MOST values added!)
        tends = ProgVarsMYNN(u=u_tend, v=v_tend, thv=thv_tends, qv=qv_tends, q_sq=q_sq_tend)
        diag = dataclasses.replace(diag, u_w=u_w, v_w=v_w, w_thv=w_thv, w_th=w_th, w_qv=w_qv)
        return tends, diag, mo_res

    return _model


def init_from_xr(f: str, t: float) -> ProgVarsMYNN:
    """Initialize model state from xarray dataset at time t."""
    import xarray as xr

    ds = xr.open_dataset(f)
    ds_t = ds.sel(time=t, method="nearest")
    state = ProgVarsMYNN(
        u=jnp.array(ds_t["u"].values),
        v=jnp.array(ds_t["v"].values),
        thv=jnp.array(ds_t["thv"].values),
        q_sq=jnp.array(ds_t["q_sq"].values),
    )
    return state


if __name__ == "__main__":
    # Ekman spiral
    # grid, init, forcing = cases.get_ekman(Nz=100)

    # YSU test case
    # t_debug = 0
    # grid, init, forcing = cases.get_ysu()

    # GABLS
    # sim = cases.get_gabls1(Nz=64)

    # Wangara
    sim = cases.get_wangara(Nz=200)

    # Cabauw from ERA5
    # sim = get_era5_sim(
    #     name="Cabauw_Test",
    #     lat_deg=52.0,
    #     lon_deg=5.0,
    #     grid=StaggeredGrid(Nz=100, H=3000.0),
    #     # time_slice=slice("2025-07-01", "2025-07-03"),
    #     time_slice="2025-07-01",
    # )

    # Init and run model
    sfc = SurfaceProperties(z0m=0.1, z0h=0.1, sim_funcs=BusingerDyerSimFuncs())
    model = init_model(sim.grid, sfc, prescribe_sfc_heat="th_s" if sim.forcing.w_th_s is None else "w_th_s")
    # state_hist, diag_hist, mo_hist, t = simulate(
    #     model,
    #     init,
    #     forcing,
    #     dt_s=0.001,
    #     t_start_s=9 * 60 * 60,
    #     t_end_s=16 * 60 * 60,
    #     dt_out_s=60 * 5,
    # )
    state_hist, diag_hist, mo_hist, t = simulate_adaptive_dt(
        model=model,
        sim=sim,
        dt_s_init=0.001,
        dt_s_max=1,
        cfl_max=0.1,
        dt_s_out=60 * 5,
    )

    # Prepare time axis
    if sim.t_index is not None:
        time = interp_dtindex(t_s=np.array(t), idx=sim.t_index)
    else:
        time = t / 3600.0  # convert to hours

    # Save output
    ds = make_dataset(state_hist, diag_hist, mo_hist, time=time, grid=sim.grid)
    ds.to_netcdf("out.nc")
    print("Written to disk.")

    # Unstack for plotting
    # state_hist = unstack_hist(state_hist)
    # diag_hist = unstack_hist(diag_hist)
    #
    # plot_hist(state_hist, t, grid, plot_sfc_val=True)
    # plot_hist(diag_hist, t, grid, plot_sfc_val=True)
