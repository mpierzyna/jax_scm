import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from scm import consts, convert
from scm.config import load_namelist
from scm.forcing.interp import get_ts_interp_fn
from scm.grid import StaggeredGrid
from scm.interfaces import Forcing, Simulation
from scm.io.local import out_to_ds
from scm.mo import BusingerDyerAltSimFuncs, MOSettings
from scm.mynn.interfaces import ProgVarsMYNN
from scm.mynn.model import init_model
from scm.time_stepping.base import simulate


def expand_array(a: jnp.ndarray, z_mask: jnp.ndarray, Nt: int, Nz: int) -> jnp.ndarray:
    """Expand 1D array `a` to 2D array (time, z) using `z_mask`."""
    a_exp = jnp.zeros((Nt, Nz))
    a_exp = jnp.where(z_mask, 1, a_exp)
    a_exp = a_exp * a[:, None]
    return a_exp


def get_gabls3(use_lf: bool, Nz: int = 100, plot: bool = False) -> Simulation:
    """Get a GABLS3 simulation setup.

    References
    ----------
    Bosveld, Fred C., et al. “The Third GABLS Intercomparison Case for Evaluation Studies of Boundary-Layer Models.
    Part A: Case Selection and Set-Up.” Boundary-Layer Meteorology, vol. 152, no. 2, Aug. 2014, pp. 133–56.
    Springer Link, https://doi.org/10.1007/s10546-014-9917-3.

    """

    # MO settings; no other soil or vegetation parameters implemented
    mo_settings = MOSettings(
        z0m=0.15,
        z0h=0.0015,
        sim_funcs=BusingerDyerAltSimFuncs(),
    )

    ## Grid
    grid = StaggeredGrid(H=3000, Nz=Nz)  # expecting BLH of 2000m

    ## Initial conditions
    # Initial wind profile
    df_uv_init = pd.read_csv("ic_bc/ic_u_v.csv")
    u = jnp.copy(jnp.interp(grid.z, jnp.array(df_uv_init["z"]), jnp.array(df_uv_init["U"])))
    v = jnp.copy(jnp.interp(grid.z, jnp.array(df_uv_init["z"]), jnp.array(df_uv_init["V"])))

    # Initial air temperature
    df_tc_qv_init = pd.read_csv("ic_bc/ic_tc_q.csv")
    z = jnp.array(df_tc_qv_init["z"])
    tk = jnp.array(df_tc_qv_init["TC"] + 273.15)  # convert to K
    tk = jnp.interp(grid.z, z, tk)

    # Initial specific humidity
    qv = jnp.array(df_tc_qv_init["q"] / 1000)  # kg/kg
    qv = jnp.interp(grid.z, z, qv)

    p_s = 1024.4 * 100  # surface pressure in Pa
    p, rho = convert.p_rho_from_tk(t=tk, qv=qv, z=grid.z, p_s=p_s)
    th = convert.tk_to_th(tk=tk, p_hPa=p / 100)  # potential temperature

    # Initial TKE
    tke = jnp.zeros(grid.Nz)
    tke = jnp.where(grid.z < 1000, 0.4 * (1 - grid.z / 1000) ** 3, tke)  # m^2 s^-2

    init = ProgVarsMYNN(u=u, v=v, th=th, qke=2 * tke, qv=qv)

    if plot:
        # Initial conditions
        fig, axarr = plt.subplots(ncols=3, figsize=(8, 2), sharey="row", layout="constrained")
        axarr[0].plot(u, grid.z, label="u")
        axarr[0].plot(v, grid.z, label="v")
        axarr[0].set_xlabel("Wind (m/s)")
        axarr[0].set_ylabel("Height (m)")
        axarr[0].legend()

        axarr[1].plot(th, grid.z)
        axarr[1].set_xlabel("Potential Temperature (K)")

        ax2 = axarr[1].twiny()
        ax2.plot(qv, grid.z, color="C1")
        ax2.set_xlabel("Specific Humidity (kg/kg)", color="C1")
        ax2.tick_params(axis="x", colors="C1")

        axarr[2].plot(tke, grid.z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

    ## Forcing
    # Geostrophic wind
    df_uv_geo_0 = pd.read_csv("ic_bc/bc_uvgeo_sfc.csv")
    time_h = jnp.array(df_uv_geo_0["t"])  # time in hours
    ug_0 = jnp.array(df_uv_geo_0["ug_0"])  # surface geostrophic wind in m/s
    vg_0 = jnp.array(df_uv_geo_0["vg_0"])  # surface geostrophic wind in m/s

    ug = ug_0[:, None] + (-2.0 - ug_0[:, None]) * (grid.z / 2000)  # m/s, linear decrease to -2 m/s at 2000m
    ug = jnp.where(grid.z > 2000, -2.0, ug)  # m/s, constant above 2000m
    vg = vg_0[:, None] + (2.0 - vg_0[:, None]) * (grid.z / 2000)  # m/s, linear decrease to 2 m/s at 2000m
    vg = jnp.where(grid.z > 2000, 2.0, vg)  # m/s, constant above 2000m

    ug_fn = get_ts_interp_fn(time_s=time_h * 3600, data=ug)
    vg_fn = get_ts_interp_fn(time_s=time_h * 3600, data=vg)

    # Sensible heat flux
    df_shfx = pd.read_csv("ic_bc/bc_shf.csv")
    time_h = jnp.array(df_shfx["t"])  # time in hours
    w_th_s = jnp.array(df_shfx["shf"]) / (consts.cp * consts.rho_0)  # W/m^2 to K m / s
    w_th_s_fn = get_ts_interp_fn(time_s=time_h * 3600, data=w_th_s)

    # Latent heat flux
    df_lhfx = pd.read_csv("ic_bc/bc_lhf.csv")
    time_h = jnp.array(df_lhfx["t"])  # time in hours
    w_qv_s = jnp.array(df_lhfx["lhf"]) / (consts.L_v * consts.rho_0)  # W/m^2 to (kg/kg) m / s
    w_qv_s_fn = get_ts_interp_fn(time_s=time_h * 3600, data=w_qv_s)

    # Plot forcings
    if plot:
        t = jnp.linspace(0, 24 * 3600, 50)  # 0 to 24 hours

        fig, (ax_ug, ax_vg, ax_shfx) = plt.subplots(
            ncols=3,
            figsize=(12, 4),
            layout="constrained",
            width_ratios=[1, 1, 2],
        )
        ax_lhfx = ax_shfx.twinx()

        i = ax_ug.imshow(
            jax.vmap(ug_fn)(t).T,  # (time, z)
            extent=(t[0], t[-1], grid.z[0], grid.z[-1]),
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(i, ax=ax_ug)

        i = ax_vg.imshow(
            jax.vmap(vg_fn)(t).T,  # (time, z)
            extent=(t[0], t[-1], grid.z[0], grid.z[-1]),
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(i, ax=ax_vg)

        ax_shfx.plot(t, w_th_s_fn(t), label="w_th_s (K m/s)", color="C0")
        ax_lhfx.plot(t, w_qv_s_fn(t), label="w_qv_s (kg/kg m/s)", color="C1")
        ax_shfx.set_xlabel("Time (s)")

        fig.show()

    ## Large scale tendencies
    # Vertical tendency
    df_omega = pd.read_csv("ic_bc/tend_omega.csv")
    time_h = jnp.array(df_omega["t"])  # time in hours
    omega = expand_array(
        a=jnp.array(df_omega["omega"]),  # Pa/s
        z_mask=(grid.z >= 1500) & (grid.z <= 5000),
        Nt=len(time_h),
        Nz=grid.Nz,
    )
    w = convert.w_eff(omega=omega, rho=rho)  # m/s
    w_fn = get_ts_interp_fn(time_s=time_h * 3600, data=w)

    # Temperature advection
    df_tk_adv = pd.read_csv("ic_bc/tend_t_adv.csv")
    time_h = jnp.array(df_tk_adv["t"])  # time in hours
    tk_adv = jnp.array(df_tk_adv["T_adv"])  # K/s
    tk_adv = expand_array(
        a=tk_adv,
        z_mask=(grid.z >= 200) & (grid.z <= 1000),
        Nt=len(time_h),
        Nz=grid.Nz,
    )
    th_adv = convert.tk_to_th(tk=tk_adv, p_hPa=p / 100)
    th_adv_fn = get_ts_interp_fn(time_s=time_h * 3600, data=th_adv)

    # Humidity advection
    df_qv_adv = pd.read_csv("ic_bc/tend_q_adv.csv")
    time_h = jnp.array(df_qv_adv["t"])  # time in hours
    qv_adv = jnp.array(df_qv_adv["q_adv"])  # kg/kg/s
    qv_adv = expand_array(
        a=qv_adv,
        z_mask=(grid.z >= 200) & (grid.z <= 1000),
        Nt=len(time_h),
        Nz=grid.Nz,
    )
    qv_adv_fn = get_ts_interp_fn(time_s=time_h * 3600, data=qv_adv)

    # Velocity advection
    df_uv_adv = pd.read_csv("ic_bc/tend_uv_adv.csv")
    time_h = jnp.array(df_uv_adv["t"])  # time in hours
    u_adv = jnp.array(df_uv_adv["U_adv"])  # m/s^2
    v_adv = jnp.array(df_uv_adv["V_adv"])  # m/s^2
    u_adv = expand_array(
        a=u_adv,
        z_mask=(grid.z >= 200) & (grid.z <= 1000),
        Nt=len(time_h),
        Nz=grid.Nz,
    )
    v_adv = expand_array(
        a=v_adv,
        z_mask=(grid.z >= 200) & (grid.z <= 1000),
        Nt=len(time_h),
        Nz=grid.Nz,
    )
    u_adv_fn = get_ts_interp_fn(time_s=time_h * 3600, data=u_adv)
    v_adv_fn = get_ts_interp_fn(time_s=time_h * 3600, data=v_adv)

    def ls_tends_fn(t_s: jnp.ndarray, state: ProgVarsMYNN, grads: ProgVarsMYNN, _) -> ProgVarsMYNN:
        w = w_fn(t_s)
        dth_dz_mean = (grads.th[1:] + grads.th[:-1]) / 2  # mean dth/dz at full levels

        return ProgVarsMYNN(
            u=u_adv_fn(t_s),
            v=v_adv_fn(t_s),
            th=th_adv_fn(t_s) - w * dth_dz_mean,  # horizontal and vertical advection
            qv=qv_adv_fn(t_s),
            qke=jnp.zeros_like(state.qke),  # no TKE tendencies from large scale forcings
        )

    if plot:
        fig, axarr = plt.subplots(ncols=5, figsize=(12, 3), sharey="row", layout="constrained")
        tend_fns = [w_fn, th_adv_fn, qv_adv_fn, u_adv_fn, v_adv_fn]
        tend_labels = ["w", "th_adv", "qv_adv", "u_adv", "v_adv"]
        for ax, tend_fn, label in zip(axarr, tend_fns, tend_labels):
            data = jax.vmap(tend_fn)(t)  # (time, z)
            i = ax.imshow(
                data.T,
                extent=(t[0], t[-1], grid.z[0], grid.z[-1]),
                aspect="auto",
                origin="lower",
            )
            ax.set_title(label)
            fig.colorbar(i, ax=ax)
        fig.show()

    # Gather all forcings
    forcing = Forcing(
        u_geo=ug_fn,
        v_geo=vg_fn,
        f_c=convert.get_fc(lat_deg=51.9711),  # Cabauw
        w_qv_s=w_qv_s_fn,
        w_th_s=w_th_s_fn,
        ls_tends=ls_tends_fn if use_lf else None,
    )

    return Simulation(
        name="GABLS3_manual",
        grid=grid,
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        t_start_s=0,
        t_end_s=24 * 60 * 60,
        th_ref=273.15 + 20.0,
    )


def run(use_lf: bool):
    cfg = load_namelist("namelist_cn.yaml")
    sim = get_gabls3(use_lf, Nz=100, plot=True)
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    ds = out_to_ds(
        out,
        sim,
        time=pd.date_range(
            "2006-07-01T12:00",
            freq=f"{cfg.dt_s_out:.0f}s",
            periods=len(out),
        ),
    )
    ds.to_netcdf(f"out_{sim.grid.Nz}_{'lf' if use_lf else 'no_lf'}.nc")
    print("Written to disk.")


if __name__ == "__main__":
    run(use_lf=True)
    run(use_lf=False)
