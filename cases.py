from typing import Tuple, Dict

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import ProgVars, TransientForcing
from scm.closures.mynn import ProgVarsMYNN


def get_gabls1(
    Nz: int = 128, plot: bool = False, random_seed: int = 0
) -> Tuple[StaggeredGrid, ProgVarsMYNN, TransientForcing]:
    ## Grid
    grid = StaggeredGrid(H=400, Nz=Nz)
    z_inv = 100

    ## Forcing
    # Geostrophic wind
    ug = jnp.ones(Nz) * 8.0  # m/s
    vg = jnp.zeros(Nz)  # m/s

    # Surface temperature forcing
    th_s_0 = 263.5  # K
    th_s_fn = lambda t_s: th_s_0 - 0.25 * t_s / (60 * 60)  # K, 0.25 K per hour cooling

    forcing = TransientForcing(
        u_geo=lambda t_s: ug,
        v_geo=lambda t_s: vg,
        f_c=1.39e-4,  # 1/s, ~73 deg latitude
        th_s=th_s_fn,
        w_q_s=lambda t_s: jnp.array(0.0),  # g/kg m/s
    )

    ## Initial conditions
    # Initial wind profile
    u = jnp.copy(ug)  # .at[0].set(0.0)  # geostrophic wind but with no-slip at surface
    v = jnp.copy(vg)

    # Initial temperature
    th = jnp.ones(Nz) * 265.0  # K
    th = jnp.where(grid.z > z_inv, th + 0.01 * (grid.z - z_inv), th)  # capping inversion
    th = jnp.where(
        grid.z < 50, th + 0.1 * jax.random.normal(key=jax.random.key(random_seed), shape=(Nz,)), th
    )  # random 0.1K perturbation near surface

    # Initial TKE
    tke = jnp.zeros(grid.Nz)
    tke = jnp.where(grid.z < 250, 0.4 * (1 - grid.z / 250) ** 3, tke)  # m^2 s^-2

    init = ProgVarsMYNN(u=u, v=v, thv=th, q_sq=2 * tke)

    ## Surface model (not in use)
    # z0m = z0h = 0.1  # m, roughness lengths for momentum and heat
    # beta_m = 4.8  # MOST momentum stability coefficent
    # beta_h = 7.8  # MOST heat stability coefficient
    #
    # lsm = {
    #     "z0m": z0m,
    #     "z0h": z0h,
    #     "beta_m": beta_m,
    #     "beta_h": beta_h,
    # }

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

        axarr[2].plot(tke, grid.z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

        # Forcing
        fig, axarr = plt.subplots(ncols=2, figsize=(8, 2), width_ratios=[1, 3], layout="constrained")
        axarr[0].plot(ug, grid.z, label="ug")
        axarr[0].plot(vg, grid.z, label="vg")
        axarr[0].set_xlabel("Geostrophic Wind (m/s)")
        axarr[0].legend()

        t = jnp.array([0, 9 * 60 * 60])  # 0 and 9 hours
        axarr[1].plot(t, th_s_fn(t))
        axarr[1].set_xlabel("Time, s")
        axarr[1].set_ylabel("Surface Potential Temperature (K)")

        fig.show()

    return grid, init, forcing


def get_ysu(
    Nz: int = 138, plot: bool = False, debug_dt: float = 0
) -> Tuple[StaggeredGrid, ProgVarsMYNN, TransientForcing]:
    """Initial conditions and forcing from HND06

    Use debug_dt to shift the time of the forcing functions for debugging purposes.
    """
    # Grid
    grid = StaggeredGrid(H=2750, Nz=Nz)
    z_inv = 500.0  # Inversion height in m

    # Initial conditions
    th = jnp.ones(grid.Nz) * 300.0  # K
    th = jnp.where(grid.z > z_inv, th + 0.01 * (grid.z - z_inv), th)  # linear decrease above inversion

    q = jnp.ones(grid.Nz) * 15.0  # g/kg
    q = jnp.where(grid.z > z_inv, q - 0.01 * (grid.z - z_inv), q)  # linear decrease above inversion up to 1500m
    q = jnp.where(grid.z > 1500, 5.0, q)  # constant above 1500m
    q = q / 1000

    thv = th * (1 + 0.61 * q)  # virtual potential temperature

    u = jnp.ones(grid.Nz) * 15.0  # m/s
    u = jnp.where(grid.z < z_inv, (15 / 500) * grid.z, u)  # linear increase to 15 m/s at z_inv

    v = jnp.zeros(grid.Nz)

    init = ProgVarsMYNN(
        u=u,
        v=v,
        thv=thv,
        q_sq=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    # Forcing
    @jax.jit
    def _shfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface heat flux as function of time in seconds after simulation begin."""
        t_h = (t_s + debug_dt) / 3600.0  # time in hours
        shfx = jnp.sin((t_h + 2) * jnp.pi / 12) * 400  # W/m2 = J/(s m2)
        shfx = shfx / (consts.rho_0 * consts.cp)  # convert to (K m/s)
        return shfx

    @jax.jit
    def _lhfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface latent heat flux as function of time in seconds after simulation begin."""
        t_h = (t_s + debug_dt) / 3600.0  # time in hours
        lhfx = jnp.sin(t_h * jnp.pi / 12) * 200  # W/m2
        lhfx = lhfx / 1225  # convert to (g/kg m/s)  # todo: correct like this?
        return lhfx

    @jax.jit
    def _u_geo(_) -> jnp.ndarray:
        """Constant geostrophic wind."""
        return jnp.ones(grid.Nz) * 15.0

    @jax.jit
    def _v_geo(_) -> jnp.ndarray:
        """Constant geostrophic wind."""
        return jnp.zeros(grid.Nz)

    forcing = TransientForcing(u_geo=_u_geo, v_geo=_v_geo, f_c=1.39e-4, w_th_s=_shfx, w_q_s=_lhfx)

    if plot:
        # Initial conditions
        fig, (ax_uv, ax_th, ax_q) = plt.subplots(ncols=3, figsize=(9, 4), layout="constrained")
        ax_uv.plot(init.u, grid.z, label="u")
        ax_uv.plot(init.v, grid.z, label="v")
        ax_uv.set_xlabel("Wind (m/s)")
        ax_uv.set_ylabel("Height (m)")
        ax_uv.legend()
        ax_th.plot(init.thv, grid.z)
        ax_th.set_xlabel("Potential Temperature (K)")
        # ax_q.plot(init.q * 1000, grid.z)
        # ax_q.set_xlabel("Specific Humidity (g/kg)")
        fig.show()

        # Forcing plots
        t_plot = jnp.linspace(0, 12 * 3600, 100)  # 0 to 12 hours
        shfx_plot = jax.vmap(forcing.w_th_s)(t_plot)
        lhfx_plot = jax.vmap(forcing.w_q_s)(t_plot)
        fig, (ax_shfx, ax_lhfx) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
        ax_shfx.plot(t_plot / 3600, shfx_plot * 1216)  # convert back to W/m2 for plotting
        ax_shfx.set_xlabel("Time (hours)")
        ax_shfx.set_ylabel("Surface Sensible Heat Flux (W/m²)")
        ax_lhfx.plot(t_plot / 3600, lhfx_plot * 1225)  # convert back to W/m2 for plotting
        ax_lhfx.set_xlabel("Time (hours)")
        ax_lhfx.set_ylabel("Surface Latent Heat Flux (W/m²)")
        fig.show()

    return grid, init, forcing


def get_ekman(Nz: int = 100, plot: bool = False):
    """Ekman spiral initial conditions and forcing."""
    grid = StaggeredGrid(H=1000, Nz=Nz)

    ug = 10.0 * jnp.ones(grid.Nz)
    vg = jnp.zeros(grid.Nz)

    forcing = TransientForcing(
        u_geo=lambda t: ug,
        v_geo=lambda t: vg,
        f_c=1e-4,
        w_th_s=lambda t: jnp.array(0.0),  # neutral stratification
        w_q_s=lambda t: jnp.array(0.0),
    )

    thv = jnp.ones(grid.Nz) * 280.0
    thv = jnp.where(grid.z > 400, thv + 0.01 * (grid.z - 500), thv)  # weak inversion above 500m

    init = ProgVarsMYNN(
        u=ug.copy(),
        v=vg.copy(),
        thv=thv,
        q_sq=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    if plot:
        fig, (ax_uv, ax_th) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
        ax_uv.plot(init.u, grid.z, label="u")
        ax_uv.plot(init.v, grid.z, label="v")
        ax_uv.set_xlabel("Wind (m/s)")
        ax_uv.set_ylabel("Height (m)")
        ax_uv.legend()
        ax_th.plot(init.thv, grid.z)
        ax_th.set_xlabel("Potential Temperature (K)")
        fig.show()

    return grid, init, forcing


def get_wangara(Nz: int = 50, plot: bool = False) -> Tuple[StaggeredGrid, ProgVarsMYNN, TransientForcing]:
    """Wangara initial conditions and forcing."""
    import pandas as pd

    ## Grid
    grid = StaggeredGrid(H=2000, Nz=Nz)

    ## Forcing
    # t_s = 0 corresponds to 00 local time
    w_thl_fn = lambda t_s: 2.16e-1 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  # K m/s
    w_qw_fn = lambda t_s: 2.29e-5 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  #  m/s
    dthl_dz_top = 0.0075  # K/m

    u_g = jnp.where(
        grid.z < 1000,
        -5.5 + 2.9e-3 * grid.z,  # linear decrease from -5.5m/s to -2.6m/s at 1000m
        -2.6 + 1.4e-3 * (grid.z - 1000),  # linear decrease from -2.6m/s to -1.2m/s at 2000m
    )
    v_g = jnp.zeros(grid.Nz)

    forcing = TransientForcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=1.39e-4,
        w_th_s=w_thl_fn,
        w_q_s=w_qw_fn,
        dth_dz_top=dthl_dz_top,
    )

    ## Initial conditions from observations
    df = pd.read_csv("data/wangara/input.dat", sep="\t", names=["u", "v", "th"])
    df["z"] = jnp.linspace(0, 2000, len(df))

    init = ProgVarsMYNN(
        u=jnp.interp(grid.z, df["z"].values, df["u"].values),
        v=jnp.interp(grid.z, df["z"].values, df["v"].values),
        thv=jnp.interp(grid.z, df["z"].values, df["th"].values),
        q_sq=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    if plot:
        # Initial conditions
        fig, (ax_uv, ax_th) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
        ax_uv.plot(init.u, grid.z, label="u")
        ax_uv.plot(init.v, grid.z, label="v")
        ax_uv.set_xlabel("Wind (m/s)")
        ax_uv.set_ylabel("Height (m)")
        ax_uv.legend()
        ax_th.plot(init.thv, grid.z)
        ax_th.set_xlabel("Potential Temperature (K)")
        fig.show()

        # Forcing plots
        t = jnp.linspace(0, 16 * 3600, 100)  # 16 hours
        fig, (ax_shfx, ax_lhfx) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
        ax_shfx.plot(t / 3600, forcing.w_th_s(t))
        ax_shfx.set_xlabel("Time (hours)")
        ax_shfx.set_ylabel("Surface Sensible Heat Flux (K m/s)")
        ax_lhfx.plot(t / 3600, forcing.w_q_s(t))
        ax_lhfx.set_xlabel("Time (hours)")
        ax_lhfx.set_ylabel("Surface Latent Heat Flux (m/s)")
        fig.show()

    return grid, init, forcing


if __name__ == "__main__":
    # get_gabls1(plot=True)
    # get_ekman(plot=True)
    # get_ysu(plot=True, debug_dt=0 * 60)
    get_wangara(plot=True)
