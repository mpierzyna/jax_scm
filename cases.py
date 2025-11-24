from typing import Tuple, Dict

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import ProgVars, TransientForcing


def get_gables1(Nz: int = 128, plot: bool = False, random_seed: int = 0) -> Tuple[Dict, Dict, Dict, Dict]:
    # Grid
    H = 400
    zh = jnp.linspace(0, H, Nz + 1)
    z = 0.5 * (zh[:-1] + zh[1:])
    z_inv = 100

    # Coriolis parameter
    f = 1.39e-4  # 1/s, ~73 deg latitude

    # Geostrophic wind
    ug = jnp.ones_like(z) * 8.0  # m/s
    vg = jnp.zeros_like(z)  # m/s

    # Surface temperature forcing
    th_sfc_0 = 263.5  # K
    th_sfc_fn = lambda t_s: th_sfc_0 - 0.25 * t_s / (60 * 60)  # K, 0.25 K per hour cooling

    # Initial wind profile
    u = jnp.copy(ug).at[0].set(0.0)  # geostrophic wind but with no-slip at surface
    v = jnp.copy(vg)

    # Initial temperature
    th = jnp.ones_like(z) * 265.0  # K
    th = jnp.where(z > z_inv, th + 0.01 * (z - z_inv), th)  # capping inversion
    th = jnp.where(
        z < 50, th + 0.1 * jax.random.normal(key=jax.random.key(random_seed), shape=(Nz,)), th
    )  # random 0.1K perturbation near surface

    # Initial TKE
    tke = jnp.zeros_like(z)
    tke = jnp.where(z < 250, 0.4 * (1 - z / 250) ** 3, tke)  # m^2 s^-2

    # Surface model
    z0m = z0h = 0.1  # m, roughness lengths for momentum and heat
    beta_m = 4.8  # MOST momentum stability coefficent
    beta_h = 7.8  # MOST heat stability coefficient

    grid = {
        "z": z,  # Nz is len
        "zh": zh,  # H is zh[-1]
    }

    forcing = {
        "ug": ug,
        "vg": vg,
        "f": f,
        "th_sfc_0": th_sfc_0,
        "th_sfc_fn": th_sfc_fn,
    }

    init = {
        "u": u,
        "v": v,
        "th": th,
        "tke": tke,
    }

    lsm = {
        "z0m": z0m,
        "z0h": z0h,
        "beta_m": beta_m,
        "beta_h": beta_h,
    }

    if plot:
        # Initial conditions
        fig, axarr = plt.subplots(ncols=3, figsize=(8, 2), sharey="row", layout="constrained")
        axarr[0].plot(u, z, label="u")
        axarr[0].plot(v, z, label="v")
        axarr[0].set_xlabel("Wind (m/s)")
        axarr[0].set_ylabel("Height (m)")
        axarr[0].legend()

        axarr[1].plot(th, z)
        axarr[1].set_xlabel("Potential Temperature (K)")

        axarr[2].plot(tke, z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

        # Forcing
        fig, axarr = plt.subplots(ncols=2, figsize=(8, 2), width_ratios=[1, 3], layout="constrained")
        axarr[0].plot(ug, z, label="ug")
        axarr[0].plot(vg, z, label="vg")
        axarr[0].set_xlabel("Geostrophic Wind (m/s)")
        axarr[0].legend()

        t = jnp.array([0, 9 * 60 * 60])  # 0 and 9 hours
        axarr[1].plot(t, th_sfc_fn(t))
        axarr[1].set_xlabel("Time, s")
        axarr[1].set_ylabel("Surface Potential Temperature (K)")

        fig.show()

    return grid, init, forcing, lsm


def get_ysu(Nz: int = 138, plot: bool = False) -> Tuple[StaggeredGrid, ProgVars, TransientForcing]:
    """Initial conditions and forcing from HND06"""
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

    u = jnp.ones(grid.Nz) * 15.0  # m/s
    u = jnp.where(grid.z < z_inv, (15 / 500) * grid.z, u)  # linear increase to 15 m/s at z_inv

    v = jnp.zeros(grid.Nz)

    init = ProgVars(u=u, v=v, th=th, q=q)

    # Forcing
    @jax.jit
    def _shfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface heat flux as function of time in seconds after simulation begin."""
        t_h = t_s / 3600.0  # time in hours
        shfx = jnp.sin((t_h + 2) * jnp.pi / 12) * 400  # W/m2 = J/(s m2)
        shfx = shfx / (consts.rho_0 * consts.cp)  # convert to (K m/s)
        return shfx

    @jax.jit
    def _lhfx(t_s: jnp.ndarray) -> jnp.ndarray:
        """Surface latent heat flux as function of time in seconds after simulation begin."""
        t_h = t_s / 3600.0  # time in hours
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
        ax_th.plot(init.th, grid.z)
        ax_th.set_xlabel("Potential Temperature (K)")
        ax_q.plot(init.q * 1000, grid.z)
        ax_q.set_xlabel("Specific Humidity (g/kg)")
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
        w_th_s=lambda t: 0.0,  # neutral stratification
        w_q_s=lambda t: 0.0,
    )

    th = jnp.ones(grid.Nz) * 280.0
    th = jnp.where(grid.z > 400, th + 0.01 * (grid.z - 500), th)  # weak inversion above 500m

    init = ProgVars(
        u=ug.copy(),
        v=vg.copy(),
        th=th,
        q=jnp.zeros(grid.Nz),
    )

    if plot:
        fig, (ax_uv, ax_th) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
        ax_uv.plot(init.u, grid.z, label="u")
        ax_uv.plot(init.v, grid.z, label="v")
        ax_uv.set_xlabel("Wind (m/s)")
        ax_uv.set_ylabel("Height (m)")
        ax_uv.legend()
        ax_th.plot(init.th, grid.z)
        ax_th.set_xlabel("Potential Temperature (K)")
        fig.show()

    return grid, init, forcing


if __name__ == "__main__":
    # get_gables1(plot=True)
    # get_ekman(plot=True)
    get_ysu(plot=True)
