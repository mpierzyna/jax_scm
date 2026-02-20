import jax.numpy as jnp
import matplotlib.pyplot as plt

from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.io.local import make_dataset
from scm.mo import MOSettings, BusingerDyerAltSimFuncs
from scm.mynn.interfaces import ProgVarsMYNN, DiagVarsMYNN
from scm.mynn.model import init_model
from scm.time_stepping import simulate
from scm.config import load_namelist


def get_gabls1(Nz: int = 64, plot: bool = False, random_seed: int = 0) -> Simulation[ProgVarsMYNN, DiagVarsMYNN]:
    """Get a GABLS1 simulation setup.

    References
    ----------
    Cuxart, J., et al. “Single-Column Model Intercomparison for a Stably Stratified Atmospheric Boundary Layer.”
    Boundary-Layer Meteorology, vol. 118, no. 2, Feb. 2006, pp. 273–303.
    https://doi.org/10.1007/s10546-005-3780-1.

    """
    ## Grid
    grid = StaggeredGrid(H=400, Nz=Nz)
    z_inv = 100

    ## Forcing
    # Geostrophic wind
    ug = jnp.ones(Nz) * 8.0  # m/s
    vg = jnp.zeros(Nz)  # m/s

    # Surface temperature forcing
    th_s_0 = 265  # K
    th_s_fn = lambda t_s: th_s_0 - 0.25 * t_s / (60 * 60)  # K, 0.25 K per hour cooling

    # No moisture
    w_qv_s = lambda t_s: jnp.array(0.0)  # g/kg m/s

    forcing = Forcing(
        u_geo=lambda t_s: ug,
        v_geo=lambda t_s: vg,
        f_c=1.39e-4,  # 1/s, ~73 deg latitude
        th_s=th_s_fn,
        w_qv_s=w_qv_s,
    )

    # MO settings
    mo_settings = MOSettings(
        z0m=0.1,
        z0h=0.1,
        sim_funcs=BusingerDyerAltSimFuncs(gamma_m=16, gamma_h=16, b_m=4.8, b_h=7.8),
    )

    ## Initial conditions
    # Initial wind profile
    u = jnp.copy(ug)
    v = jnp.copy(vg)

    # Initial temperature
    th = jnp.ones(Nz) * 265.0  # K
    th = jnp.where(grid.z > z_inv, th + 0.01 * (grid.z - z_inv), th)  # capping inversion
    # th = jnp.where(
    #     grid.z < 50, th + 0.1 * jax.random.normal(key=jax.random.key(random_seed), shape=(Nz,)), th
    # )  # random 0.1K perturbation near surface

    # No moisture
    qv = jnp.zeros(grid.Nz)

    # Initial TKE
    tke = jnp.zeros(grid.Nz)
    tke = jnp.where(grid.z < 250, 0.4 * (1 - grid.z / 250) ** 3, tke)  # m^2 s^-2

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

    return Simulation(
        name="GABLS1",
        grid=grid,
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        t_start_s=0,
        t_end_s=9 * 60 * 60,
    )


if __name__ == "__main__":
    cfg = load_namelist("namelist_cn.yaml")
    sim = get_gabls1(Nz=64, plot=False)
    model = init_model(sim, implicit=True)
    state_hist, diag_hist, mo_hist, t = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    ds = make_dataset(state_hist, diag_hist, mo_hist, time=t / 60 / 60, grid=sim.grid)
    ds.to_netcdf(f"out_{sim.grid.Nz}.nc")
    print("Written to disk.")
