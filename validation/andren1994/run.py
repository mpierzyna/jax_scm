from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from PIL import Image
from scm import convert
from scm.config import load_namelist
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.io.local import out_to_ds
from scm.mo import MOSettings
from scm.mynn.interfaces import ProgVarsMYNN
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate
from scm import consts
from scm import convert


def get_a94(Nz: int = 40) -> Simulation:
    ## Grid
    grid = StaggeredGrid(Nz=Nz, H=1500)

    ## Forcing
    f_c = convert.get_fc(lat_deg=45)
    u_g = jnp.ones(Nz) * 10
    v_g = jnp.zeros(Nz)
    mo_settings = MOSettings(z0h=0.1, z0m=0.1)

    forcing = Forcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=f_c,
        w_th_s=lambda t_s: jnp.array(0.0),
        w_qv_s=lambda t_s: jnp.array(0.0),
        dth_dz_top=0.0,
    )

    ## Initial conditions
    df = pd.read_csv("ref/andren94_tab_A1.csv")
    u = jnp.interp(grid.z, df["z"].values, df["u"].values)
    v = jnp.interp(grid.z, df["z"].values, df["v"].values)
    qke = jnp.interp(grid.z, df["z"].values, df["tke"].values) * 2
    init = ProgVarsMYNN(
        u=u,
        v=v,
        th=jnp.ones(Nz) * 273.15,  # neutral
        qv=jnp.zeros(Nz),  # neutral
        qke=qke,
    )

    return Simulation(
        name="Andren1994",
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        grid=grid,
        th_ref=273.15,
        t_start_s=0,
        t_end_s=int(10 / f_c),
    )


def run():
    sim = get_a94()
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    ds.to_netcdf("andren1994.nc")


def make_report(ds: xr.Dataset):
    f = ds["frc_f_c"].item()
    tf = ds["time"] * f  # normalized time

    ## Fig 3: Steady-state deviation
    uw0 = ds["mo_u_w"]
    C_u = -f / uw0 * np.trapezoid(y=ds["v"] - ds["frc_v_geo"], x=ds["z"])

    vw0 = ds["mo_v_w"]
    C_v = f / vw0 * np.trapezoid(y=ds["u"] - ds["frc_u_geo"], x=ds["z"])

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
    ax1.plot(tf, C_u)
    ax1.set_xlabel("t f")
    ax1.set_ylabel("$C_u$")

    ax2.plot(tf, C_v)
    ax2.set_xlabel("t f")
    ax2.set_ylabel("$C_v$")
    fig.show()

    ## Fig 2: Integrated normalized TKE
    tke_int = np.trapezoid(y=ds["qke"] / 2, x=ds["z"])
    tke_norm = f / ds["mo_u_st"] ** 3 * tke_int

    fig, ax = plt.subplots()
    ax.plot(tf, tke_norm)
    ax.set_xlabel("t f")
    ax.set_ylabel("f / u_*^3 * int(tke dz)")
    fig.show()

    ## Subset for statistics
    ds_sub = ds.sel(time=slice(7 / f, None))  # "last 3/f are used for statistics" (Andren 1994)
    ds_sub = ds_sub.mean("time")

    ## Fig 6: Momentum flux profiles
    zh_norm = ds_sub["zh"] * f / ds_sub["mo_u_st"]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3), sharey="row")
    ax1.scatter(ds_sub["u_w"] / ds_sub["mo_u_st"] ** 2, zh_norm)
    ax2.scatter(ds_sub["v_w"] / ds_sub["mo_u_st"] ** 2, zh_norm)
    fig.show()

    ## Fig 4: SL gradients
    du_dz = np.gradient(ds_sub["u"], ds_sub["z"])
    dv_dz = np.gradient(ds_sub["v"], ds_sub["z"])
    phi_m = consts.kappa * ds_sub["z"] / ds_sub["mo_u_st"] * np.sqrt(du_dz**2 + dv_dz**2)
    z_norm = ds_sub["z"] * f / ds_sub["mo_u_st"]

    fig, ax = plt.subplots()
    ax.scatter(phi_m, z_norm)
    ax.set_ylim(0, 0.1)
    ax.axvline(1, color="k", ls="--")
    fig.show()

    return


if __name__ == "__main__":
    # run()
    ds = xr.open_dataset("andren1994.nc")
    make_report(ds)
