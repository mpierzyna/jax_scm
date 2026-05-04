import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import Forcing, Simulation
from scm.mo import MOSettings
from scm.mynn.interfaces import ProgVarsMYNN


def get_andren1994(Nz: int = 40) -> Simulation:
    grid = StaggeredGrid(Nz=Nz, H=1500)

    # f_c = convert.get_fc(lat_deg=45)
    f_c = 1e-4
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

    df = pd.read_csv(pathlib.Path(__file__).parent / "andren1994_tab_A1.csv")
    u = jnp.interp(grid.z, df["z"].values, df["u"].values)
    v = jnp.interp(grid.z, df["z"].values, df["v"].values)
    qke = jnp.interp(grid.z, df["z"].values, df["tke"].values) * 2
    init = ProgVarsMYNN(
        u=u,
        v=v,
        th=jnp.ones(Nz) * 273.15,
        qv=jnp.zeros(Nz),
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
        t_index_fn=lambda t_s: t_s * f_c,  # inertial periods
    )


def postproc_andren1994(ds: xr.Dataset) -> xr.Dataset:
    """Diagnosed values for Andren 1994 validation"""
    # Normalized vertically integrated TKE
    f = ds["frc_f_c"].item()
    tke_int = np.trapezoid(y=ds["qke"] / 2, x=ds["z"])
    tke_int_norm = f / ds["mo_u_st"] ** 3 * tke_int

    # C_u and C_v deviation from steady state
    uw0 = ds["mo_u_w"]
    C_u = -f / uw0 * np.trapezoid(y=ds["v"] - ds["frc_v_geo"], x=ds["z"])

    vw0 = ds["mo_v_w"]
    C_v = f / vw0 * np.trapezoid(y=ds["u"] - ds["frc_u_geo"], x=ds["z"])

    # Time-averaged statistics over last 3/f (Andren 1994)
    ds_sub = ds.sel(time=slice(7.0, None)).mean("time")  # time is in normalized units, so just avg 7 to 10
    u_st = float(ds_sub["mo_u_st"])

    # Surface layer phi_m gradient function
    mo_settings = MOSettings.deserialize(ds.attrs["mo_settings"])  # deserialize for z0m

    # Add u=v=0 at roughness height for better gradient calculation near the surface
    u = ds_sub["u"].values
    u = np.insert(u, 0, 0)
    v = ds_sub["v"].values
    v = np.insert(v, 0, 0)
    z = ds_sub["z"].values
    z = np.insert(z, 0, mo_settings.z0h)
    zh = np.diff(z) / np.log(z[1:] / z[:-1])  # log-mean height: correct for log-law profiles near surface

    dz = np.diff(z)
    du_dz = (u[1:] - u[:-1]) / dz
    dv_dz = (v[1:] - v[:-1]) / dz
    phi_m = consts.kappa * zh / u_st * np.sqrt(du_dz**2 + dv_dz**2)

    # assign normalized height, different than momentum fluxes below because of log-mean height calculation
    phi_m = xr.DataArray(phi_m, coords={"zh_phi_": zh * f / u_st}, dims=["zh_phi_"])

    # Normalized momentum flux profiles
    zh_norm = (ds_sub["zh"] * f / u_st).values
    uw_norm = ds_sub["u_w"] / u_st**2
    uw_norm = xr.DataArray(uw_norm.values, coords={"zh_": zh_norm}, dims=["zh_"])
    vw_norm = ds_sub["v_w"] / u_st**2
    vw_norm = xr.DataArray(vw_norm.values, coords={"zh_": zh_norm}, dims=["zh_"])

    return xr.Dataset(
        {
            "tke_int_norm": tke_int_norm,
            "C_u": C_u,
            "C_v": C_v,
            "phi_m": phi_m,
            "uw_norm": uw_norm,
            "vw_norm": vw_norm,
        }
    )
