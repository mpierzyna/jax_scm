from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp

from scm import consts, convert
from scm.grid import StaggeredGrid
from scm.interfaces import Forcing, Simulation
from scm.mo import MOSettings
from scm.mynn.closure import MYNNParams
from scm.mynn.interfaces import ProgVarsMYNN

TIMES = ["09:00", "10:00", "12:00", "14:00", "16:00"]
_T_LONG = [f"1967-08-16T{t}" for t in TIMES]
_T_1400 = "1967-08-16T14:00"


def get_wangara_day33(Nz: int = 50) -> Simulation:
    ## Grid
    grid = StaggeredGrid(H=2000, Nz=Nz)

    ## Surface and model parameters — defined early so they can be reused below
    mo_settings = MOSettings(z0m=0.01, z0h=0.01)  # todo: get good numbers from Hicks (1981)
    params = MYNNParams()

    ## Initial conditions
    df = pd.read_csv(pathlib.Path(__file__).parent / "day33_0900.csv")
    tk = df["tc"] + 273.15  # Convert to K
    p_hPa = df["p"]
    th = convert.tk_to_th(tk=tk, p_hPa=p_hPa)
    th = np.interp(grid.z, df["z"], th)  # Interpolate to model grid

    r = df["r"] / 1000  # kg/kg
    qv = r / (1 + r)
    qv = np.interp(grid.z, df["z"], qv)

    u = np.interp(grid.z, df["z"], df["u"])
    v = np.interp(grid.z, df["z"], df["v"])

    # Estimate surface TKE from initial wind via neutral log-law, then decay exponentially.
    u_st_est = 0.175  # m/s from obs (Hicks et al 1981)
    qke_sfc_est = params.B1 ** (2 / 3) * u_st_est**2
    H_est = 100  # estimated BLH at 9:00
    qke_init = jnp.zeros(grid.Nz)
    qke_init = jnp.where(grid.z < H_est, qke_sfc_est * (1 - grid.z / H_est) ** 3, qke_init)  # adapted from GABLS1

    init = ProgVarsMYNN(
        u=jnp.array(u),
        v=jnp.array(v),
        th=jnp.array(th),
        qv=jnp.array(qv),
        qke=qke_init,
    )

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

    forcing = Forcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=convert.get_fc(lat_deg=np.abs(-34.5)),  # 34.5°S latitude
        w_th_s=w_thl_fn,
        w_qv_s=w_qw_fn,
        dth_dz_top=dthl_dz_top,
    )

    ## Simulation
    sim = Simulation(
        name="Wangara_Day33",
        grid=grid,
        init=init,
        forcing=forcing,
        th_ref=277.0,  # Pot. temp. close to surface from soundings
        mo_settings=mo_settings,
        t_start_s=9 * 3600,  # 9:00 local time
        t_end_s=16 * 3600,  # 16:00 local time
        t_index_fn=lambda t_s: pd.to_datetime("1967-08-16") + pd.to_timedelta(t_s, unit="s"),
    )
    return sim


def postproc_wangara(ds: xr.Dataset) -> xr.Dataset:
    """Get Wangara Day 33 diagnostics for validation plots."""
    # Inversion height as height of minimum sensible heatflux
    zi = ds["zh"].isel(zh=ds["w_th"].argmin("zh"))  # tab 1 caption

    # Heat flux at zi, normalized by surface flux
    R = ds["w_th"].sel(zh=zi) / ds["mo_w_th"]  # m, tab 1 caption

    # Convective velocity scale with BUOYANCY flux
    w_st = (consts.g / ds.attrs["th_ref"] * ds["mo_w_thv"] * zi) ** (1 / 3)  # m/s, tab 1 caption

    # Normalized TKE budget. Divide by 2 for QKE to TKE.
    tke_scale = w_st**3 / zi
    tke_P_S = ds["qke_P_S"] / tke_scale / 2
    tke_P_B = ds["qke_P_B"] / tke_scale / 2
    tke_eps = ds["qke_eps"] / tke_scale / 2

    # Transport term: divergence of tke flux, normalized by tke_scale.
    div_w_tke = ds["w_qke"].diff("zh") / ds["zh"].diff("zh")
    div_w_tke = div_w_tke / tke_scale / 2

    ds_pp = xr.Dataset(
        {
            "zi": zi,
            "R": R,
            "w_st": w_st,
            "tke_P_S": tke_P_S,
            "tke_P_B": tke_P_B,
            "tke_eps": tke_eps,
            "div_w_tke": div_w_tke,
        }
    )
    return ds_pp
