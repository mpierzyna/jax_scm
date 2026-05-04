from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from jax import numpy as jnp

from scm import convert, consts
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.mo import MOSettings
from scm.mynn.interfaces import MYNNParams, ProgVarsMYNN


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
    u_st_est = np.sqrt(u[0] ** 2 + v[0] ** 2) * consts.kappa / np.log(grid.z[0] / mo_settings.z0m)
    qke_sfc_est = params.B1 ** (2 / 3) * u_st_est**2
    qke_init = np.maximum(qke_sfc_est * np.exp(-grid.z / 100.0), consts.qke_min)

    init = ProgVarsMYNN(
        u=jnp.array(u),
        v=jnp.array(v),
        th=jnp.array(th),
        qv=jnp.array(qv),
        qke=jnp.array(qke_init),
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
