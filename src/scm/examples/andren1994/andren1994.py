import pandas as pd
from jax import numpy as jnp
import pathlib

from scm import convert
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
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
    )
