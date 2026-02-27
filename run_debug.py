from __future__ import annotations

import logging

import jax
import numpy as np

import cases
from scm.config import Namelist, AdaptiveTimestepConfig, TimeIntMethod
from scm.forcing.era5 import get_era5_sim  # noqa
from scm.grid import StaggeredGrid
from scm.forcing.interp import interp_dtindex
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scm")

if __name__ == "__main__":
    # Ekman spiral
    # sim = cases.get_ekman(Nz=100)

    # YSU test case
    # sim = cases.get_ysu()

    # GABLS
    # sim = cases.get_gabls1(Nz=64)

    # Wangara
    # sim = cases.get_wangara(Nz=200)

    # Cabauw from ERA5
    # sim = get_era5_sim(
    #     name="Cabauw_Test",
    #     lat_deg=52.0,
    #     lon_deg=5.0,
    #     grid=StaggeredGrid(Nz=200, H=4000.0),
    #     # time_slice=slice("2025-07-01", "2025-07-03"),
    #     time_slice="2025-07-01",
    #     source="google",
    # )
    sim = get_era5_sim(
        name="ERA5 Test Simulation",
        lat_deg=52.0,
        lon_deg=5.0,
        time_slice=("2006-07-01T12:00", "2006-07-02T12:00"),
        grid=StaggeredGrid(Nz=150, H=3000.0),
        source="cds",
    )

    cfg_ab2 = Namelist(
        time_int=TimeIntMethod.EXPLICIT,
        dt_s=0.001,
        dt_s_out=300.0,
        adaptive_timestep=AdaptiveTimestepConfig(cfl_max=0.1, dt_s_max=10.0),
    )

    cfg_cn = Namelist(
        time_int=TimeIntMethod.IMPLICIT,
        dt_s=2,
        dt_s_out=300.0,
    )
    cfg = cfg_cn

    # Init and run model
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Prepare time axis
    if sim.t_index is not None:
        time = interp_dtindex(t_s=np.array(out.t_s), idx=sim.t_index)
    else:
        time = out.t_s / 3600.0  # convert to hours

    # Save output
    ds = out_to_ds(out=out, sim=sim, time=time)
    ds.to_netcdf("out.nc")
    print("Written to disk.")
