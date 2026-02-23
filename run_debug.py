from __future__ import annotations

import logging

import jax
import numpy as np

import cases
from scm.config import Namelist
from scm.forcing.era5 import get_era5_sim  # noqa
from scm.forcing.interp import interp_dtindex
from scm.io.local import make_dataset
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
    sim = cases.get_wangara(Nz=200)

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

    cfg_ab2 = Namelist(
        time_int="explicit",
        dt_s=0.001,
        dt_s_out=300.0,
        adaptive_timestep=dict(cfl_max=0.05, dt_s_max=1.0),
    )

    cfg_cn = Namelist(
        time_int="implicit",
        dt_s=0.5,
        dt_s_out=300.0,
    )
    cfg = cfg_ab2

    # Init and run model
    model = init_model(sim, implicit=cfg.is_implicit)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Prepare time axis
    if sim.t_index is not None:
        time = interp_dtindex(t_s=np.array(t), idx=sim.t_index)
    else:
        time = t / 3600.0  # convert to hours

    # Save output
    ds = make_dataset(out=out, sim=sim, time=out.t_s / 60 / 60)
    ds.to_netcdf("out.nc")
    print("Written to disk.")
