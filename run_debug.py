from __future__ import annotations

import logging

import jax.experimental.checkify
import numpy as np

import cases
from scm.forcing.era5 import get_era5_sim  # noqa
from scm.forcing.interp import interp_dtindex
from scm.io.local import make_dataset
from scm.mynn.model import init_model
from scm.time_stepping import simulate_adaptive_dt

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("scm")

if __name__ == "__main__":
    # Ekman spiral
    # grid, init, forcing = cases.get_ekman(Nz=100)

    # YSU test case
    # t_debug = 0
    # grid, init, forcing = cases.get_ysu()

    # GABLS
    # sim = cases.get_gabls1(Nz=64)

    # Wangara
    sim = cases.get_wangara(Nz=50)

    # Cabauw from ERA5
    # sim = get_era5_sim(
    #     name="Cabauw_Test",
    #     lat_deg=52.0,
    #     lon_deg=5.0,
    #     grid=StaggeredGrid(Nz=300, H=3000.0),
    #     # time_slice=slice("2025-07-01", "2025-07-03"),
    #     time_slice="2025-07-01",
    # )

    # Init and run model
    model = init_model(sim)
    # state_hist, diag_hist, mo_hist, t = simulate(
    #     model,
    #     init,
    #     forcing,
    #     dt_s=0.001,
    #     t_start_s=9 * 60 * 60,
    #     t_end_s=16 * 60 * 60,
    #     dt_out_s=60 * 5,
    # )
    state_hist, diag_hist, mo_hist, t = simulate_adaptive_dt(
        model=model,
        sim=sim,
        dt_s_init=0.001,
        dt_s_max=1,
        cfl_max=0.1,
        dt_s_out=60 * 5,
    )

    # Prepare time axis
    if sim.t_index is not None:
        time = interp_dtindex(t_s=np.array(t), idx=sim.t_index)
    else:
        time = t / 3600.0  # convert to hours

    # Save output
    ds = make_dataset(state_hist, diag_hist, mo_hist, time=time, grid=sim.grid)
    ds.to_netcdf("out.nc")
    print("Written to disk.")
