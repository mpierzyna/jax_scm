from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from scm import consts
from scm.forcing import convert
from scm.forcing.interp import get_ts_interp_fn, xr_interp_vert
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, TransientForcing
from scm.io import era5
from scm.io.cache import XRCache
from scm.mynn.interfaces import ProgVarsMYNN


def get_era5_sim(
    name: str,
    lat_deg: float,
    lon_deg: float,
    time_slice: str | slice,
    grid: StaggeredGrid,
    cache_dir: str | None = ".era5_cache",
) -> Simulation:
    """Get a Simulation object with ERA5 forcing data."""
    # Set up data download function with optional caching
    if cache_dir is not None:
        xr_cache = XRCache(cache_dir)
        download_data = xr_cache.cache(era5.download_data)
    else:
        download_data = era5.download_data

    # Download ERA5 data
    ds = download_data(lat_deg, lon_deg, time_slice)
    ds = ds.rename_dims(isobaricInhPa="bottom_top")  # more telling coordinate names

    # Compute geostrophic wind
    ds_uv_geo = convert.uv_geo_from_z(lat_deg=ds["latitude"], lon_deg=ds["longitude"], z=ds["z"])
    ds = ds.merge(ds_uv_geo, compat="override")

    # Compute potential temperature
    p_hPa = ds["isobaricInhPa"]
    ds["th"] = convert.th_from_tk(t_k=ds["t"], p_hPa=p_hPa)

    # We don't need neighbours anymore
    ds = ds.sel(latitude=lat_deg, longitude=lon_deg, method="nearest")

    # Geopotential to height
    z = ds["z"] / 9.81  # in m
    z_target = xr.DataArray(grid.z, dims=["bottom_top"])

    # Interpolate ERA5 to grid levels
    # As ERA5 has very few levels near surface, we fill missing values with constants after interpolation
    # Disable jax nan checking temporarily because it will otherwise raise exception
    with jax.debug_nans(False):
        # Interpolate on log(z)
        # todo: this should be agl!
        ds_interp = xr_interp_vert(ds, z=np.log(z), z_target=np.log(z_target), dim="bottom_top")
        ds_interp = ds_interp.interpolate_na(dim="bottom_top", method="nearest", fill_value="extrapolate")

    # Create time coordinate in s
    t = ds["valid_time"]
    t = t - t[0]
    t = t.astype("timedelta64[s]").astype(int)
    t_start_s, t_end_s = t[[0, -1]].values

    # Create forcing functions for geostrophic wind
    u_geo_fn = get_ts_interp_fn(time_s=jnp.array(t.values), data=jnp.array(ds_interp["ug"].values))
    v_geo_fn = get_ts_interp_fn(time_s=jnp.array(t.values), data=jnp.array(ds_interp["vg"].values))

    # Create surface temperature forcing function
    t_s_fn = get_ts_interp_fn(
        time_s=jnp.array(t.values),
        data=jnp.array(ds["skt"].values),  # todo: convert to pot temp?
    )

    # Create moisture flux forcing
    rho = ds["sp"] / (consts.Rd * ds["skt"])  # todo: acc to Gemini, virtual skin temp should be used
    w_qv = -ds["ie"] / rho
    w_qv_s_fn = get_ts_interp_fn(time_s=jnp.array(t.values), data=jnp.array(w_qv.values))

    # Coriolis parameter
    f_c = 2 * 7.2921e-5 * jnp.sin(jnp.deg2rad(lat_deg))

    # Gather forcing in TransientForcing
    frc = TransientForcing(
        u_geo=u_geo_fn,
        v_geo=v_geo_fn,
        th_s=t_s_fn,
        w_qv_s=w_qv_s_fn,
        f_c=float(f_c),  # coriolis parameter
    )

    # Create initial conditions
    ds_era5_init = ds_interp.isel(valid_time=0)
    init = ProgVarsMYNN(
        u=jnp.array(ds_era5_init["u"].values),
        v=jnp.array(ds_era5_init["v"].values),
        th=jnp.array(ds_era5_init["th"].values),
        qv=jnp.array(ds_era5_init["q"].values),
        qke=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    # Create simulation object
    sim = Simulation(
        name=name,
        grid=grid,
        init=init,
        forcing=frc,
        t_start_s=int(t_start_s),
        t_end_s=int(t_end_s),
        t_index=ds.indexes["valid_time"],
    )

    return sim


if __name__ == "__main__":

    # jax.config.update("jax_disable_jit", True)

    sim = get_era5_sim(
        name="ERA5 Test Simulation",
        lat_deg=52.0,
        lon_deg=4.0,
        time_slice="2020-01-01",
        grid=StaggeredGrid(Nz=100, H=2000.0),
    )
    print(sim)
