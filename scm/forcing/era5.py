from __future__ import annotations
from typing import Literal, Dict

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

from scm import consts, convert
from scm.forcing.interp import get_ts_interp_fn
from scm.mo import MOSettings
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, TransientForcing
from scm.io import era5
from scm.io.cache import XRCache
from scm.mynn.interfaces import ProgVarsMYNN


def interp(
    y: xr.DataArray,
    z: xr.DataArray,
    z_target: xr.DataArray,
    dim: str,
    dim_out: str = "grid",
    extra: Dict[int, xr.DataArray] | None = None,
) -> xr.DataArray:
    """Interpolate y(z) to new vertical levels z_target using cubic spline interpolation.

    Parameters
    ----------
    y : xr.DataArray
        DataArray with vertical profiles to interpolate.
    z : xr.DataArray
        DataArray with vertical levels corresponding to y.
    z_target : xr.DataArray
        DataArray with target vertical levels for interpolation.
    dim : str
        Name of the vertical dimension in y and z.
    dim_out : str, optional
        Name of the vertical dimension in the output DataArray, by default "grid".
    extra: Dict[int, xr.DataArray] | None
        Extra points to include in the interpolation.
        Keys are the z values, values are the corresponding y values.
    """

    def _interp_cubic(y: np.ndarray, z: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        """Interpolate profile y(z) to new vertical levels z_target."""
        z_sorting = np.argsort(z)
        spl = CubicSpline(z[z_sorting], y[z_sorting], extrapolate=False)
        y_ = spl(z_target)
        return y_

    if extra is not None:
        # concatenate extra levels
        z = xr.concat([z, xr.DataArray(list(extra.keys()), dims=[dim])], dim=dim)

        # concatenate extra points
        y = xr.concat([y, *list(extra.values())], dim=dim)

    y = xr.apply_ufunc(
        _interp_cubic,
        y,
        z,
        z_target,
        input_core_dims=[[dim], [dim], [dim_out]],
        output_core_dims=[[dim_out]],
        vectorize=True,
    )

    # Eliminate NaNs by nearest neighbour interpolation
    y = y.interpolate_na(dim=dim_out, method="nearest", fill_value="extrapolate")

    return y


def get_era5_sim(
    name: str,
    lat_deg: float,
    lon_deg: float,
    time_slice: str | slice,
    grid: StaggeredGrid,
    source: Literal["destine", "google"],
    cache_dir: str | None = ".era5_cache",
) -> Simulation:
    """Get a Simulation object with ERA5 forcing data."""
    # Set up data download function with optional caching
    if cache_dir is not None:
        xr_cache = XRCache(cache_dir)
        download_data = xr_cache.cache(era5.download_data)
    else:
        download_data = era5.download_data

    if source == "destine":
        # I need to make some modifications to adjust coordinates
        # Also below, we have hard coded inversion of level dim for Google version
        raise NotImplementedError

    # Download ERA5 data
    ds = download_data(lat_deg, lon_deg, time_slice, source)
    p_hPa = ds["level"].drop_vars("level")  # save pressure
    ds = ds.drop_vars("level")  # drop because it causes problems for interpolation

    # Compute geostrophic wind
    ds_uv_geo = convert.uv_geo_from_z(lat_deg=ds["latitude"], lon_deg=ds["longitude"], z=ds["z"])
    ds = ds.merge(ds_uv_geo, compat="override")

    # Compute potential temperature
    ds["th"] = convert.tk_to_th(tk=ds["t"], p_hPa=p_hPa)

    # We don't need neighbours anymore
    ds = ds.sel(latitude=lat_deg, longitude=lon_deg, method="nearest")

    # Geopotential to height
    z_agl = (ds["z"] - ds["z_sfc"]) / 9.81  # in m
    z_target = xr.DataArray(grid.z, dims=["grid"])

    # Interpolate ERA5 to grid levels
    # As ERA5 has very few levels near surface, we add surface values as extra points
    # Disable jax nan checking temporarily because it will otherwise raise exception
    with jax.debug_nans(False):
        uv0 = xr.DataArray(np.zeros(ds.sizes["time"]), dims=["time"], coords=ds["u10"].coords)
        u = interp(ds["u"], z_agl, z_target, dim="level", extra={10: ds["u10"], 0: uv0})
        v = interp(ds["v"], z_agl, z_target, dim="level", extra={10: ds["v10"], 0: uv0})
        th = interp(ds["th"], z_agl, z_target, dim="level", extra={2: ds["t2m"], 0: ds["skt"]})  # todo: to pot temp

        # No extra points, so extrpolation automatically used in `interp`
        qv = interp(ds["q"], z_agl, z_target, dim="level")
        u_geo = interp(ds["ug"], z_agl, z_target, dim="level")
        v_geo = interp(ds["vg"], z_agl, z_target, dim="level")

    # Create time coordinate in s
    t = ds["time"]
    t = t - t[0]
    t = t.astype("timedelta64[s]").astype(int)
    t_start_s, t_end_s = t[[0, -1]].values

    # Create forcing functions for geostrophic wind
    u_geo_fn = get_ts_interp_fn(time_s=jnp.array(t.values), data=jnp.array(u_geo.values))
    v_geo_fn = get_ts_interp_fn(time_s=jnp.array(t.values), data=jnp.array(v_geo.values))

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
    init = ProgVarsMYNN(
        u=jnp.array(u.isel(time=0).values),
        v=jnp.array(v.isel(time=0).values),
        th=jnp.array(th.isel(time=0).values),
        qv=jnp.array(qv.isel(time=0).values),
        qke=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    # Create simulation object
    sim = Simulation(
        name=name,
        grid=grid,
        init=init,
        forcing=frc,
        mo_settings=MOSettings(z0h=0.1, z0m=0.1),
        t_start_s=int(t_start_s),
        t_end_s=int(t_end_s),
        t_index=ds.indexes["time"],
    )

    return sim


if __name__ == "__main__":

    # jax.config.update("jax_disable_jit", True)

    sim = get_era5_sim(
        name="ERA5 Test Simulation",
        lat_deg=52.0,
        lon_deg=5.0,
        time_slice="2020-01-01",
        grid=StaggeredGrid(Nz=100, H=1000.0),
        source="google",
    )
    print(sim)
