from __future__ import annotations

import datetime
import pathlib
from typing import Literal, Dict, Tuple

import jax
import jax.numpy as jnp
import ls2d
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

from scm import consts, convert
from scm.forcing.interp import get_ts_interp_fn
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.io import era5
from scm.io.cache import XRCache
from scm.mo import MOSettings
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


def _get_sim_from_ls2d(
    name: str,
    start: datetime.date,
    end: datetime.date,
    lat_deg: float,
    lon_deg: float,
    th_ref: float,
    grid: StaggeredGrid,
    cache_dir: pathlib.Path,
) -> Simulation:

    cache_dir = cache_dir / "ls2d"
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "central_lat": lat_deg,
        "central_lon": lon_deg,
        "area_size": 1,
        "case_name": "cabauw",
        "era5_path": str(cache_dir),
        "era5_expver": 1,  # 1=normal ERA5, 5=ERA5 near-realtime
        "start_date": start,
        "end_date": end,
        "write_log": False,
        "data_source": "CDS",
    }

    # will sys.exit until download done
    ls2d.download_era5(settings, exit_when_waiting=True)

    # Read ERA5 data, and calculate derived properties (thl, etc.):
    era5_reader = ls2d.Read_era5(settings)

    # Calculate large-scale forcings:
    # `n_av` is the number of ERA5 gridpoints (+/-) over which
    # the ERA5 variables and forcings are averaged.
    era5_reader.calculate_forcings(n_av=0, method="2nd")  # todo: not sure what to set

    # Interpolate ERA5 to fixed height grid:
    ds = era5_reader.get_les_input(z=grid.z)

    # Create time coordinate in s
    t = ds["time_sec"].values
    t_start_s, t_end_s = t[0], t[-1]

    # Create forcing functions for geostrophic wind
    u_geo = jnp.array(ds["ug"].values)
    v_geo = jnp.array(ds["vg"].values)
    u_geo_fn = get_ts_interp_fn(time_s=jnp.array(t), data=u_geo)
    v_geo_fn = get_ts_interp_fn(time_s=jnp.array(t), data=v_geo)

    # Create surface temperature forcing function
    t_s_fn = get_ts_interp_fn(
        time_s=jnp.array(t),
        data=jnp.array(ds["ts"].values),
    )

    # Create moisture flux forcing
    w_qv_s_fn = get_ts_interp_fn(time_s=jnp.array(t), data=jnp.array(ds["wq"].values))

    # Coriolis parameter
    f_c = float(ds.attrs["fc"])

    # Extract large-scale advective tendencies and large-scale vertical  velocity (subsidence).
    # LS2D provides horizontal advective tendencies and wls separately;
    # subsidence (-wls * d/dz) is computed at run-time.
    th_adv = jnp.array(ds["dtthl_advec"].values)  # (time, z), K s-1
    qv_adv = jnp.array(ds["dtqt_advec"].values)  # (time, z), kg kg-1 s-1
    u_adv = jnp.array(ds["dtu_advec"].values)  # (time, z), m s-2
    v_adv = jnp.array(ds["dtv_advec"].values)  # (time, z), m s-2
    wls = jnp.array(ds["wls"].values)  # (time, z), m s-1

    th_adv_fn = get_ts_interp_fn(time_s=jnp.array(t), data=th_adv)
    qv_adv_fn = get_ts_interp_fn(time_s=jnp.array(t), data=qv_adv)
    u_adv_fn = get_ts_interp_fn(time_s=jnp.array(t), data=u_adv)
    v_adv_fn = get_ts_interp_fn(time_s=jnp.array(t), data=v_adv)
    wls_fn = get_ts_interp_fn(time_s=jnp.array(t), data=wls)

    def ls_tends_fn(t_s: jnp.ndarray, state: ProgVarsMYNN, grads: ProgVarsMYNN, _) -> ProgVarsMYNN:
        """Large-scale tendencies: horizontal advection + subsidence.
        - LS2D provides horizontal advective tendencies directly.
        - Subsidence is computed as ``-wls * d(phi)/dz`` where ``grads`` holds vertical gradients at half levels,
          averaged to full levels before multiplication.
        """
        w = wls_fn(t_s)  # large-scale vertical velocity, (Nz,)

        # Average half-level gradients to full levels for subsidence
        du_dz = (grads.u[1:] + grads.u[:-1]) / 2
        dv_dz = (grads.v[1:] + grads.v[:-1]) / 2
        dth_dz = (grads.th[1:] + grads.th[:-1]) / 2
        dqv_dz = (grads.qv[1:] + grads.qv[:-1]) / 2

        return ProgVarsMYNN(
            u=u_adv_fn(t_s) - w * du_dz,
            v=v_adv_fn(t_s) - w * dv_dz,
            th=th_adv_fn(t_s) - w * dth_dz,
            qv=qv_adv_fn(t_s) - w * dqv_dz,
            qke=jnp.zeros_like(state.qke),  # no large-scale TKE forcing
        )

    # Gather forcing in Forcing object
    frc = Forcing(
        u_geo=u_geo_fn,
        v_geo=v_geo_fn,
        th_s=t_s_fn,
        w_qv_s=w_qv_s_fn,
        f_c=f_c,
        ls_tends=ls_tends_fn,
    )

    # Create initial conditions
    ds_init = ds.isel(time=0)
    init = ProgVarsMYNN(
        u=jnp.array(ds_init["u"].values),
        v=jnp.array(ds_init["v"].values),
        th=jnp.array(ds_init["thl"].values),  # note: LS2D uses thl (liquid water potential temperature)
        qv=jnp.array(ds_init["qt"].values),  # note: LS2D uses qt (total specific humidity)
        qke=jnp.ones(grid.Nz) * 0.01,  # small initial TKE
    )

    # Get roughness lengths (use time-averaged values or initial values)
    z0m = ds["z0m"].mean("time").item()  # todo: implement time varying
    z0h = ds["z0h"].mean("time").item()  # todo: implement time varying

    # Create simulation object
    sim = Simulation(
        name=name,
        grid=grid,
        init=init,
        forcing=frc,
        mo_settings=MOSettings(z0h=z0h, z0m=z0m),
        th_ref=th_ref,
        t_start_s=int(t_start_s),
        t_end_s=int(t_end_s),
        t_index=ds.indexes["time"],
    )

    return sim


def get_era5_sim(
    name: str,
    lat_deg: float,
    lon_deg: float,
    time_slice: str | datetime.date | Tuple[str, str] | Tuple[datetime.datetime, datetime.datetime],
    grid: StaggeredGrid,
    th_ref: float,
    source: Literal["destine", "google", "cds"],
    cache_dir: str | None = ".era5_cache",
) -> Simulation:
    """Get a Simulation object with ERA5 forcing data.

    Parameters
    ----------
    name : str
        Name of the simulation
    lat_deg : float
        Latitude in degrees
    lon_deg : float
        Longitude in degrees
    time_slice : str | datetime.date | Tuple[str, str] | Tuple[datetime.datetime, datetime.datetime]
        Time slice for the simulation. For "cds" source, must be a tuple of (start, end).
    grid : StaggeredGrid
        Vertical grid for the simulation
    source : Literal["destine", "google", "cds"]
        Data source. "cds" uses the LS2D downloader which downloads ERA5 from CDS.
    cache_dir : str | None
        Directory for caching downloaded data. Default is ".era5_cache".

    Returns
    -------
    Simulation

    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "cds":
        # Use LS2D implementation here!
        try:
            start, end = time_slice
        except TypeError:
            raise ValueError("For LS2D, time_slice must be a tuple of (start, end) datetimes or strings.")

        if isinstance(start, str):
            start = datetime.datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.datetime.fromisoformat(end)

        return _get_sim_from_ls2d(
            name=name,
            start=start,
            end=end,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            th_ref=th_ref,
            grid=grid,
            cache_dir=cache_dir,
        )
    else:
        # Zarr-based sources (Destine, Google)
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
    frc = Forcing(
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
        th_ref=th_ref,
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
        time_slice=("2006-06-30T12:00", "2006-07-03T00:00"),
        grid=StaggeredGrid(Nz=100, H=1000.0),
        source="cds",
    )
    print(sim)
