from __future__ import annotations

import numpy as np
import xarray as xr
from scm import consts


def uv_geo_from_z(
    lat_deg: xr.DataArray,
    lon_deg: xr.DataArray,
    z: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Compute geostrophic wind components (ug, vg) from geopotential (z) using finite differences.

    Parameters
    ----------
    lat_deg : xr.DataArray
        Latitude in degrees.
    lon_deg : xr.DataArray
        Longitude in degrees.
    z : xr.DataArray
        Geopotential in m^2/s^2. Attention, not geopotential height!
    lat_dim : str
        Name of latitude dimension in the DataArray.
    lon_dim : str
        Name of longitude dimension in the DataArray.

    Returns
    -------
    xr.Dataset
        Dataset containing geostrophic wind components 'ug' and 'vg' in m/s.
    """
    # Constants
    Omega = 7.2921e-5  # Earth's angular velocity (rad/s)
    R = 6371e3  # Earth's mean radius (m)

    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    # 1. Calculate Coriolis parameter (f = 2 * Omega * sin(lat))
    f = 2 * Omega * np.sin(lat_rad)

    # 2. Calculate grid spacing in meters
    # dx = R * cos(lat) * dlon, dy = R * dlat
    dlon = lon_rad.diff(lon_dim)  # noqa: lon_rad remains xr.DataArray
    dlat = lat_rad.diff(lat_dim)  # noqa: lon_rad remains xr.DataArray

    dx = R * np.cos(lat_rad) * dlon.mean()
    dy = R * dlat.mean()

    # 3. Compute gradients of geopotential (z)
    # ERA5 'z' is geopotential (m^2/s^2). If you have geopotential height, multiply by 9.80665
    dz_dy = z.differentiate(lat_dim) / dy
    dz_dx = z.differentiate(lon_dim) / dx

    # 4. Compute components
    ug = -dz_dy / f
    vg = dz_dx / f

    return xr.Dataset({"ug": ug, "vg": vg})


def th_from_tk(
    t_k: xr.DataArray,
    p_hPa: xr.DataArray,
) -> xr.DataArray:
    """Convert temperature (K) to potential temperature (K).

    Parameters
    ----------
    t_k : xr.DataArray
        Temperature in Kelvin.
    p_hPa : xr.DataArray
        Pressure in hPa.

    Returns
    -------
    xr.DataArray
        Potential temperature in Kelvin.
    """
    p0_hPa = 1000.0  # Reference pressure in hPa
    exp = (consts.gamma - 1) / consts.gamma
    th_k = t_k * (p0_hPa / p_hPa) ** exp
    return th_k
