from __future__ import annotations

from typing import TypeVar
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr

from scm import consts

T = TypeVar("T")


@jax.jit
def thv_to_th(*, thv: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Virtual potential temperature to dry potential temperature."""
    return thv / (1 + 0.61 * qv)


@jax.jit
def th_to_thv(*, th: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Dry potential temperature to virtual potential temperature."""
    return th * (1 + 0.61 * qv)


@jax.jit
def w_th_to_w_thv(*, th: jnp.ndarray, w_th: jnp.ndarray, w_qv: jnp.ndarray) -> jnp.ndarray:
    """Sensible heat flux (w'theta') to buoyancy flux (w'theta_v')."""
    return w_th + 0.61 * th * w_qv


@jax.jit
def w_thv_to_w_th(*, th: jnp.ndarray, w_thv: jnp.ndarray, w_qv: jnp.ndarray) -> jnp.ndarray:
    """Buoyancy flux (w'theta_v') to sensible heat flux (w'theta')."""
    return w_thv - 0.61 * th * w_qv


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
    f = get_fc(lat_deg=lat_deg)

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


def get_fc(*, lat_deg: float) -> float:
    """Coriolis parameter at given latitude."""
    omega = 7.2921e-5  # rad/s, Earth's angular velocity
    return float(2 * omega * jnp.sin(jnp.deg2rad(lat_deg)))


def tk_to_th(*, tk: T, p_hPa: T) -> T:
    """Convert temperature (K) to potential temperature (K).

    Parameters
    ----------
    tk : xr.DataArray
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
    th_k = tk * (p0_hPa / p_hPa) ** exp
    return th_k


def get_p_hypsometric(*, z: jnp.ndarray, tkv: jnp.ndarray, p_0_hPa: jnp.ndarray | float) -> jnp.ndarray:
    """Convert geopotential height (m) to pressure (hPa) using hypsometric equation."""
    p_hPa = []
    for i in range(1, len(z) + 1):
        rhs = -consts.g / consts.Rd * jnp.trapezoid(1 / tkv[:i], x=z[:i])
        p_i = p_0_hPa * jnp.exp(rhs)
        p_hPa.append(p_i)
    return jnp.array(p_hPa)
