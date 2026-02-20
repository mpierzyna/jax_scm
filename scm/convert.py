from __future__ import annotations

from typing import TypeVar, Literal, Callable
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr

from scm import consts

T = TypeVar("T")


def tv_to_t(*, tv: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Compute dry temperature (from virtual air/potential temperature)"""
    return tv / (1 + 0.61 * qv)


def t_to_tv(*, t: jnp.ndarray, qv: jnp.ndarray) -> jnp.ndarray:
    """Compute virtual temperature (from air/potential temperature)"""
    return t * (1 + 0.61 * qv)


def w_th_to_w_thv(*, th: jnp.ndarray, w_th: jnp.ndarray, w_qv: jnp.ndarray) -> jnp.ndarray:
    """Sensible heat flux (w'theta') to buoyancy flux (w'theta_v')."""
    return w_th + 0.61 * th * w_qv


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


def get_p_rho_fn(mode: Literal["th", "tk"]) -> Callable:
    """Get function to compute pressure and density profiles"""

    def get_p_rho(t, qv, z, p_s):
        """Computes density and pressure profiles using the hypsometric equation.

        Parameters
        ----------
        t : jnp.ndarray
            Air temperature profile (K) or potential temperature profile (K), depending on mode.
        qv : jnp.ndarray
            Specific humidity profile (kg/kg).
        z : jnp.ndarray
            Geopotential height profile (m).
        p_s : float
            Surface pressure (Pa).

        Returns
        -------
        p_profile : jnp.ndarray
            Pressure profile (Pa).
        rho_profile : jnp.ndarray
            Density profile (kg/m^3).
        """
        p0 = 100000.0

        def scan_fn(p_i, carry_vars):
            """Compute pressure and density from previous level using hypsometric equation."""
            if mode == "th":
                # Estimate air temp from pot temp
                th_j, qv_j, dz = carry_vars
                tk_j = th_j * (p_i / p0) ** (consts.Rd / consts.cp)
            elif mode == "tk":
                # Directly use air temp
                tk_j, qv_j, dz = carry_vars
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'th' or 'tk'.")

            # Get virtual air temp
            tkv_j = t_to_tv(t=tk_j, qv=qv_j)

            # Use hypsometric equation to compute pressure at this level
            # p_j = p_prev * exp(-g * dz / (Rd * Tv))
            p_j = p_i * jnp.exp(-consts.g * dz / (consts.Rd * tkv_j))

            # Density from ideal gas law
            rho_j = p_j / (consts.Rd * tkv_j)

            return p_j, (p_j, rho_j)

        dz = jnp.diff(z)
        dz = jnp.concat([jnp.array([z[0]]), dz])

        # Integrate from surface to top of the column
        _, (p_profile, rho_profile) = jax.lax.scan(scan_fn, p_s, (t, qv, dz))

        return p_profile, rho_profile

    return get_p_rho


p_rho_from_th = get_p_rho_fn(mode="th")
p_rho_from_tk = get_p_rho_fn(mode="tk")


def w_eff(*, omega, rho):
    """Compute vertical velocity (w) from pressure vertical velocity (omega)."""
    return -omega / (rho * consts.g)
