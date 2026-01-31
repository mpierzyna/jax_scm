from __future__ import annotations

from typing import Callable

import jax
import numpy as np
import xarray as xr
from jax import numpy as jnp


def get_ts_interp_fn(time_s: jnp.ndarray, data: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get a jitted time series interpolation function for jax."""
    idx = jnp.arange(data.shape[0])

    @jax.jit
    def interp_fn(t_s: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(t_s, time_s, data)

    @jax.jit
    def interp_2d_fn(t_s: jnp.ndarray) -> jnp.ndarray:
        # Interpolate index
        i = jnp.interp(t_s, time_s, idx)
        i_low = jnp.floor(i).astype(int)
        i_high = jnp.ceil(i).astype(int)
        i = i - i_low  # fractional part
        return data[i_low, :] + i * (data[i_high, :] - data[i_low, :])

    if data.ndim == 1:
        interp_fn(time_s[0])  # warm up jitting
        return interp_fn
    elif data.ndim == 2:
        interp_2d_fn(time_s[0])  # warm up jitting
        return interp_2d_fn
    else:
        raise ValueError("data must be 1D or 2D")


def xr_interp_vert(ds: xr.Dataset, z: xr.DataArray, z_target: xr.DataArray, dim: str) -> xr.Dataset:
    """ATTENTION! z and z target must increase monotonically!"""

    def _searchsorted(a, v):
        # Break out function for debugging apply_ufunc
        res = np.searchsorted(a, v, side="left")
        return res

    # Compute indices of bracketing bottom_top levels
    bt_j = xr.apply_ufunc(
        _searchsorted,
        z,
        z_target,
        input_core_dims=[[dim], [dim]],
        exclude_dims={dim},
        output_core_dims=[[dim]],
        vectorize=True,
    )
    bt_i = (bt_j - 1).clip(min=0)
    bt_j = bt_j.clip(max=ds.sizes[dim] - 1)

    # Get values for bracketing levels
    z_i = z.sel({dim: bt_i})
    z_j = z.sel({dim: bt_j})

    # Get data for bracketing levels
    vars_interp = [v for v in ds if dim in ds[v].dims]
    ds_i = ds[vars_interp].sel({dim: bt_i})
    ds_j = ds[vars_interp].sel({dim: bt_j})

    # Interpolate
    ds_interp = ds_i + (z_target - z_i) * (ds_j - ds_i) / (z_j - z_i)
    for v in vars_interp:
        # Attributes are lost in interpolation. Restore them.
        ds_interp[v].attrs.update(ds[v].attrs)

    # Assign interpolation target values as coordinate
    ds_interp = ds_interp.assign_coords({dim: z_target})
    return ds_interp
