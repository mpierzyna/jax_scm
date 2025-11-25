from __future__ import annotations

import dataclasses

import xarray as xr
from jax import numpy as jnp

from scm.grid import StaggeredGrid
from scm.interfaces import ProgVars, DiagVars
from scm.mo import MOResult


def make_dataset(
    state_hist: ProgVars,
    diag_hist: DiagVars,
    mo_hist: MOResult,
    time: jnp.ndarray,
    grid: StaggeredGrid,
) -> xr.Dataset:
    """Convert history to xarray Dataset."""

    def _get_dims(a: jnp.ndarray) -> tuple[str, ...]:
        if a.ndim == 1:
            return ("time",)
        elif a.ndim == 2:
            _, nz = a.shape
            if nz == grid.Nz:
                return ("time", "z")
            elif nz == grid.Nz_h:
                return ("time", "zh")
            else:
                raise ValueError(f"Unexpected vertical dimension size: {nz}")
        else:
            raise ValueError(f"Unsupported number of dimensions: {a.ndim}")

    state_dict = dataclasses.asdict(state_hist)
    state_dict = {v: (_get_dims(v_data), v_data) for v, v_data in state_dict.items()}
    state_ds = xr.Dataset(state_dict, coords={"time": time, "z": grid.z, "zh": grid.zh})

    diag_dict = dataclasses.asdict(diag_hist)
    diag_dict = {v: (_get_dims(v_data), v_data) for v, v_data in diag_dict.items()}
    diag_ds = xr.Dataset(diag_dict, coords={"time": time, "zh": grid.zh, "z": grid.z})

    mo_dict = dataclasses.asdict(mo_hist)
    mo_dict = {f"{v}_sfc": (("time",), v_data) for v, v_data in mo_dict.items()}
    mo_ds = xr.Dataset(mo_dict, coords={"time": time})

    return xr.merge([state_ds, diag_ds, mo_ds])
