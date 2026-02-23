from __future__ import annotations

import dataclasses

import pandas as pd
import xarray as xr
from jax import numpy as jnp

from scm.interfaces import Output, Simulation


def make_dataset(
    out: Output,
    sim: Simulation,
    time: jnp.ndarray | pd.DatetimeIndex | None = None,
) -> xr.Dataset:
    """Convert history to xarray Dataset."""

    def _get_dims(a: jnp.ndarray) -> tuple[str, ...]:
        if a.ndim == 1:
            return ("time",)
        elif a.ndim == 2:
            _, nz = a.shape
            if nz == sim.grid.Nz:
                return ("time", "z")
            elif nz == sim.grid.Nz_h:
                return ("time", "zh")
            else:
                raise ValueError(f"Unexpected vertical dimension size: {nz}")
        else:
            raise ValueError(f"Unsupported number of dimensions: {a.ndim}")

    if time is None:
        time = out.t_s

    state_dict = dataclasses.asdict(out.state_traj)
    state_dict = {v: (_get_dims(v_data), v_data) for v, v_data in state_dict.items()}
    state_ds = xr.Dataset(state_dict, coords={"time": time, "z": sim.grid.z, "zh": sim.grid.zh})

    diag_dict = dataclasses.asdict(out.diag_traj)
    diag_dict = {v: (_get_dims(v_data), v_data) for v, v_data in diag_dict.items()}
    diag_ds = xr.Dataset(diag_dict, coords={"time": time, "zh": sim.grid.zh, "z": sim.grid.z})

    mo_dict = dataclasses.asdict(out.mo_traj)
    mo_dict = {f"mo_{v}": (("time",), v_data) for v, v_data in mo_dict.items()}
    mo_ds = xr.Dataset(mo_dict, coords={"time": time})

    return xr.merge([state_ds, diag_ds, mo_ds])
