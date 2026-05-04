from __future__ import annotations

import dataclasses

import pandas as pd
import xarray as xr
from jax import numpy as jnp

from scm.forcing.utils import sample_forcing
from scm.interfaces import Output, Simulation


def out_to_ds(
    out: Output,
    sim: Simulation,
    time: jnp.ndarray | pd.DatetimeIndex | None = None,
) -> xr.Dataset:
    """Convert simulation output to xarray Dataset."""

    def _get_dims(a: jnp.ndarray) -> tuple[str, ...]:
        if isinstance(a, float) or a.ndim == 0:
            return ()

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

    def _add_metadata(ds: xr.Dataset, cls, prefix: str = "") -> None:
        """Add metadata from dataclass fields to xarray Dataset variables."""
        metadata = {f"{prefix}{f.name}": f.metadata for f in dataclasses.fields(cls) if not f.metadata == {}}
        for vname in ds.data_vars:
            vname = str(vname)
            if vname not in metadata:
                print(f"Warning: No metadata found for variable {vname}")
                continue
            ds[vname].attrs.update(metadata[vname])

    # Create coordinates for xarray
    if time is None:
        if sim.t_index_fn is not None:
            time = sim.t_index_fn(out.t_s)
        else:
            time = out.t_s
    coords = {"time": time, "z": sim.grid.z, "zh": sim.grid.zh}

    # Convert simulation output to xarray Datasets
    state_dict = dataclasses.asdict(out.state_traj)
    state_dict = {v: (_get_dims(v_data), v_data) for v, v_data in state_dict.items()}
    state_ds = xr.Dataset(state_dict, coords=coords)
    _add_metadata(state_ds, cls=type(out.state_traj))

    diag_dict = dataclasses.asdict(out.diag_traj)
    diag_dict = {v: (_get_dims(v_data), v_data) for v, v_data in diag_dict.items()}
    diag_ds = xr.Dataset(diag_dict, coords=coords)
    _add_metadata(diag_ds, cls=type(out.diag_traj))

    mo_dict = dataclasses.asdict(out.mo_traj)
    mo_dict = {f"mo_{v}": (("time",), v_data) for v, v_data in mo_dict.items()}
    mo_ds = xr.Dataset(mo_dict, coords=coords)
    _add_metadata(mo_ds, cls=type(out.mo_traj), prefix="mo_")

    ds = xr.merge([state_ds, diag_ds, mo_ds])

    # Add metadata
    # keep original simulation time axis
    ds["_t_s"] = xr.DataArray(
        out.t_s,
        dims="time",
        attrs={
            "long_name": "simulation time",
            "units": "s",
        },
    )
    ds.attrs["name"] = sim.name
    ds.attrs["t_start_s"] = sim.t_start_s
    ds.attrs["t_end_s"] = sim.t_end_s
    ds.attrs["th_ref"] = sim.th_ref

    try:
        # Sample forcing and add to dataset
        forcing_dict = sample_forcing(sim.forcing, out.t_s)
        forcing_dict = {
            f"frc_{v}": (
                _get_dims(v_data),
                v_data,
            )
            for v, v_data in forcing_dict.items()
            if v_data is not None
        }
        forcing_ds = xr.Dataset(forcing_dict, coords=coords)
        _add_metadata(forcing_ds, cls=type(sim.forcing), prefix="frc_")
        ds = xr.merge([ds, forcing_ds])
    except Exception as e:
        print(f"Warning: Could not sample forcing for output dataset. Error: {e}")

    try:
        # Serialize MO settings
        mo = sim.mo_settings.serialize()
        ds.attrs["mo_settings"] = mo
    except Exception as e:
        print(f"Warning: Could not serialize MO settings for output dataset. Error: {e}")

    return ds
