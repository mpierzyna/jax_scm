from __future__ import annotations
from typing import Literal, List

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger("jax_scm.data.era5")

# Variables to download from ERA5
VARS_PL = ["z", "u", "v", "q", "t"]
VARS_SL = ["skt", "ie", "sp", "ishf", "t2m", "u10", "v10", "z"]  # ishf and onwards mostly for checking


def _era5_pl_destine() -> xr.Dataset:
    """Destine zarr has PL and SL data in separate stores and fewer PL levels"""
    ds_pl = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    ds_pl = ds_pl[VARS_PL]

    ds_sl = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    ds_sl = ds_sl[VARS_SL]
    ds_sl = ds_sl.rename({"z": "z_sfc"})  # avoid conflict with pressure level z

    ds = xr.merge([ds_pl, ds_sl], compat="override")
    # ds = ds.rename_dims(valid_time="time")
    # ds = ds.assign_coords(time="valid_time")
    # ds = ds.rename_vars({"valid_time": "time"})

    return ds


def _era5_pl_google() -> xr.Dataset:
    # Here, SL and PL in same zarr store
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )

    # Rename to variables to their short names for easier access
    vars_query = VARS_PL + VARS_SL
    names = [(v, ds[v].attrs["short_name"]) for v in ds.data_vars]
    names = [(v, v_short) for v, v_short in names if v_short in vars_query]
    names = dict(names)
    for v in vars_query:
        if v not in names.values():
            raise ValueError(f"Variable {v} not found in dataset.")
    names["geopotential_at_surface"] = "z_sfc"  # avoid conflict with pressure level z

    ds = ds[list(names.keys())]
    ds = ds.rename(names)

    return ds


def download_data(
    lat_deg: float,
    lon_deg: float,
    time_slice: slice | str,
    source: Literal["destine", "google"],
) -> xr.Dataset:
    """Download ERA5 data for given lat/lon and time range.
    Attention: Pycharm debugger doesn't work here because of async zarr and xarray.
    """
    print(f"Downloading ERA5 data ({source})... This may take a while.")

    if lon_deg < 0:
        lon_deg += 360  # ERA5 uses 0 to 360 for longitude

    # Force lat/lon to 0.25 deg grid
    lat_deg = np.round(lat_deg * 4) / 4
    lon_deg = np.round(lon_deg * 4) / 4

    # Create bounding box for geostrophic wind calculation
    ddeg = 0.25
    lat_sel = slice(lat_deg + ddeg, lat_deg - ddeg)  # lat is ordered from north (90) to south (-90), so reverse slice
    lon_sel = slice(lon_deg - ddeg, lon_deg + ddeg)

    # Open dataset and select variables, region, and time
    if source == "google":
        ds = _era5_pl_google()
    elif source == "destine":
        ds = _era5_pl_destine()
    else:
        raise ValueError(f"Unknown source: {source}")
    ds = ds.sel(time=time_slice)
    ds = ds.sel(latitude=lat_sel, longitude=lon_sel)

    # Merge
    # logger.info(f"Download size: {ds.nbytes / 1e6:.1f} MB")
    logger.info(f"Size on disk: {ds.nbytes / 1e6:.1f} MB")
    logger.info(ds.sizes)
    logger.info(list(ds.data_vars))

    # Load into memory
    ds = ds.load()
    print("Download complete.")

    return ds


if __name__ == "__main__":
    from scm.io.cache import XRCache

    logging.basicConfig(level="INFO")
    xr_cache = XRCache("../forcing/.era5_cache", disable=False)
    ds = xr_cache.cache(download_data)(52.0, 5.0, "2020-01-01", "google")
