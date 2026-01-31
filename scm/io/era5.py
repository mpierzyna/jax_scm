from __future__ import annotations
import numpy as np
import xarray as xr
import logging


logger = logging.getLogger("jax_scm.data.era5")


def download_data(lat_deg: float, lon_deg: float, time_slice: slice | str) -> xr.Dataset:
    """Download ERA5 data for given lat/lon and time range.
    Attention: Pycharm debugger doesn't work here because of async zarr and xarray.
    """
    # Variables to download from ERA5
    VARS_PL = ["z", "u", "v", "q", "t"]
    VARS_SL = ["skt"]

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
    ds_pl = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    ds_pl = ds_pl[VARS_PL].sel(valid_time=time_slice)
    ds_pl = ds_pl.sel(latitude=lat_sel, longitude=lon_sel)

    ds_sl = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    ds_sl = ds_sl[VARS_SL].sel(valid_time=time_slice)
    ds_sl = ds_sl.sel(latitude=lat_sel, longitude=lon_sel)

    # Merge
    ds = xr.merge([ds_pl, ds_sl])
    # logger.info(f"Download size: {ds.nbytes / 1e6:.1f} MB")
    logger.info(f"Size on disk: {ds.nbytes / 1e6:.1f} MB")
    logger.info(ds.sizes)
    logger.info(list(ds.data_vars))

    return ds


if __name__ == "__main__":
    from scm.io.cache import XRCache

    logging.basicConfig(level="INFO")
    xr_cache = XRCache(".era5_cache")
    ds = xr_cache.cache(download_data)(52, 4, "2020-01-01")
