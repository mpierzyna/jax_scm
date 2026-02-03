from __future__ import annotations

import hashlib
import logging
import pathlib
import pickle
from functools import wraps
from typing import Callable

import xarray as xr

logger = logging.getLogger("scm.io.cache")


class XRCache:
    """Simple cache for xarray datasets to avoid redundant downloading."""

    def __init__(self, cache_dir: str | pathlib.Path, disable: bool = False):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disable = disable

    @staticmethod
    def get_hash(fn: Callable, *args, **kwargs) -> str:
        """Generate a hash string from function arguments and function itself."""

        # Hash func code
        # This is based on https://github.com/joblib/joblib/blob/main/joblib/memory.py#L655
        # fn_code_h = hash(getattr(fn, "__code__", None))
        # fn_hash = (id(fn), hash(fn), fn_code_h)

        hasher = hashlib.md5()
        hasher.update(pickle.dumps((args, frozenset(kwargs.items()))))  # hash arguments
        # hasher.update(pickle.dumps(fn_hash))  # hash function code
        return hasher.hexdigest()

    def cache(self, fn: Callable[..., xr.Dataset]) -> Callable[..., xr.Dataset]:
        """Decorator to cache xarray datasets returned by the decorated function."""

        @wraps(fn)
        def wrapper(*args, **kwargs) -> xr.Dataset:
            if self.disable:
                logger.debug("Cache disabled. Computing result without caching.")
                return fn(*args, **kwargs)

            # Compute hash from arguments and use it as cache file path
            input_hash = self.get_hash(fn, *args, **kwargs)
            cache_file = self.cache_dir / f"{fn.__name__}_{input_hash}.nc"

            if cache_file.exists():
                logger.debug(f"Cache hit: {cache_file}")
                ds = xr.open_dataset(cache_file)
            else:
                logger.debug(f"Cache miss: {cache_file}. Computing and caching result.")
                ds = fn(*args, **kwargs)
                ds = ds.load()  # load here so data is available when returned
                ds.to_netcdf(cache_file)
            return ds

        return wrapper
