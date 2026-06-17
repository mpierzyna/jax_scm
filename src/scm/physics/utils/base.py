from __future__ import annotations

from typing import TypeVar

import jax
import pandas as pd
import xarray as xr

# Type variable for functions that perform pure computations without library specific functions
ArrayT = TypeVar("ArrayT", jax.Array, xr.DataArray, pd.Series)
