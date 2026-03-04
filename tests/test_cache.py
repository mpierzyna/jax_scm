import pytest
from scm.io.cache import XRCache


@pytest.fixture
def xr_cache(tmp_path):
    return XRCache(cache_dir=tmp_path)


def test_argument_caching(xr_cache):
    """If arguments are the same, cached result should be used."""
    call_count = 0

    @xr_cache.cache
    def dummy_function(x, y=1):
        nonlocal call_count
        call_count += 1
        import xarray as xr
        import numpy as np

        data = np.array([[x + y]])
        return xr.Dataset({"data": (("dim1", "dim2"), data)})

    # First call should compute the result
    ds1 = dummy_function(2, y=3)
    assert call_count == 1
    assert ds1["data"].values[0, 0] == 5

    # Second call with same arguments should use cache
    ds2 = dummy_function(2, y=3)
    assert call_count == 1  # No increment
    assert ds2["data"].values[0, 0] == 5

    # Call with different arguments should compute again
    ds3 = dummy_function(4, y=1)
    assert call_count == 2
    assert ds3["data"].values[0, 0] == 5

    # Check that we have two files in the cache directory
    cached_files = list(xr_cache.cache_dir.glob("*.nc"))
    assert len(cached_files) == 2


@pytest.mark.skip("Don't know if this was ever working... Need to check it at some point.")
def test_fn_code_caching(xr_cache):
    """If arguments are the same, but function body changed, cache should not be used."""
    call_count = 0

    @xr_cache.cache
    def dummy_function(x):
        nonlocal call_count
        call_count += 1
        import xarray as xr
        import numpy as np

        data = np.array([[x * 2]])
        return xr.Dataset({"data": (("dim1", "dim2"), data)})

    # First call should compute the result
    ds1 = dummy_function(3)
    assert call_count == 1
    assert ds1["data"].values[0, 0] == 6

    # Next call should come from cache
    ds_cached = dummy_function(3)
    assert call_count == 1  # No increment
    assert ds_cached["data"].values[0, 0] == 6

    # Redefine the function with a different body
    @xr_cache.cache
    def dummy_function(x):
        nonlocal call_count
        call_count += 1
        import xarray as xr
        import numpy as np

        data = np.array([[x * 3]])  # Changed multiplier from 2 to 3
        return xr.Dataset({"data": (("dim1", "dim2"), data)})

    # Call the new function with same argument; should compute again due to code change
    ds2 = dummy_function(3)
    assert call_count == 2  # Incremented
    assert ds2["data"].values[0, 0] == 9
