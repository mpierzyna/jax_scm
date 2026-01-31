import pytest
from scm.io.cache import XRCache


@pytest.fixture
def xr_cache(tmp_path):
    return XRCache(cache_dir=tmp_path)


def test_xr_cache(xr_cache):
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
