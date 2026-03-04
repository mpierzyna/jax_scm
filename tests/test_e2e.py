import pytest
import xarray as xr

from scm.config import load_namelist
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate
from shared import FIXTURE_ROOT

# Cases used for end-to-end testing.
CASES = {
    "cn": ("out_cn.nc", "namelist_cn.yaml"),
    "ab2": ("out_ab2.nc", "namelist_ab2.yaml"),
}


@pytest.mark.parametrize("case", ["cn", "ab2"])
def test_e2e(case: str) -> None:
    """Test that the full simulation can be run and produces expected output."""
    # Load simulation output and config
    sim_ds_name, sim_cfg = CASES[case]  # simulation to return as fixture
    ds = xr.open_dataset(FIXTURE_ROOT / "gabls1" / sim_ds_name)
    cfg = load_namelist(FIXTURE_ROOT / "gabls1" / sim_cfg)

    # Run simulation again
    # sim = sim_from_ds(ds)  # do not load from disk right now because something wrong
    sim = get_gabls1(Nz=64)
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds_new = out_to_ds(out, sim=sim, time=out.t_s / 60 / 60)

    # Compare new output to original
    err = (ds - ds_new) / ds.mean()
    for var in err.data_vars:
        # Skip forcing
        if "frc" in var:
            continue
        # Skip humidity because not in case
        if "qv" in var:
            continue

        # Check that error is small
        assert err[var].max() < 1e-5, f"Variable {var} differs between runs by more than 1e-5 relative error"
