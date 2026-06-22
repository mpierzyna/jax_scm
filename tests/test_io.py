import dataclasses

import jax
import numpy as np
import pytest
import xarray as xr

from scm.config import Namelist, TimeIntMethod
from scm.examples import get_gabls1
from scm.io.local import ds_to_dataclass, out_to_ds
from scm.mynn.interfaces import ProgVarsMYNN
from scm.mynn.io import sim_from_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate


def test_sim_from_ds():
    """Test restoring from ds output dataset."""
    # Test that a simulation can be restored from an output dataset, and that the new output matches the original.
    cfg = Namelist(time_int=TimeIntMethod.IMPLICIT, dt_s=1.0)

    # Run simulation normally
    sim = get_gabls1(Nz=64)
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out=out, sim=sim)

    # Restore simulation from output dataset
    sim_ = sim_from_ds(ds)
    model_ = init_model(sim_, cfg=cfg)
    out_ = simulate(model=model_, sim=sim_, cfg=cfg)
    ds_ = out_to_ds(out=out_, sim=sim_)

    # Compute errors between original and restored simulations
    ds_scale = ds.isel(time=-1).mean()  # normalize by final time step (mean for profiles)
    rel_mae = (ds - ds_).mean() / ds_scale
    rel_rmse = ((ds - ds_) ** 2).mean() ** 0.5 / ds_scale

    # Skip humidity variables and v_geo because they are zero in GABLS1
    vars_skip = [
        "qv",
        "w_qv",
        "mo_w_qv",
        "mo_dqvdz",
        "frc_v_geo",
        "frc_w_qv_s",
        "mo_zeta_err",
        "mo_L",  # todo: why nan?
        "dudt",  # Tendencies not correct for implicit timestepping
        "dvdt",
        "dthdt",
        "dqvdt",
        "dqkedt",
    ]

    for var in ds.data_vars:
        if var in vars_skip:
            continue

        assert rel_mae[var] < 1e-5, f"Rel. MAE for {var} exceeds threshold: {rel_mae[var]:.2e}"
        assert rel_rmse[var] < 1e-5, f"Rel. RMSE for {var} exceeds threshold: {rel_rmse[var]:.2e}"


def test_ds_metadata():
    """Test that all variables in ds output have metadata (long_name and units)"""
    cfg = Namelist(time_int=TimeIntMethod.IMPLICIT, dt_s=1.0, dt_s_out=30)

    # Run simulation normally
    sim = get_gabls1(Nz=16)
    sim = sim.update(t_end_s=60)  # shorten simulation for test speed
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out=out, sim=sim)

    for var in ds.data_vars:
        assert "long_name" in ds[var].attrs, f"{var} is missing long_name metadata"
        assert "units" in ds[var].attrs, f"{var} is missing units metadata"


def test_to_nc(tmpdir):
    """Test that serialization to netcdf works"""
    cfg = Namelist(time_int=TimeIntMethod.IMPLICIT, dt_s=1.0, dt_s_out=30)

    # Run simulation normally
    sim = get_gabls1(Nz=16)
    sim = sim.update(t_end_s=60)  # shorten simulation for test speed
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Convert to dataset and save to netcdf
    ds = out_to_ds(out=out, sim=sim)
    ds.to_netcdf(tmpdir / "test.nc")


@pytest.mark.parametrize("with_prefix", [False, True])
def test_ds_to_state(with_prefix: bool):
    """Test conversion of xarray dataset to dataclass"""
    # Random state
    Nt, Nz = 100, 64
    fields = dataclasses.fields(ProgVarsMYNN)
    data = {
        f.name: (
            ("time", "z"),
            np.random.random((Nt, Nz)),
        )
        for f in fields
    }

    # Test that prefix gets removed correctly
    if with_prefix:
        data = {"mo_" + f: v for (f, v) in data.items()}

    ds = xr.Dataset(data)

    if with_prefix:
        # Prefix should be removed, ie mo_u and u match
        state = ds_to_dataclass(ds, cls=ProgVarsMYNN, prefix="mo")
        assert np.allclose(state.u, ds["mo_u"])
    else:
        # Standard test
        state = ds_to_dataclass(ds, cls=ProgVarsMYNN)
        assert np.allclose(state.u, ds["u"])

    assert isinstance(state, ProgVarsMYNN)
    assert isinstance(state.u, jax.Array)
