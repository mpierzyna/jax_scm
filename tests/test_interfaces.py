import dataclasses

import jax.numpy as jnp
import jax.tree
import numpy as np
import pytest
import xarray as xr
from shared import FIXTURE_ROOT

from scm.examples.gabls1 import get_gabls1
from scm.interfaces import Forcing, Output, Simulation
from scm.io.local import ds_to_dataclass
from scm.mo import MOResult
from scm.mynn.interfaces import DiagVarsMYNN, ProgVarsMYNN, TendsVarsMYNN

Nz = 64


@pytest.fixture
def sim() -> Simulation:
    """GABLS1 simulation for testing."""
    return get_gabls1(Nz=Nz)


@pytest.fixture
def out() -> Output:
    ds = xr.open_dataset(FIXTURE_ROOT / "gabls1/out_ab2.nc")
    state_traj = ds_to_dataclass(ds, ProgVarsMYNN)
    diag_traj = ds_to_dataclass(ds, DiagVarsMYNN)
    mo_traj = ds_to_dataclass(ds, MOResult, prefix="mo")
    t_s = jnp.array(ds["_t_s"].values)
    tends_traj = ds_to_dataclass(ds, TendsVarsMYNN)
    return Output(state_traj=state_traj, diag_traj=diag_traj, mo_traj=mo_traj, t_s=t_s, tends_traj=tends_traj)


class TestUpdateInit:
    def test_swap_full_init(self, sim: Simulation):
        """Passing new_init replaces all prognostic variables and updates t_start_s."""
        new_init = dataclasses.replace(sim.init, u=jnp.zeros(sim.grid.Nz))
        updated = sim.update_init(new_t_start_s=3600, new_init=new_init)

        assert updated.t_start_s == 3600
        assert updated.t_end_s == sim.t_end_s
        assert jnp.all(updated.init.u == 0.0)
        # other fields unchanged
        assert jnp.allclose(updated.init.th, sim.init.th)

    def test_update_single_field(self, sim: Simulation):
        """Passing a single keyword field updates only that field; others remain unchanged."""
        new_u = jnp.ones(sim.grid.Nz) * 5.0
        updated = sim.update_init(new_t_start_s=1800, u=new_u)

        assert updated.t_start_s == 1800
        assert jnp.allclose(updated.init.u, new_u)
        assert jnp.allclose(updated.init.th, sim.init.th)
        assert jnp.allclose(updated.init.qke, sim.init.qke)

    def test_update_multiple_fields(self, sim: Simulation):
        """Multiple keyword fields are all updated while untouched fields are preserved."""
        new_u = jnp.ones(sim.grid.Nz) * 3.0
        new_th = jnp.ones(sim.grid.Nz) * 270.0
        updated = sim.update_init(new_t_start_s=7200, u=new_u, th=new_th)

        assert jnp.allclose(updated.init.u, new_u)
        assert jnp.allclose(updated.init.th, new_th)
        assert jnp.allclose(updated.init.v, sim.init.v)

    def test_does_not_mutate_original(self, sim: Simulation):
        """The original Simulation and its init are not modified by the call."""
        orig_u = sim.init.u.copy()
        _ = sim.update_init(new_t_start_s=3600, u=jnp.zeros(sim.grid.Nz))
        assert jnp.allclose(sim.init.u, orig_u)

    def test_t_start_at_boundary(self, sim: Simulation):
        """new_t_start_s equal to the original t_start_s is a valid lower bound."""
        # new_t_start_s == t_start_s is allowed
        updated = sim.update_init(new_t_start_s=sim.t_start_s, u=jnp.zeros(sim.grid.Nz))
        assert updated.t_start_s == sim.t_start_s

    def test_t_start_just_before_end(self, sim: Simulation):
        """new_t_start_s one second before t_end_s is the valid upper bound."""
        # new_t_start_s == t_end_s - 1 is allowed (< t_end_s)
        updated = sim.update_init(new_t_start_s=sim.t_end_s - 1, u=jnp.zeros(sim.grid.Nz))
        assert updated.t_start_s == sim.t_end_s - 1

    def test_invalid_t_start_at_end(self, sim: Simulation):
        """new_t_start_s equal to t_end_s is rejected (upper bound is exclusive)."""
        with pytest.raises(ValueError, match="new_t_start_s"):
            sim.update_init(new_t_start_s=sim.t_end_s, u=jnp.zeros(sim.grid.Nz))

    def test_invalid_t_start_beyond_end(self, sim: Simulation):
        """new_t_start_s beyond t_end_s is rejected."""
        with pytest.raises(ValueError, match="new_t_start_s"):
            sim.update_init(new_t_start_s=sim.t_end_s + 3600, u=jnp.zeros(sim.grid.Nz))

    def test_invalid_t_start_before_start(self, sim: Simulation):
        """new_t_start_s before the original t_start_s is rejected."""
        with pytest.raises(ValueError, match="new_t_start_s"):
            sim.update_init(new_t_start_s=sim.t_start_s - 1, u=jnp.zeros(sim.grid.Nz))

    def test_both_new_init_and_fields_raises(self, sim: Simulation):
        """Supplying both new_init and keyword fields raises ValueError."""
        new_init = dataclasses.replace(sim.init, u=jnp.zeros(sim.grid.Nz))
        with pytest.raises(ValueError):
            sim.update_init(new_t_start_s=3600, new_init=new_init, u=jnp.zeros(sim.grid.Nz))

    def test_returns_simulation_type(self, sim: Simulation):
        """The return value is a Simulation instance, not a raw dataclass replacement."""
        updated = sim.update_init(new_t_start_s=3600, u=jnp.zeros(sim.grid.Nz))
        assert type(updated) is type(sim)

    def test_forcing_and_grid_preserved(self, sim: Simulation):
        """All non-init Simulation fields (forcing, grid, mo_settings, th_ref) are carried over unchanged."""
        updated = sim.update_init(new_t_start_s=3600, u=jnp.zeros(sim.grid.Nz))
        assert updated.forcing is sim.forcing
        assert updated.grid is sim.grid
        assert updated.mo_settings is sim.mo_settings
        assert updated.th_ref == sim.th_ref


class TestOutput:
    def test_single(self, out: Output):
        """Select single timestep and check that flattened"""
        i = 10
        out_ = out[i]

        # Dimensionality of all variables should be one less (time dimension removed)
        dims = jax.tree.map(lambda x: x.ndim - 1, out)
        dims_ = jax.tree.map(lambda x: x.ndim, out_)
        assert dims == dims_

        # Length should be zero as time dimension is removed
        assert len(out_) == 0

        # Compare
        comp = jax.tree.map(lambda x, x_: jnp.allclose(x[i], x_), out, out_)
        comp, _ = jax.tree.flatten(comp)
        assert jnp.all(jnp.array(comp))

    def test_slicing(self, out: Output):
        """Select subset using slicing"""
        out_ = out[:100]  # first 100 steps
        assert len(out_) == 100

        out__ = out_[::5]  # every 5th sample
        assert len(out__) == 20

    def test_masking(self, out: Output):
        """Select subset using mask"""
        mask = np.random.uniform(size=len(out))
        mask = mask < 0.5
        out_ = out[mask]
        assert len(out_) == mask.sum()

    def test_iter(self, out: Output):
        """Iterate over output. Should return each time step as 1D Output"""
        out = out[:10]  # first 10 steps for testing

        for i, out_ in enumerate(out):
            # Test that correct type returned. Rest tested by `test_single`
            assert isinstance(out_, Output)
            assert len(out_) == 0


class TestForcing:
    def test_fn_validation(self):
        """Test that forcing function returns jnp.ndarray"""

        def _fn_ok(t):
            return jnp.array([0.0, 1.0])  # returns jnp.ndarray

        def _fn_bad(t):
            return 0.0  # returns float, not jnp.ndarray

        shared = {
            "u_geo": _fn_ok,
            "v_geo": _fn_ok,
            "f_c": 1e-4,
            "w_qv_s": _fn_ok,
        }

        with pytest.raises(ValueError, match="must return jnp.ndarray"):
            _ = Forcing(**shared, w_th_s=_fn_bad)  # typical mistake to not wrap float in jnp.array

        # This doesn't raise error
        _ = Forcing(**shared, w_th_s=_fn_ok)
