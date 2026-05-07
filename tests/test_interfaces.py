import dataclasses

import jax
import jax.numpy as jnp
import pytest

from scm.examples.gabls1 import get_gabls1
from scm.interfaces import Output, Simulation

Nz = 64


@pytest.fixture
def sim() -> Simulation:
    """GABLS1 simulation for testing."""
    return get_gabls1(Nz=Nz)


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
