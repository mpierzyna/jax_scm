import jax.numpy as jnp
import pytest

from scm.config import AdaptiveTimestepConfig, Namelist, TimeIntMethod
from scm.examples.gabls1 import get_gabls1
from scm.mynn.model import init_model
from scm.time_stepping import simulate

# Define the three test configurations
CONFIGS = [
    # Explicit Fixed
    Namelist(
        time_int=TimeIntMethod.EXPLICIT,
        dt_s=0.01,
        dt_s_out=60.0,
        adaptive_timestep=None,
    ),
    # Explicit Adaptive
    Namelist(
        time_int=TimeIntMethod.EXPLICIT,
        dt_s=0.01,
        dt_s_out=60.0,
        adaptive_timestep=AdaptiveTimestepConfig(cfl_max=0.1, dt_s_max=0.1),
    ),
    # Implicit Fixed
    Namelist(
        time_int=TimeIntMethod.IMPLICIT,
        dt_s=1.0,
        dt_s_out=60.0,
        adaptive_timestep=None,
    ),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_time_stepping_configs(cfg):
    # Setup short simulation (3h to make it over the initial Km build up)
    sim = get_gabls1()
    sim.t_end_s = 3 * 60 * 60  # 3 hours

    # Initialize model (implicit must be True for implicit time stepping)
    model = init_model(sim, cfg)

    # Run simulation
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Verify no NaNs in prognostic variables
    label = f"{cfg.time_int} (adaptive={cfg.adaptive_timestep is not None})"
    assert not jnp.any(jnp.isnan(out.state_traj.u)), f"NaN detected in u for {label}"
    assert not jnp.any(jnp.isnan(out.state_traj.v)), f"NaN detected in v for {label}"
    assert not jnp.any(jnp.isnan(out.state_traj.th)), f"NaN detected in th for {label}"
    assert not jnp.any(jnp.isnan(out.state_traj.qke)), f"NaN detected in qke for {label}"
