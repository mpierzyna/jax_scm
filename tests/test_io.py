import jax.numpy as jnp
from scm.examples.gabls1 import get_gabls1
from scm.mynn.model import init_model
from scm.config import Namelist, TimeIntMethod
from scm.time_stepping import simulate
from scm.mynn.io import sim_from_ds
from scm.io.local import out_to_ds


def test_sim_from_ds():
    cfg = Namelist(time_int=TimeIntMethod.IMPLICIT, dt_s=1.0)

    # Run simulation normally
    sim = get_gabls1()
    sim.t_end_s = 60 * 60  # 1 hour
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out=out, sim=sim)

    # Restore simulation from output dataset
    sim_ = sim_from_ds(ds)
    out_ = simulate(model=model, sim=sim_, cfg=cfg)

    # Compare outputs
    assert jnp.all(jnp.abs(out.state_traj.u - out_.state_traj.u) < 1e-5)
    assert jnp.all(jnp.abs(out.state_traj.v - out_.state_traj.v) < 1e-5)
    assert jnp.all(jnp.abs(out.state_traj.th - out_.state_traj.th) < 1e-5)
