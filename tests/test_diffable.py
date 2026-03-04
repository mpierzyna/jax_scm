from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from scm import mo
from scm.grid import StaggeredGrid
from scm.mynn import closure
from scm.mynn.interfaces import ProgVarsMYNN
from shared import FIXTURE_ROOT


@pytest.fixture
def mynn_state() -> Tuple[
    StaggeredGrid,
    ProgVarsMYNN,
    ProgVarsMYNN,
    mo.MOResult,
]:
    Nz = 50
    grid = StaggeredGrid(H=2000, Nz=Nz)

    # Synthetic state (full levels, Nz=50)
    state = ProgVarsMYNN(
        u=jnp.linspace(0, 10, Nz),
        v=jnp.zeros(Nz),
        th=jnp.linspace(288, 300, Nz),
        qke=jnp.ones(Nz) * 0.1,
        qv=jnp.ones(Nz) * 0.001,
    )

    # Synthetic gradients (half levels, Nz_h=51)
    grads = ProgVarsMYNN(
        u=jnp.ones(Nz + 1) * 0.1,
        v=jnp.zeros(Nz + 1),
        th=jnp.ones(Nz + 1) * 0.01,
        qke=jnp.zeros(Nz + 1),
        qv=jnp.zeros(Nz + 1),
    )

    # Synthetic MO result
    mo_res = mo.MOResult(
        u_st=jnp.array(0.3),
        w_th=jnp.array(0.05),
        w_thv=jnp.array(0.05),
        w_qv=jnp.array(0.0),
        L=jnp.array(-100.0),  # Unstable
        zeta=jnp.array(-0.1),
        zeta_err=jnp.array(0.0),
        m10=jnp.array(5.0),
        th2=jnp.array(289.0),
        th_s=jnp.array(290.0),
        u_w=jnp.array(-0.1),
        v_w=jnp.array(0.0),
    )

    return grid, state, grads, mo_res


@pytest.mark.parametrize(
    "prescribe,th_s,w_th_s",
    [
        ("th_s", 292, None),  # unstable, th_s
        ("th_s", 288, None),  # stable, th_s
        ("w_th_s", None, -0.05),  # unstable, w_th_s
        ("w_th_s", None, +0.05),  # stable, w_th_s
    ],
)
def test_mo_diffable(prescribe, th_s, w_th_s):
    """Check differentiability of MO solver by optimizing wind speed to match u_st.
    Check all possible solver branches (th_s forcing/w_th_s forcing, stable/unstable)
    """
    mo_fn = mo.init_mo_sfc(
        z0m=0.1,
        z0h=0.01,
        z=5,
        sim_funcs=mo.BusingerDyerSimFuncs(),
        prescribe=prescribe,
    )

    @jax.jit
    def objective(u):
        """Example objective: optimize u to match a certain surface friction velocity"""
        res = mo_fn(u_0=u, v_0=0, th_0=290, qv_0=0, w_qv_s=0.0, th_s=th_s, w_th_s=w_th_s)
        return (res.u_st - 0.5) ** 2

    u = 2.0
    lam = 0.95
    for i in range(10):
        obj_grad, obj_loss = jax.value_and_grad(objective)(u)
        u = u - obj_grad * lam**i

    assert jnp.isfinite(u)
    assert u < 2.0


def test_mynn_diffable(mynn_state):
    """Test differentiability with respect to prognostic wind speed."""
    grid, state, grads, mo_res = mynn_state
    closure_fn = closure.init_closure(grid)

    @jax.jit
    def objective(u_vals):
        state_ = ProgVarsMYNN(u=u_vals, v=state.v, th=state.th, qke=state.qke, qv=state.qv)
        diag = closure_fn(state_, grads, mo_res)

        # Km, Kh, ct2 are on half levels (Nz+1 = 51)
        # qke_eps is on full levels (Nz = 50)
        return jnp.mean(diag.Km) + jnp.mean(diag.Kh) + jnp.mean(diag.qke_eps) + jnp.mean(diag.ct2)

    # Calculate gradient with respect to u
    d_du = jax.grad(objective)(state.u)

    assert jnp.all(jnp.isfinite(d_du))
    # As noted before, state.u is not directly used in closure_fn, so d_du should be 0.
    assert jnp.all(d_du == 0.0)


def test_mynn_grads_diffable(mynn_state):
    """Test differentiability with respect to wind gradient."""
    grid, state, grads, mo_res = mynn_state
    closure_fn = closure.init_closure(grid)

    @jax.jit
    def objective(du_dz):
        grads_ = ProgVarsMYNN(u=du_dz, v=grads.v, th=grads.th, qke=grads.qke, qv=grads.qv)
        diag = closure_fn(state, grads_, mo_res)

        # u_w is on half levels (51)
        # qke_P_S is on full levels (50)
        return jnp.mean(diag.u_w) + jnp.mean(diag.qke_P_S)

    d_dgrads = jax.grad(objective)(grads.u)

    assert jnp.all(jnp.isfinite(d_dgrads))
    assert jnp.any(d_dgrads != 0.0)


def test_mynn_mo_res_diffable(mynn_state):
    """Test differentiability with respect to MO results (surface coupling)."""
    grid, state, grads, mo_res = mynn_state
    closure_fn = closure.init_closure(grid)

    @jax.jit
    def objective(u_st):
        mo_res_ = mo.MOResult(
            u_st=u_st,
            w_th=mo_res.w_th,
            w_thv=mo_res.w_thv,
            w_qv=mo_res.w_qv,
            L=mo_res.L,
            zeta=mo_res.zeta,
            zeta_err=mo_res.zeta_err,
            m10=mo_res.m10,
            th2=mo_res.th2,
            th_s=mo_res.th_s,
            u_w=mo_res.u_w,
            v_w=mo_res.v_w,
        )
        diag = closure_fn(state, grads, mo_res_)
        return jnp.mean(diag.Km) + jnp.mean(diag.u_w)

    d_dust = jax.grad(objective)(mo_res.u_st)

    assert jnp.isfinite(d_dust)
    assert d_dust != 0.0


def test_e2e_diffable():
    """Test end-to-end differentiability of a short simulation"""
    from scm.examples.gabls1 import get_gabls1
    from scm.config import load_namelist
    from scm.mynn.model import init_model
    from scm.time_stepping import simulate

    cfg = load_namelist(FIXTURE_ROOT / "gabls1/namelist_cn.yaml")
    cfg.dt_s_out = cfg.dt_s_out  # output every step
    cfg.print_advanced_status = False  # simulation cannot be jitted with advanced outputs

    # Setup a few steps of a simulation
    sim = get_gabls1(Nz=32)
    sim.t_end_s = int(cfg.dt_s * 100)
    model = init_model(sim=sim, cfg=cfg)

    @jax.jit
    def objective(u):
        out = simulate(model=model, sim=sim, cfg=cfg)
        return jnp.mean(out.state_traj.u - u)

    d_du = jax.grad(objective)(jnp.ones(sim.grid.Nz) * 5.0)

    assert jnp.all(jnp.isfinite(d_du))
    assert jnp.any(d_du != 0.0)
