from typing import Tuple
import pytest
import jax
import jax.numpy as jnp

from scm import mo
from scm.closures import mynn
from scm.grid import StaggeredGrid
import xarray as xr


@pytest.fixture
def mynn_state() -> Tuple[StaggeredGrid, mynn.ProgVarsMYNN, mynn.ProgVarsMYNN, mo.MOResult]:

    def _d_dz(a: jnp.ndarray, dz: float) -> jnp.ndarray:
        Nz = len(a)
        da_dz = jnp.zeros(Nz + 1)  # half levels
        da_dz = da_dz.at[1:-1].set((a[1:] - a[:-1]) / dz)
        return da_dz

    grid = StaggeredGrid(H=2000, Nz=50)
    ds = xr.open_dataset("../out.nc").isel(time=50)

    state = mynn.ProgVarsMYNN(
        u=jnp.array(ds["u"]),
        v=jnp.array(ds["v"]),
        thv=jnp.array(ds["thv"]),
        q_sq=jnp.array(ds["q_sq"]),
    )
    grads = mynn.ProgVarsMYNN(
        u=_d_dz(state.u, grid.dz),
        v=_d_dz(state.v, grid.dz),
        thv=_d_dz(state.thv, grid.dz),
        q_sq=_d_dz(state.q_sq, grid.dz),
    )
    mo_res = mo.MOResult(
        u_st=jnp.array(ds["u_st_sfc"]),
        w_th=jnp.array(ds["w_th_sfc"]),
        th_s=jnp.array(ds["th_s_sfc"]),
        v_w=jnp.array(ds["v_w_sfc"]),
        u_w=jnp.array(ds["u_w_sfc"]),
        w_q=jnp.array(ds["w_q_sfc"]),
        L=jnp.array(ds["L_sfc"]),
        zeta=jnp.array(ds["zeta_sfc"]),
        zeta_err=jnp.array(ds["zeta_err_sfc"]),
        m10=jnp.array(ds["m10_sfc"]),
        th2=jnp.array(ds["th2_sfc"]),
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
        res = mo_fn(u_0=u, v_0=0, th_0=290, th_s=th_s, w_th_s=w_th_s, w_q_s=None)
        return (res.u_st - 0.5) ** 2

    u = 2.0
    lam = 0.95
    for i in range(10):
        obj_grad, obj_loss = jax.value_and_grad(objective)(u)
        u = u + obj_grad * lam**i

    assert u > 2.0


def test_mynn_diffable(mynn_state):
    grid, state, grads, mo_res = mynn_state
    closure_fn = mynn.init_mynn(grid)

    def objective(u):
        """Pretend to minimize TKE terms. Just a test of differentiability because results of all equations."""
        state_ = mynn.ProgVarsMYNN(u=u, v=state.v, thv=state.thv, q_sq=state.q_sq)
        diag = closure_fn(state_, grads, mo_res)
        q_sq_tt_ = (diag.q_sq_tt[1:] + diag.q_sq_tt[:-1]) / 2  # to full levels
        return jnp.mean(q_sq_tt_ + diag.q_sq_P_S + diag.q_sq_P_B + diag.q_sq_eps)

    d_du = jax.grad(objective)(state.u)
    assert True
