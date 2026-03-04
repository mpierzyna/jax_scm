from typing import Tuple
import pytest
import jax
import jax.numpy as jnp

import scm.mynn.interfaces
from scm import mo
from scm.mynn import closure
from scm.grid import StaggeredGrid
import xarray as xr


@pytest.fixture
def mynn_state() -> Tuple[
    StaggeredGrid,
    scm.mynn.interfaces.ProgVarsMYNN,
    scm.mynn.interfaces.ProgVarsMYNN,
    mo.MOResult,
]:

    def _d_dz(a: jnp.ndarray, dz: float) -> jnp.ndarray:
        Nz = len(a)
        da_dz = jnp.zeros(Nz + 1)  # half levels
        da_dz = da_dz.at[1:-1].set((a[1:] - a[:-1]) / dz)
        return da_dz

    grid = StaggeredGrid(H=2000, Nz=50)
    ds = xr.open_dataset("../out.nc").isel(time=50)

    state = scm.mynn.interfaces.ProgVarsMYNN(
        u=jnp.array(ds["u"]),
        v=jnp.array(ds["v"]),
        th=jnp.array(ds["th"]),
        qke=jnp.array(ds["qke"]),
        qv=jnp.array(ds["qv"]),
    )
    grads = scm.mynn.interfaces.ProgVarsMYNN(
        u=_d_dz(state.u, grid.dz),
        v=_d_dz(state.v, grid.dz),
        th=_d_dz(state.th, grid.dz),
        qke=_d_dz(state.qke, grid.dz),
        qv=_d_dz(state.qv, grid.dz),
    )
    mo_res = mo.MOResult(
        u_st=jnp.array(ds["u_st_sfc"]),
        w_th=jnp.array(ds["w_th_sfc"]),
        w_thv=jnp.array(ds["w_thv_sfc"]),
        th_s=jnp.array(ds["th_s_sfc"]),
        v_w=jnp.array(ds["v_w_sfc"]),
        u_w=jnp.array(ds["u_w_sfc"]),
        w_qv=jnp.array(ds["w_qv_sfc"]),
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
        res = mo_fn(u_0=u, v_0=0, th_0=290, th_s=th_s, w_th_s=w_th_s, w_qv_s=0.0, qv_0=0)
        return (res.u_st - 0.5) ** 2

    u = 2.0
    lam = 0.95
    for i in range(10):
        obj_grad, obj_loss = jax.value_and_grad(objective)(u)
        u = u + obj_grad * lam**i

    assert u > 2.0


@pytest.mark.skip(reason="Outdated. Needs to be updated")
def test_mynn_diffable(mynn_state):
    grid, state, grads, mo_res = mynn_state
    closure_fn = mynn.init_closure(grid)

    def objective(u):
        """Pretend to minimize TKE terms. Just a test of differentiability because results of all equations."""
        state_ = scm.mynn.interfaces.ProgVarsMYNN(u=u, v=state.v, th=state.th, qke=state.qke, qv=state.qv)
        diag = closure_fn(state_, grads, mo_res)
        q_sq_tt_ = (diag.w_qke[1:] + diag.w_qke[:-1]) / 2  # to full levels
        return jnp.mean(q_sq_tt_ + diag.qke_P_S + diag.qke_P_B + diag.qke_eps)

    d_du = jax.grad(objective)(state.u)
    assert True
