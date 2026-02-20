"""Semi-implicit Crank-Nicolson solver for vertical diffusion.

Discretization
--------------
For a prognostic variable φ on Nz full levels with diffusivity K on Nz+1 half levels:

    ∂φ/∂t = -(∂F/∂z) + S,    F[i] = -K[i] * (φ[i] - φ[i-1]) / dz

Boundary conditions
-------------------
* Bottom face (i=0): flux is prescribed externally (from MO, treated **explicitly**).
  Internally K_eff[0] = 0 so the surface face is excluded from the implicit operator.
* Top face (i=Nz): zero flux. Enforced by setting K_eff[Nz] = 0 in the operator.

Crank-Nicolson scheme (K fixed at current time step)
-----------------------------------------------------
    (I - dt/2 * A) φ^{n+1} = (I + dt/2 * A) φ^n + dt * S + dt * b_sfc

where A is the tridiagonal diffusion matrix with coefficients (using K_eff):
    A[j, j-1] = K_eff[j] / dz²           (sub-diagonal)
    A[j,   j] = -(K_eff[j] + K_eff[j+1]) / dz²   (diagonal)
    A[j, j+1] = K_eff[j+1] / dz²         (super-diagonal)

and b_sfc[0] = sfc_flux / dz,  b_sfc[j>0] = 0.
"""

from __future__ import annotations

from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

from scm.grid import StaggeredGrid
from scm.interfaces import ModelFn
from scm.mynn.interfaces import ProgVarsMYNN
from scm.time_stepping.utils import clip_state


def cn_solve(
    phi_i: jnp.ndarray,
    K: jnp.ndarray,
    dt: float,
    dz: float,
    S: jnp.ndarray,
    sfc_flux: float | jnp.ndarray,
) -> jnp.ndarray:
    """Solve one Crank-Nicolson diffusion step for a single variable.

    Parameters
    ----------
    phi_i : jnp.ndarray, shape (Nz,)
        State at current time step (full levels).
    K : jnp.ndarray, shape (Nz+1,)
        Eddy diffusivity on half levels (from closure, kept constant).
    dt : float
        Time step [s].
    dz : float
        Grid spacing [m].
    S : jnp.ndarray, shape (Nz,)
        Explicit (non-diffusive) tendency at current step (e.g. Coriolis, production).
    sfc_flux : float or scalar jnp.ndarray
        Surface flux F_sfc = -K_sfc * dφ/dz|_sfc  (sign convention: positive = upward).
        Contributes as +sfc_flux/dz to the tendency at the bottom cell.

    Returns
    -------
    phi_i_next : jnp.ndarray, shape (Nz,)
        Updated state after one CN step.
    """
    # Zero out boundary faces so they are handled as explicit BCs, not implicit diffusion
    K_eff = K.at[0].set(0.0).at[-1].set(0.0)

    # Sub-diagonal: K at lower face of each cell
    A_dl = K_eff[:-1] / dz**2  # shape (Nz,); a[j] = K_eff[j]
    # Super-diagonal: K at upper face of each cell
    A_du = K_eff[1:] / dz**2  # shape (Nz,); c[j] = K_eff[j+1]
    # Diagonal
    A_d = -(A_dl + A_du)  # shape (Nz,)

    # Assemble tridiagonal diffusion matrix A
    A = jnp.diag(A_d) + jnp.diag(A_du[:-1], k=1) + jnp.diag(A_dl[1:], k=-1)

    I = jnp.eye(N=phi_i.shape[0])

    # Surface-flux explicit source: b_sfc[0] = sfc_flux / dz, zero elsewhere
    b_sfc = jnp.zeros_like(phi_i).at[0].set(sfc_flux / dz)

    # lhs = I - (dt / 2.0) * A
    lhs_d = 1 - (dt / 2.0) * A_d
    lhs_dl = -(dt / 2.0) * A_dl
    lhs_du = -(dt / 2.0) * A_du
    rhs = (I + (dt / 2.0) * A) @ phi_i + dt * S + dt * b_sfc

    # Tridiagonal solver expects batches, so reshape to (1, Nz) and (1, Nz, 1) for b
    phi_i_next = tridiagonal_solve(
        dl=lhs_dl[None, :],
        d=lhs_d[None, :],
        du=lhs_du[None, :],
        b=rhs[None, :, None],
    )
    return phi_i_next.squeeze()


def get_cn_step_fn(
    model: ModelFn,
    grid: StaggeredGrid,
) -> Tuple[Callable, Callable]:
    """Factory for semi-implicit Crank-Nicolson stepper (MYNN-specific).

    Diffusion terms are solved implicitly with K fixed at the current time step
    (K diagnosed from the closure and held constant within each CN step).
    Non-diffusive explicit sources (Coriolis, QKE production/dissipation,
    large-scale advection) use AB2 extrapolation after the warmup step.

    The ``model`` **must** be initialized with ``implicit=True`` so that it
    returns only non-diffusive tendencies as the forcing vector S.

    Parameters
    ----------
    model : ModelFn
        MYNN model function (initialized with ``implicit=True``).
    grid : StaggeredGrid
        Vertical grid (provides ``dz``).

    Returns
    -------
    _cn_warmup : Callable
        ``(t_s, dt_s, y0) -> (y1, S0, diag0, mo_res0)``
        Warmup step: CN with S^{prev} = S^0 (first-order accurate in time).
    _cn : Callable
        ``(t_s, dt_s, y1, S_prev) -> (y2, S1, diag1, mo_res1)``
        Regular CN step with AB2-extrapolated explicit sources.
    """

    def _apply_cn(
        y: ProgVarsMYNN,
        S: ProgVarsMYNN,
        diag,
        mo_res,
        dt_s: float,
    ) -> ProgVarsMYNN:
        """Apply one CN diffusion solve for every prognostic variable.

        K values come from the diagnosed diffusivities (fixed at current step).
        Surface BCs come from MO (treated explicitly as a source term).
        """
        dz = grid.dz
        u_new = cn_solve(y.u, diag.Km, dt_s, dz, S.u, mo_res.u_w)
        v_new = cn_solve(y.v, diag.Km, dt_s, dz, S.v, mo_res.v_w)
        th_new = cn_solve(y.th, diag.Kh, dt_s, dz, S.th, mo_res.w_th)
        qv_new = cn_solve(y.qv, diag.Kh, dt_s, dz, S.qv, mo_res.w_qv)
        # QKE surface flux is always zero (surface BC set in closure)
        qke_new = cn_solve(y.qke, diag.Kq, dt_s, dz, S.qke, 0.0)
        return ProgVarsMYNN(u=u_new, v=v_new, th=th_new, qv=qv_new, qke=qke_new)

    def _cn_warmup(t_s, dt_s, y0):
        """Warmup step: CN with S_prev = S^0 (no previous tendency stored yet).

        AB2 extrapolation degenerates to pure S^0, making this equivalent
        to first-order-accurate forward Euler for the explicit sources.
        """
        S0, diag0, mo_res0 = model(t_s, y0)
        y1 = _apply_cn(y0, S0, diag0, mo_res0, dt_s)
        y1 = clip_state(y1)
        return y1, S0, diag0, mo_res0

    def _cn(t_s, dt_s, y1, S_prev):
        """CN step with AB2-extrapolated non-diffusive explicit sources.

        Explicit sources (Coriolis, QKE budget, advection) are extrapolated
        as S_ab2 = (3/2)*S^n - (1/2)*S^{n-1} before solving the CN system.
        Diffusivities K are taken from the current-step closure diagnostics.
        """
        S1, diag1, mo_res1 = model(t_s, y1)
        S_ab2 = jax.tree_util.tree_map(lambda s1, s0: (3 / 2) * s1 - (1 / 2) * s0, S1, S_prev)
        y2 = _apply_cn(y1, S_ab2, diag1, mo_res1, dt_s)
        y2 = clip_state(y2)
        return y2, S1, diag1, mo_res1

    return _cn_warmup, _cn
