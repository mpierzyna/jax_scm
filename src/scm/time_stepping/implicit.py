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


def get_cn_sparse_lin_system(
    phi_i: jnp.ndarray,
    K: jnp.ndarray,
    dt: float,
    dz: float,
    S: jnp.ndarray,
    sfc_flux: float | jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get sparse linear system for one CN diffusion step of a single variable.

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
    lhs_dl : jnp.ndarray, shape (Nz,)
        Sub-diagonal of the CN lhs matrix.
    lhs_d : jnp.ndarray, shape (Nz,)
        Diagonal of the CN lhs matrix.
    lhs_du : jnp.ndarray, shape (Nz,)
        Super-diagonal of the CN lhs matrix.
    rhs : jnp.ndarray, shape (Nz,)
        Right-hand side of the CN system, including explicit sources and surface flux.
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
    # for dense computation, leave for clarity
    # A = jnp.diag(A_d) + jnp.diag(A_du[:-1], k=1) + jnp.diag(A_dl[1:], k=-1)
    # I = jnp.eye(N=phi_i.shape[0])

    # Surface-flux explicit source: b_sfc[0] = sfc_flux / dz, zero elsewhere
    b_sfc = jnp.zeros_like(phi_i).at[0].set(sfc_flux / dz)

    # lhs = I - (dt / 2.0) * A  # dense version
    lhs_d = 1 - (dt / 2.0) * A_d
    lhs_dl = -(dt / 2.0) * A_dl
    lhs_du = -(dt / 2.0) * A_du

    # rhs = (I + (dt / 2.0) * A) @ phi_i + dt * S + dt * b_sfc  # dense version
    rhs = (
        phi_i
        + (dt / 2.0) * A_d * phi_i
        + (dt / 2.0) * A_dl * jnp.concatenate([jnp.zeros(1), phi_i[:-1]])  # phi[j-1]
        + (dt / 2.0) * A_du * jnp.concatenate([phi_i[1:], jnp.zeros(1)])  # phi[j+1]
        + dt * S
        + dt * b_sfc
    )

    return lhs_dl, lhs_d, lhs_du, rhs


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

        # Get lhs and rhs of tridiagnal system for all variables
        u_sys = get_cn_sparse_lin_system(y.u, diag.Km, dt_s, dz, S.u, mo_res.u_w)
        v_sys = get_cn_sparse_lin_system(y.v, diag.Km, dt_s, dz, S.v, mo_res.v_w)
        th_sys = get_cn_sparse_lin_system(y.th, diag.Kh, dt_s, dz, S.th, mo_res.w_th)
        qv_sys = get_cn_sparse_lin_system(y.qv, diag.Kh, dt_s, dz, S.qv, mo_res.w_qv)

        # todo: confirm. QKE surface FLUX is always zero. Surface QKE as BC used in closure
        qke_sys = get_cn_sparse_lin_system(y.qke, diag.Kq, dt_s, dz, S.qke, 0.0)

        # Stack to solve in batch
        lhs_dl, lhs_d, lhs_du, rhs = zip(*[u_sys, v_sys, th_sys, qv_sys, qke_sys])
        lhs_dl = jnp.stack(lhs_dl)  # shape (5, Nz)
        lhs_d = jnp.stack(lhs_d)  # shape (5, Nz)
        lhs_du = jnp.stack(lhs_du)  # shape (5, Nz)
        rhs = jnp.stack(rhs)  # shape (5, Nz)
        rhs = rhs[:, :, None]  # shape (5, Nz, 1) for tridiagonal_solve

        # Apply tridiagonal solver
        # Note, I thought batch solving would be faster, but no speed up.
        y_new = tridiagonal_solve(dl=lhs_dl, d=lhs_d, du=lhs_du, b=rhs).squeeze(-1)

        return ProgVarsMYNN(u=y_new[0], v=y_new[1], th=y_new[2], qv=y_new[3], qke=y_new[4])

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
