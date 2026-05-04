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
    (I - dt/2 * A + dt/2 * R) φ^{n+1} = (I + dt/2 * A - dt/2 * R) φ^n + dt * S + dt * b_sfc

where A is the tridiagonal diffusion matrix with coefficients (using K_eff):
    A[j, j-1] = K_eff[j] / dz²           (sub-diagonal)
    A[j,   j] = -(K_eff[j] + K_eff[j+1]) / dz²   (diagonal)
    A[j, j+1] = K_eff[j+1] / dz²         (super-diagonal)

and R = diag(decay_rate) is an optional diagonal destruction term (used for
semi-implicit QKE dissipation) and b_sfc[0] = sfc_flux / dz,  b_sfc[j>0] = 0.

Semi-implicit QKE dissipation
------------------------------
The dissipation term eps = q³/(B1·L) = qke^{3/2}/(B1·L) is nonlinear and can
drive instability when the dissipation timescale τ = B1·L/(2·q) is shorter than
the timestep.  Explicit treatment gives qke^{n+1} = qke^n·(1 - dt·r), which goes
negative once dt > τ.

To avoid this, eps is quasi-linearised: eps ≈ r·qke with the rate

    r = qke_eps / qke = 2·sqrt(qke) / (B1·L_full)    [1/s]

evaluated at the current step (lagged, consistent with how K is lagged).  The
semi-implicit form qke^{n+1} = qke^n / (1 + dt·r) is unconditionally positive.

Typical values: near the surface (L~5 m, q~1 m/s) τ ≈ 60 s; at the BL top
(L~50 m, q~0.5 m/s) τ ≈ 1200 s.  The clip to qke_min in the denominator
prevents r from blowing up when qke sits at the numerical floor.

AB2 extrapolation
-----------------
Explicit sources S and surface fluxes (from MO) are both AB2-extrapolated from
the two most recent time levels before being passed to the CN solve.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import ModelFn
from scm.mynn.interfaces import ProgVarsMYNN
from scm.time_stepping.utils import StepCarry, clip_state


def get_cn_sparse_lin_system(
    phi_i: jnp.ndarray,
    K: jnp.ndarray,
    dt: float,
    dz: float,
    S: jnp.ndarray,
    sfc_flux: float | jnp.ndarray,
    decay_rate: float | jnp.ndarray = 0.0,
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
    decay_rate : float or jnp.ndarray, shape () or (Nz,), optional
        Linear destruction rate r such that the implicit decay term is -r*φ.
        Adds (dt/2)*r to the LHS diagonal and subtracts (dt/2)*r*φ from the RHS.
        Default 0 (no implicit decay).  Used for semi-implicit QKE dissipation.

    Returns
    -------
    lhs_dl, lhs_d, lhs_du, rhs : jnp.ndarray, shape (Nz,)
        Tridiagonal system ready for ``tridiagonal_solve``.
    """
    # Zero out boundary faces so they are handled as explicit BCs, not implicit diffusion
    K_eff = K.at[0].set(0.0).at[-1].set(0.0)

    # Sub-diagonal: K at lower face of each cell
    A_dl = K_eff[:-1] / dz**2  # shape (Nz,); a[j] = K_eff[j]
    # Super-diagonal: K at upper face of each cell
    A_du = K_eff[1:] / dz**2  # shape (Nz,); c[j] = K_eff[j+1]
    # Diagonal
    A_d = -(A_dl + A_du)  # shape (Nz,)

    # Surface-flux explicit source: b_sfc[0] = sfc_flux / dz, zero elsewhere
    b_sfc = jnp.zeros_like(phi_i).at[0].set(sfc_flux / dz)

    # CN LHS: I - dt/2 * A + dt/2 * R  (R = diag(decay_rate))
    lhs_d = 1 - (dt / 2.0) * A_d + (dt / 2.0) * decay_rate
    lhs_dl = -(dt / 2.0) * A_dl
    lhs_du = -(dt / 2.0) * A_du

    # CN RHS: (I + dt/2 * A - dt/2 * R) φ^n + dt * S + dt * b_sfc
    rhs = (
        phi_i
        + (dt / 2.0) * A_d * phi_i
        + (dt / 2.0) * A_dl * jnp.concatenate([jnp.zeros(1), phi_i[:-1]])  # phi[j-1]
        + (dt / 2.0) * A_du * jnp.concatenate([phi_i[1:], jnp.zeros(1)])  # phi[j+1]
        - (dt / 2.0) * decay_rate * phi_i
        + dt * S
        + dt * b_sfc
    )

    return lhs_dl, lhs_d, lhs_du, rhs


def get_cn_step_fn(
    model: ModelFn,
    grid: StaggeredGrid,
) -> Tuple[Callable, Callable]:
    """Factory for the semi-implicit Crank-Nicolson stepper (MYNN-specific).

    Returns
    -------
    cn_warmup : Callable
        ``(t_s, dt_s, y0, params) -> StepCarry``
        Builds the initial carry from a raw state.  AB2 degenerates to Euler
        because prev_tends and prev_mo are set equal to the current-step values.
    cn_step : Callable
        ``(carry: StepCarry, t_s, dt_s, params) -> StepCarry``
        Regular CN step.  AB2-extrapolates both explicit sources and MO surface
        fluxes; treats QKE dissipation semi-implicitly on the CN diagonal.
    """

    def _apply_cn(
        y: ProgVarsMYNN,
        S: ProgVarsMYNN,
        diag,
        mo_res,
        dt_s: float,
    ) -> ProgVarsMYNN:
        """Solve one CN step for every prognostic variable.

        K is fixed at the current-step value (lagged).  Surface fluxes are taken
        from ``mo_res`` which has already been AB2-extrapolated by the caller.
        QKE dissipation is treated semi-implicitly via the decay_rate argument.
        """
        dz = grid.dz

        # Semi-implicit QKE decay rate: r = eps/qke = qke^{1/2} / (B1 * L_full)
        qke_decay = diag.qke_eps / jnp.clip(y.qke, min=consts.qke_min)

        u_sys = get_cn_sparse_lin_system(y.u, diag.Km, dt_s, dz, S.u, mo_res.u_w)
        v_sys = get_cn_sparse_lin_system(y.v, diag.Km, dt_s, dz, S.v, mo_res.v_w)
        th_sys = get_cn_sparse_lin_system(y.th, diag.Kh, dt_s, dz, S.th, mo_res.w_th)
        qv_sys = get_cn_sparse_lin_system(y.qv, diag.Kh, dt_s, dz, S.qv, mo_res.w_qv)
        qke_sys = get_cn_sparse_lin_system(y.qke, diag.Kq, dt_s, dz, S.qke, 0.0, decay_rate=qke_decay)

        # Stack to solve in batch
        lhs_dl, lhs_d, lhs_du, rhs = zip(*[u_sys, v_sys, th_sys, qv_sys, qke_sys])
        lhs_dl = jnp.stack(lhs_dl)
        lhs_d = jnp.stack(lhs_d)
        lhs_du = jnp.stack(lhs_du)
        rhs = jnp.stack(rhs)[:, :, None]

        y_new = tridiagonal_solve(dl=lhs_dl, d=lhs_d, du=lhs_du, b=rhs).squeeze(-1)
        return ProgVarsMYNN(u=y_new[0], v=y_new[1], th=y_new[2], qv=y_new[3], qke=y_new[4])

    def cn_warmup(t_s, dt_s, y0, params) -> StepCarry:
        """Build initial carry from a raw state.

        Evaluates the model once and applies one CN step with AB2 degenerated to
        Euler (prev_tends = S0, prev_mo = mo0 so the 3/2 - 1/2 = 1 cancels).
        """
        S0, diag0, mo0 = model(t_s, y0, params)
        y1 = _apply_cn(y0, S0, diag0, mo0, dt_s)
        y1 = clip_state(y1)
        return StepCarry(y=y1, prev_tends=S0, prev_mo=mo0, diag=diag0, mo=mo0)

    def cn_step(carry: StepCarry, t_s, dt_s, params) -> StepCarry:
        """Regular CN step with AB2-extrapolated explicit sources and surface fluxes."""
        S1, diag1, mo1 = model(t_s, carry.y, params)

        # AB2-extrapolate explicit sources and MO surface fluxes to the midpoint
        S_ab2 = jax.tree_util.tree_map(lambda s1, s0: (3 / 2) * s1 - (1 / 2) * s0, S1, carry.prev_tends)
        mo_ab2 = jax.tree_util.tree_map(lambda a, b: (3 / 2) * a - (1 / 2) * b, mo1, carry.prev_mo)

        y2 = _apply_cn(carry.y, S_ab2, diag1, mo_ab2, dt_s)
        y2 = clip_state(y2)
        return StepCarry(y=y2, prev_tends=S1, prev_mo=mo1, diag=diag1, mo=mo1)

    return cn_warmup, cn_step
