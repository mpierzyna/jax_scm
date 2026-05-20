"""Semi-implicit Crank-Nicolson solver for vertical diffusion.

Scheme
------
For a prognostic variable φ on Nz full levels, diffusivity K on Nz+1 half levels,
and an optional linear decay rate r:

    ∂φ/∂t = A φ − r φ + b,    A = ∂/∂z (K ∂/∂z)

Boundary conditions: K_eff[0] = K_eff[Nz] = 0, so both boundary faces carry zero
flux from the implicit operator. The surface flux enters explicitly as b[0] = F_sfc/dz.

Crank-Nicolson (K and r fixed at current step)
-----------------------------------------------
    (I − α·A + α·R) φⁿ⁺¹ = (I + α·A − α·R) φⁿ + dt·b,   α = dt/2

This module exposes three pure primitives that map directly onto that equation:
  build_diffusion_op  — K → TridiagOp (sparse representation of A)
  apply_op            — (A, φ) → A·φ  (matrix-vector product without allocation)
  cn_solve_1d         — one CN step for a single 1-D variable

These are assembled into the MYNN-specific multi-variable stepper by
``get_cn_step_fn``, which vmaps the single-variable solve over all five prognostic
variables and returns the ``(cn_warmup, cn_step)`` pair expected by ``simulate()``.

AB2 extrapolation
-----------------
Explicit sources S and MO surface fluxes are AB2-extrapolated inside ``cn_step``
before being handed to the CN solve. On warmup both history fields equal the
current-step value, so AB2 degenerates to Euler.

Semi-implicit QKE dissipation
------------------------------
ε = qke^{3/2}/(B1·L) is quasi-linearised as r·qke with
    r = ε/qke = √qke/(B1·L)   [s⁻¹]
evaluated at the current step and passed as the ``decay`` argument to ``cn_solve_1d``.
The implicit form φⁿ⁺¹ = φⁿ/(1 + dt·r) is unconditionally positive.

In other words, we factor qke out into (qke_new)*(qke_old)^{1/2}/(B1 L_old), where
the decay rate r = (qke_old)^{1/2}/(B1 L_old) is kept fixed during CN solve and qke_new is solved
implicitly by CN.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import ModelFn
from scm.mynn.interfaces import ProgVarsMYNN
from scm.time_stepping.utils import StepCarry, clip_state


class TridiagOp(NamedTuple):
    """Sparse tridiagonal diffusion operator (sub-diagonal, diagonal, super-diagonal)."""

    dl: jnp.ndarray  # K[j]/dz², shape (Nz,)
    d: jnp.ndarray  # -(K[j]+K[j+1])/dz², shape (Nz,)
    du: jnp.ndarray  # K[j+1]/dz², shape (Nz,)


def build_diffusion_op(K: jnp.ndarray, dz: float) -> TridiagOp:
    """Build the tridiagonal diffusion operator A from half-level diffusivities.

    Zero-flux BCs are imposed by zeroing K at both boundary faces before
    computing coefficients, so neither the bottom surface flux nor the top
    no-flux condition appear in the implicit operator.
    """
    K = K.at[0].set(0.0).at[-1].set(0.0)
    dl = K[:-1] / dz**2
    du = K[1:] / dz**2
    return TridiagOp(dl=dl, d=-(dl + du), du=du)


def apply_op(A: TridiagOp, phi: jnp.ndarray) -> jnp.ndarray:
    """Compute A @ phi without materialising the full matrix."""
    phi_below = jnp.concatenate([jnp.zeros(1), phi[:-1]])
    phi_above = jnp.concatenate([phi[1:], jnp.zeros(1)])
    return A.dl * phi_below + A.d * phi + A.du * phi_above


def cn_solve_1d(
    A: TridiagOp,
    phi: jnp.ndarray,
    dt: float,
    rhs: jnp.ndarray,
    decay: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """One Crank-Nicolson step for a single 1-D variable.

    Solves:  (I − α·A + α·R) φⁿ⁺¹ = (I + α·A − α·R) φⁿ + dt·rhs,   α = dt/2

    Parameters
    ----------
    A     : diffusion operator from build_diffusion_op
    phi   : current state, shape (Nz,)
    dt    : time step [s]
    rhs   : explicit source including any surface-flux term, shape (Nz,)
    decay : linear destruction rate r [s⁻¹], scalar or shape (Nz,)
    """
    alpha = dt / 2.0

    # RHS: (I + α·A − α·R) φⁿ + dt·rhs
    b = phi + alpha * apply_op(A, phi) - alpha * decay * phi + dt * rhs

    # LHS: I − α·A + α·R  (tridiagonal)
    lhs = TridiagOp(
        dl=-alpha * A.dl,
        d=1.0 - alpha * A.d + alpha * decay,
        du=-alpha * A.du,
    )

    return tridiagonal_solve(lhs.dl, lhs.d, lhs.du, b[:, None]).squeeze(-1)


def get_cn_step_fn(
    model: ModelFn,
    grid: StaggeredGrid,
) -> Tuple[Callable, Callable]:
    """Factory for the semi-implicit Crank-Nicolson stepper (MYNN-specific).

    Returns
    -------
    cn_warmup : (t_s, dt_s, y0, params) -> StepCarry
    cn_step   : (carry: StepCarry, t_s, dt_s, params) -> StepCarry
    """
    dz = grid.dz

    def _sfc_rhs(flux: jnp.ndarray, Nz: int) -> jnp.ndarray:
        """Spread a surface flux into the bottom-cell explicit source."""
        return jnp.zeros(Nz).at[0].set(flux / dz)

    def _apply_cn(
        y: ProgVarsMYNN,
        S: ProgVarsMYNN,
        diag,
        mo_res,
        dt_s: float,
    ) -> ProgVarsMYNN:
        """Batched CN solve over all five prognostic variables in one call."""
        Nz = y.u.shape[0]

        phi = jnp.stack([y.u, y.v, y.th, y.qv, y.qke])  # (5, Nz)
        K_vars = jnp.stack([diag.Km, diag.Km, diag.Kh, diag.Kh, diag.Kq])  # (5, Nz+1)
        rhs = jnp.stack(
            [  # (5, Nz)
                S.u + _sfc_rhs(mo_res.u_w, Nz),
                S.v + _sfc_rhs(mo_res.v_w, Nz),
                S.th + _sfc_rhs(mo_res.w_th, Nz),
                S.qv + _sfc_rhs(mo_res.w_qv, Nz),
                S.qke,  # no surface flux for QKE
            ]
        )

        # Factor qke (q^2) out to create quasi-linear decay term for implicit solver.
        qke_decay = diag.qke_eps / jnp.clip(y.qke, min=consts.qke_min)
        decay = jnp.zeros_like(phi).at[4].set(qke_decay)  # (5, Nz)

        ops = jax.vmap(build_diffusion_op, in_axes=(0, None))(K_vars, dz)
        phi_new = jax.vmap(cn_solve_1d, in_axes=(0, 0, None, 0, 0))(ops, phi, dt_s, rhs, decay)  # (5, Nz)

        return ProgVarsMYNN(u=phi_new[0], v=phi_new[1], th=phi_new[2], qv=phi_new[3], qke=phi_new[4])

    def _ab2(a, b):
        return jax.tree_util.tree_map(lambda x, y: 1.5 * x - 0.5 * y, a, b)

    def cn_warmup(t_s, dt_s, y0, params) -> StepCarry:
        S0, diag0, mo0 = model(t_s, y0, params)
        y1 = clip_state(_apply_cn(y0, S0, diag0, mo0, dt_s))
        return StepCarry(y=y1, prev_tends=S0, prev_mo=mo0, diag=diag0, mo=mo0)

    def cn_step(carry: StepCarry, t_s, dt_s, params) -> StepCarry:
        S1, diag1, mo1 = model(t_s, carry.y, params)
        y2 = clip_state(_apply_cn(carry.y, _ab2(S1, carry.prev_tends), diag1, _ab2(mo1, carry.prev_mo), dt_s))
        return StepCarry(y=y2, prev_tends=S1, prev_mo=mo1, diag=diag1, mo=mo1)

    return cn_warmup, cn_step
