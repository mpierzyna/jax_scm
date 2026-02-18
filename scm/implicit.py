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

import jax
import jax.numpy as jnp


def _build_diffusion_matrix(K_eff: jnp.ndarray, dz: float) -> jnp.ndarray:
    """Build the (Nz x Nz) diffusion matrix A from effective half-level diffusivities.

    Parameters
    ----------
    K_eff : jnp.ndarray, shape (Nz+1,)
        Diffusivity on half levels with boundary faces already zeroed out
        (K_eff[0] = K_eff[-1] = 0 for Neumann BCs).
    dz : float
        Uniform grid spacing.

    Returns
    -------
    A : jnp.ndarray, shape (Nz, Nz)
    """
    # Sub-diagonal: K at lower face of each cell
    a = K_eff[:-1] / dz**2  # shape (Nz,); a[j] = K_eff[j]
    # Super-diagonal: K at upper face of each cell
    c = K_eff[1:] / dz**2   # shape (Nz,); c[j] = K_eff[j+1]
    # Diagonal
    b = -(a + c)             # shape (Nz,)

    A = jnp.diag(b) + jnp.diag(c[:-1], k=1) + jnp.diag(a[1:], k=-1)
    return A


@jax.jit
def cn_solve(
    phi_n: jnp.ndarray,
    K: jnp.ndarray,
    dt: float,
    dz: float,
    S: jnp.ndarray,
    sfc_flux: float | jnp.ndarray,
) -> jnp.ndarray:
    """Solve one Crank-Nicolson diffusion step for a single variable.

    Parameters
    ----------
    phi_n : jnp.ndarray, shape (Nz,)
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
    phi_np1 : jnp.ndarray, shape (Nz,)
        Updated state after one CN step.
    """
    # Zero out boundary faces so they are handled as explicit BCs, not implicit diffusion
    K_eff = K.at[0].set(0.0).at[-1].set(0.0)

    A = _build_diffusion_matrix(K_eff, dz)

    N = phi_n.shape[0]
    I = jnp.eye(N)

    # Surface-flux explicit source: b_sfc[0] = sfc_flux / dz, zero elsewhere
    b_sfc = jnp.zeros(N).at[0].set(sfc_flux / dz)

    lhs = I - (dt / 2.0) * A
    rhs = (I + (dt / 2.0) * A) @ phi_n + dt * S + dt * b_sfc

    phi_np1 = jnp.linalg.solve(lhs, rhs)
    return phi_np1
