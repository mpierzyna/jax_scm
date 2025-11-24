import dataclasses

import jax
import jax.numpy as jnp

from scm import consts
from scm.grid import StaggeredGrid
from scm.interfaces import DiagVars, ClosureFn, ProgVars
from scm.mo import MOResult


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ProgVarsMYNN(ProgVars):
    """Prognostic variables"""

    u: jnp.ndarray
    v: jnp.ndarray
    th_l: jnp.ndarray  # liquid water potential temperature
    q_w: jnp.ndarray  # total water content q_w = q_l + q_v (liquid + vapor)
    q: jnp.ndarray  #  2 * tke


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DiagVarsMYNN(DiagVars):
    L: jnp.ndarray  # turbulent length scale


def init_mynn(grid: StaggeredGrid) -> ClosureFn:
    """
    References
    ----------
    [1]_ Nakanishi, Mikio, and Hiroshi Niino. “Development of an Improved Turbulence Closure Model for the Atmospheric Boundary Layer.” Journal of the Meteorological Society of Japan. Ser. II, vol. 87, no. 5, 2009, pp. 895–912. DOI.org (Crossref), https://doi.org/10.2151/jmsj.87.895.

    """
    # MYNN closure constants
    A1, A2, B1, B2, C1 = 1.18, 0.665, 24.0, 15.0, 0.137  # eq 66, NN09
    C2, C3, C4, C5 = 0.75, 0.352, 0.0, 0.2  # eq 66, NN09

    def _closure(state: ProgVarsMYNN, grads: ProgVarsMYNN, mo_res: MOResult) -> DiagVarsMYNN:
        # in MYNN, q is 2*TKE not specific humidity!
        u, v, th_l, q = state.u, state.v, state.th_l, state.q
        th_v = th_l  # todo: virtual potential temperature
        dthv_dz = th_v  # todo: virtual potential temperature gradient

        ## Length scale
        # Surface length scale (eq 53, NN09)
        zeta = grid.z / mo_res.L
        L_S = jnp.where(
            zeta < 0,
            consts.kappa * grid.z * (1 - 100 * zeta) ** 0.2,
            jnp.where(
                zeta < 1,
                consts.kappa * grid.z * (1 + 2.7 * zeta) ** (-1),  # 0 <= zeta < 1
                consts.kappa * grid.z / 3.7,  # zeta >= 1
            ),
        )

        # Turbulent length scale (eq 54, NN09)
        L_T = 0.23 * jnp.trapezoid(q * grid.z, grid.z) / jnp.trapezoid(q, grid.z)

        # Buoyance length scale (eq 55, NN09)
        N = jnp.sqrt(consts.g / th_0 * dthv_dz)
        q_c = ((consts.g / th_0) * w_thv_0 * L_T) ** (1 / 3)  # in line after eq 55, NN09
        L_B = jnp.where(
            dthv_dz <= 0,
            jnp.inf,
            jnp.where(
                zeta >= 0,
                q / N,
                (1 + 5 * (q_c / L_T * N) ** (1 / 2)) * q / N,
            ),
        )

        # Final length scale
        L = (1 / L_S + 1 / L_T + 1 / L_B) ** -1

        ## Stability functions
        alpha_c = jnp.where(q < q2, q / q2, 1.0)  # eq 42, NN09

        G_M = L**2 / q**2 * (grads.u**2 + grads.v**2)  # eq 39, NN09
        G_H = -(L**2) / q**2 * consts.g / th_0 * (beta_th * grads.th_l + beta_q * grads.q_w)  # eq 40, NN09

        phi_1 = 1 - 3 * alpha_c**2 * A2 * B2 * (1 - C3) * G_H  # eq 33, NN09
        phi_2 = 1 - 9 * alpha_c**2 * A1 * A2 * (1 - C2) * G_H  # eq 34, NN09
        phi_3 = phi_1 + 9 * alpha_c**2 * A2**2 * (1 - C2) * (1 - C5) * G_H  # eq 35, NN09
        phi_4 = phi_1 - 12 * alpha_c**2 * A1 * A2 * (1 - C2) * G_H  # eq 36, NN09
        phi_5 = 6 * alpha_c**2 * A1**2 * G_M  # eq 37, NN09

        D25 = phi_2 * phi_4 + phi_5 * phi_3  # eq 31, NN09
        SM25 = alpha_c * A1 * (phi_3 - 3 * C1 * phi_4) / D25  # eq 27, NN09
        SH25 = alpha_c * A2 * (phi_2 + 3 * C1 * phi_5) / D25  # eq 28, NN09

        # level-3 model (not needed)
        # D_prime = phi_2 * (phi_4 - phi_1 + 1) + phi_5 * (phi_3 - phi_1 + 1)  # eq 32, NN09
        # C_theta = thl_thl / (L**2 * grads.th_l**2)  # eq 41, NN09
        # phi_prime = 3 * (1 - C3) * G_H * (C_theta - alpha_c * B2 * SH25)  # eq 38, NN09
        # SM_prime = alpha_c * A1 * (phi_3 - phi_4) / D_prime * phi_prime  # eq 29, NN09
        # SH_prime = alpha_c * A2 * (phi_2 + phi_5) / D_prime * phi_prime  # eq 30, NN09

        # SM = SM25 + SM_prime
        # SH = SH25 + SH_prime

        SM = SM25
        SH = SH25
        Sq = 3 * SM  # eq 67, NN09

        # Eddy diffusivities
        Km = L * q * SM
        Kh = L * q * SH

        u_w = -Km * grads.u
        v_w = -Km * grads.v
        thl_w = -Kh * grads.th_l

        # TKE production and dissipation
        P_S = -(u_w * grads.u + v_w * grads.v)  # shear production, eq. 5, NN09
        P_B = consts.g / th_0 * thv_w  # buoyancy production, eq. 5, NN09
        eps = q**3 / (B1 * L)  # dissipation, eq. 12, NN09

        # TKE equaion: D(q^2/2)Dt = d/dz[L q Sq d/dz(q^2/2)] + P_S + P_B - eps
        q_w = L * q * Sq * grads.q  # eq 24, MY82, turbulent transport  # todo, sign?
        q_frc = P_S + P_B - eps  # source terms

    return _closure
