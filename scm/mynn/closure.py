from __future__ import annotations

import jax.numpy as jnp

from scm import consts
from scm import convert as conv
from scm.grad import d_dz
from scm.grid import StaggeredGrid
from scm.interfaces import ClosureFn
from scm.mo import MOResult
from scm.mynn.interfaces import ProgVarsMYNN, DiagVarsMYNN

# MYNN closure constants
A1, A2, B1, B2, C1 = 1.18, 0.665, 24.0, 15.0, 0.137  # eq 66, NN09
C2, C3, C4, C5 = 0.75, 0.352, 0.0, 0.2  # eq 66, NN09
gamma1 = 0.235  # below A4, NN09
g_m_min = 1e-12


def get_qke_sfc(u_st: jnp.ndarray) -> jnp.ndarray:
    """Surface boundary condition for QKE (q^2)"""
    return B1 ** (2 / 3) * u_st**2  # MY82, eq. 54


def init_closure(grid: StaggeredGrid) -> ClosureFn[ProgVarsMYNN, DiagVarsMYNN]:
    """MYNN level-2.5 turbulence closure scheme.

    References
    ----------
    [1]_ Nakanishi, Mikio, and Hiroshi Niino. "Development of an Improved Turbulence Closure Model for
    the Atmospheric Boundary Layer." Journal of the Meteorological Society of Japan. Ser. II, vol. 87,
    no. 5, 2009, pp. 895–912. DOI.org (Crossref), https://doi.org/10.2151/jmsj.87.895.

    """

    # def _partial_condensation(wth_l, w_qw, SH25):
    #     """Not in use"""
    #     # Compute liquid water content (ql)
    #     a = (1 + Lv / cp * del_qsl) ** (-1)  # B5, NN09
    #     b = (T / th) * del_qsl  # B6, NN09
    #     sigma_s = jnp.sqrt(
    #         (a**2 * L**2 * alpha_c * B2 * SH25) / 4 * (dqw_dz - b * dthl_dz) ** 2
    #     )  # B7, NN09 (2.5 level version)
    #     q1 = a * (qw - qsl) / (2 * sigma_s)  # B4, NN09
    #     R = 0.5 * (1 + erf(q1 / jnp.sqrt(2)))  # B2, NN09 (cloud fraction)
    #
    #     ql = 2 * sigma_s * (R * q1 + 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(q1**2) / 2))  # B1, NN09
    #
    #     # Compute buoyancy flux w_thv
    #     R_tilde = R - ql / (2 * sigma_s * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(q1**2) / 2)  # B11, NN09
    #     beta_th = 1 + 0.61 * qw - 1.61 * ql - R_tilde * a * b * c  # B5, NN09
    #     beta_q = 0.61 * th + R_tilde * a * c  # B10, NN09
    #     w_thv = beta_th * w_thl + beta_q * w_qw  # B8, NN09
    #
    #     return w_thv, ql

    def _closure(state: ProgVarsMYNN, grads: ProgVarsMYNN, mo_res: MOResult) -> DiagVarsMYNN:
        # In MYNN, qke (q^2) is 2*TKE not specific humidity!
        qke_h = jnp.pad((state.qke[:-1] + state.qke[1:]) / 2, 1, mode="edge")
        qke_h = qke_h.at[0].set(get_qke_sfc(u_st=mo_res.u_st))  # apply surface BC
        q = jnp.sqrt(jnp.clip(qke_h, min=consts.qke_min))  # turbulent velocity scale

        # Virtual potential temperature gradient needed for buoyancy terms
        thv = conv.t_to_tv(t=state.th, qv=state.qv)
        dthv_dz = d_dz(thv, dz=grid.dz, bot="edge", top=grads.th[-1])

        ## Length scale (all on half-levels)
        # Surface length scale (eq 53, NN09)
        zeta = grid.zh / mo_res.L
        L_S = jnp.where(
            zeta < 0,
            consts.kappa * grid.zh * jnp.clip(1 - 100 * zeta, a_min=0.0) ** 0.2,
            jnp.where(
                zeta < 1,
                consts.kappa * grid.zh * (1 + 2.7 * zeta) ** (-1),  # 0 <= zeta < 1
                consts.kappa * grid.zh / 3.7,  # zeta >= 1
            ),
        )

        # Turbulent length scale (eq 54, NN09)
        L_T = 0.23 * jnp.trapezoid(q * grid.zh, grid.zh) / jnp.trapezoid(q, grid.zh)  # todo: maybe better on non-interp

        # Buoyance length scale (eq 55, NN09)
        th_0 = state.th[0]  # todo: where is reference temp?
        N = jnp.sqrt(jnp.clip(consts.g / th_0 * grads.th, a_min=0.0))  # in line after eq 55, NN09
        q_c = jnp.clip((consts.g / th_0) * mo_res.w_thv * L_T, a_min=0.0) ** (1 / 3)  # in line after eq 55, NN09
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

        # todo: check what this does
        G_M = L**2 / q**2 * (grads.u**2 + grads.v**2)  # eq 39, NN09
        # G_H = -(L**2) / q**2 * consts.g / th_0 * (beta_th * grads.th_l + beta_q * grads.q_w)  # eq 40, NN09
        G_H = -(L**2) / q**2 * consts.g / th_0 * dthv_dz  # eq 40, NN09 (virt. pot. temp. version)
        Ri = -G_H / (G_M + g_m_min)  # above eq A11, NN09, gradient Richardson number

        ## Level-2 closure
        gamma2 = (2 * A1 * (3 - 2 * C2) + B2 * (1 - C3)) / B1  # eq A5, NN09
        F1 = B1 * (gamma1 - C1) + 2 * A1 * (3 - 2 * C2) + 3 * A2 * (1 - C2) * (1 - C5)  # eq A6, NN09
        F2 = B1 * (gamma1 + gamma2) - 3 * A1 * (1 - C2)  # eq A7, NN09

        # Flux Richardson number
        Rf1 = B1 * (gamma1 - C1) / F1  # eq A8, NN09
        Rf2 = B1 * gamma1 / F2  # eq A9, NN09
        Rfc = gamma1 / (gamma1 + gamma2)  # eq A10, NN09, critical flux Richardson number

        Ri1 = 0.5 * A2 * F2 / (A1 * F1)
        Ri2 = 0.5 * Rf1 / Ri1
        Ri3 = (2 * Rf2 - Rf1) / Ri1
        Rf = Ri1 * (Ri + Ri2 - (jnp.clip(Ri**2 - Ri3 * Ri + Ri2**2, a_min=0.0)) ** (1 / 2))  # eq A11, NN09

        # Level-2 stability functions
        SH2 = 3 * A2 * (gamma1 + gamma2) * (Rfc - Rf) / (1 - Rf)  # eq A4, NN09
        SM2 = (A1 * F1) / (A2 * F2) * (Rf1 - Rf) / (Rf2 - Rf) * SH2  # eq A3, NN09

        # Diagnosed level-2 tke
        q2_sq = B1 * L**2 * SM2 * (1 - Rf) * (grads.u**2 + grads.v**2)  # eq A2, NN09
        q2 = jnp.sqrt(jnp.clip(q2_sq, a_min=0.0))

        ## Level-2.5 closure
        alpha_c = jnp.where(q < q2, q / q2, 1.0)  # eq 42, NN09

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
        Kq = L * q * Sq  # QKE turbulent transport diffusivity (eq 24, MY82 variant)

        # Simple 1-2-1 filter to dampen vertical oscillations
        # todo: find reference for this
        Km_padded = jnp.pad(Km, 1, mode="edge")
        Km = (Km_padded[:-2] + 2 * Km_padded[1:-1] + Km_padded[2:]) / 4
        Kh_padded = jnp.pad(Kh, 1, mode="edge")
        Kh = (Kh_padded[:-2] + 2 * Kh_padded[1:-1] + Kh_padded[2:]) / 4

        # Parameterized fluxes
        u_w = -Km * grads.u  # eq. 18, NN09
        v_w = -Km * grads.v  # eq. 19, NN09
        w_th = -Kh * grads.th  # sensible heat flux, eq. 20, NN09
        w_thv = -Kh * dthv_dz  # my own assumption for buoyancy flux. NN09 have partial condensation
        w_qv = -Kh * grads.qv  # moisture flux, eq. 21, NN09

        # TKE turbulent transport
        w_qke = Kq * grads.qke  # eq 24, MY82

        # Apply lower boundary conditions from MO before computing TKE budget
        u_w = u_w.at[0].set(mo_res.u_w)
        v_w = v_w.at[0].set(mo_res.v_w)
        w_th = w_th.at[0].set(mo_res.w_th)
        w_thv = w_thv.at[0].set(mo_res.w_thv)
        w_qv = w_qv.at[0].set(mo_res.w_qv)
        w_qke = w_qke.at[0].set(0.0)  # todo: Gemini says this. Confirm!

        # Parameterized dry pot. temp. variance
        lam2 = B2 * L  # eq 12, MY82
        th_th = -lam2 / q * w_th * grads.th  # eq 29, MY82

        # TKE production and dissipation (needed on full levels)
        P_S = -(u_w * grads.u + v_w * grads.v)  # shear production, eq. 5, NN09
        P_S = (P_S[1:] + P_S[:-1]) / 2  # average to full levels

        P_B = consts.g / th_0 * w_thv  # buoyancy production, eq. 5, NN09
        P_B = (P_B[1:] + P_B[:-1]) / 2  # average to full levels

        L_full = (L[1:] + L[:-1]) / 2  # average first to eliminate L=0 at surface leading to div by zero below
        eps = state.qke ** (3 / 2) / (B1 * L_full)  # dissipation, eq. 12, NN09 (q^3 = (q^2)^(3/2))

        # CT2
        # Simplified to avoid NaN from 0 * inf when L=0, since th_th is proportional to L.
        # ct2 = 3.2 * B1 ** (1 / 3) / B2 * L ** (-2 / 3) * th_th
        # th_th = -lam2 / q * th_w * dth_dz, where lam2 = B2 * L
        ct2 = -3.2 * B1 ** (1 / 3) * jnp.clip(L, a_min=0.0) ** (1 / 3) / q * w_th * grads.th

        return DiagVarsMYNN(
            u_w=u_w,
            v_w=v_w,
            w_th=w_th,
            w_thv=w_thv,
            w_qv=w_qv,
            th_th=th_th,
            L=L,
            L_S=L_S,
            L_T=L_T,
            L_B=L_B,
            Km=Km,
            Kh=Kh,
            Kq=Kq,
            w_qke=w_qke,
            qke_P_S=P_S,
            qke_P_B=P_B,
            qke_eps=eps,
            ct2=ct2,
        )

    return _closure
