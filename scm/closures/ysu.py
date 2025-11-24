from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp
import jax.scipy.optimize

import scm
import scm.consts as consts
from scm.interfaces import DiagVars, ClosureFn, ProgVars


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DiagVarsYSU(DiagVars):
    Km: jnp.ndarray  # Horizontal momentum diffusivity
    Kh: jnp.ndarray  # Horizontal heat diffusivity
    h: jnp.ndarray  # Boundary layer height (BLH)

    # Entrainment velocity
    w_e: jnp.ndarray

    # Mixed layer velocity scale at z = h/2
    w_s0: jnp.ndarray

    # Temperature enhancement due to surface buoyancy flux
    th_t: jnp.ndarray

    # Countergradient correction terms
    gamma_u: jnp.ndarray
    gamma_v: jnp.ndarray
    gamma_th: jnp.ndarray
    gamma_q: jnp.ndarray


def init_ysu_closure(grid: scm.grid.StaggeredGrid) -> ClosureFn:
    """Following HND06

    Nomenclature:
    - index a: lowest model level
    - index 0: surface


    References
    ----------
    [1]_ Hong, Song-You, and Hua-Lu Pan. “Nonlocal Boundary Layer Vertical Diffusion in a Medium-Range Forecast Model.”
         Monthly Weather Review, vol. 124, no. 10, Oct. 1996, pp. 2322–39. Monthly Weather Review. journals.ametsoc.org,
         https://doi.org/10.1175/1520-0493(1996)124%253C2322:NBLVDI%253E2.0.CO;2.
    [2]_ https://github.com/wrf-model/WRF/blob/release-v4.5.1/phys/module_bl_ysu.F
    """
    rib_cr_1 = 0.5  # Critical Richardson number (initial blh), inline after eq. 3
    rib_cr_2 = 0.0  # Critical Richardson number (enhanced blh), inline after eq. A12
    a = b = 6.8  # following WRF [2]
    p = 2
    d1, d2, d3 = 0.02, 0.05, 0.001  # constants, inline after eq. A14., d3 from [2]
    lam0 = 150  # m, inline after eq. 17

    zh = grid.zh

    @jax.jit
    def _get_w_s_Pr(h, w_thv_s, u_st, L, th_va, z):
        # Flux profiles evaluated at top of SL (=0.1 h)
        phi_m = jnp.where(
            w_thv_s >= 0,
            (1 - 16 * 0.1 * h / L) ** (-1 / 4),  # unstable / neutral, eq. A5
            (1 + 5 * 0.1 * h / L),  # stable, eq. A6
        )
        phi_t = jnp.where(
            w_thv_s >= 0,
            (1 - 16 * 0.1 * h / L) ** (-1 / 2),  # unstable / neutral, eq. A5
            (1 + 5 * 0.1 * h / L),  # stable, eq. A6
        )

        # Mixed-layer velocity scale for moist air (virtual pot temp)
        w_st_b = ((consts.g / th_va) * w_thv_s * h) ** (1 / 3)  # inline after eq. A2
        w_s = (u_st**3 + phi_m * consts.kappa * w_st_b**3 * z / h) ** (1 / 3)  # eq. A2

        # Prandtl number (function of height)
        eps = 0.1  # inline after eq. A4
        Pr0 = phi_t / phi_m + b * consts.kappa * eps  # inline after eq. A4
        Pr = 1 + (Pr0 - 1) * jnp.exp(-3 * (z - eps * h) ** 2 / h**2)  # eq. A4

        return w_s, Pr

    @jax.jit
    def _get_blh_init(m, th_v, th_va):
        """Initial estimate of boundary layer height."""

        @jax.jit
        def _opt_fn(h, m_h, th_v_h):
            th_T = 0  # for initial blh determination
            th_s = th_va + th_T  # eq. 2
            rhs = rib_cr_1 * th_va * m_h**2 / (consts.g * (th_v_h - th_s))  # eq. 1
            return jnp.abs(h - rhs)

        # Evaluate along entire column to get initial guess
        res = _opt_fn(h=grid.z, m_h=m, th_v_h=th_v)
        i_init = jnp.argmin(res)

        # Refine using interpolation around initial guess
        i_init = jnp.array([-1, 1]) + i_init  # indices around initial guess
        i_a, i_b = jnp.clip(i_init, 0, grid.Nz - 1)  # ensure within bounds

        # h_range = jnp.arange(jnp.floor(grid.z[i_a]), jnp.ceil(grid.z[i_b]), step=1)  # 1m resolution
        h_range = jnp.linspace(grid.z[i_a], grid.z[i_b], num=100)
        m_h = jnp.interp(h_range, grid.z, m)
        th_v_h = jnp.interp(h_range, grid.z, th_v)
        res = _opt_fn(h=h_range, m_h=m_h, th_v_h=th_v_h)
        i_refined = jnp.argmin(res)
        h = h_range[i_refined]

        return h

    @jax.jit
    def _get_blh_enhanced(m, th_v, th_va, th_T):
        """Enhanced estimate of boundary layer height.

        Parameters
        ----------
        m : jnp.ndarray
            Wind speed profile, shape (Nz,).
        th_v : jnp.ndarray
            Virtual potential temperature profile, shape (Nz,).
        th_va : jnp.array
            Virtual potential temperature at lowest model level. Shape (1, ).
        th_T : jnp.array
            Temperature enhancement due to surface buoyancy flux. Shape (1, ).

        Returns
        -------
        h : jnp.ndarray
            Enhanced boundary layer height.
        (i_bot, i_top) : Tuple[jnp.ndarray, jnp.ndarray]
            Indices bracketing the enhanced boundary layer height.
        """

        @jax.jit
        def _get_rib(h, m_h, th_v_h):
            """Bulk Richardson number at heigh h"""
            th_s = th_va + th_T  # eq. 2
            rib = consts.g * (th_v_h - th_s) * h / (th_va * m_h**2)
            return rib

        # Initial guess: evaluate along entire column
        rib_h = _get_rib(h=grid.z, m_h=m, th_v_h=th_v)
        i_top = jnp.searchsorted(-rib_h[::-1], -rib_cr_2) - 1  # find first index where rib_h <= rib_cr_2
        i_top = grid.Nz - 1 - i_top  # flip index back
        i_bot = i_top - 1

        # Select levels around initial guess and interpolate to find h where rib = rib_cr_2
        i_interp = jnp.array([0, 1]) + i_bot  # indices around initial guess
        i_interp = jnp.clip(i_interp, 0, grid.Nz - 1)  # ensure within bounds
        h = jnp.interp(rib_cr_2, rib_h[i_interp], grid.z[i_interp])

        return h, (i_bot, i_top)

    @jax.jit
    def _get_Rig(th_v, dm_dz, dthv_dz):
        """Calculate the gradient Richardson number (non-cloudy layer) on half-levels"""
        # Average th from full to half grid
        thv_mean = jnp.zeros_like(dthv_dz)
        thv_mean = thv_mean.at[1:-1].set((th_v[1:] + th_v[:-1]) / 2)
        thv_mean = thv_mean.at[0].set(th_v[0])  # surface value
        thv_mean = thv_mean.at[-1].set(th_v[-1])  # top value

        return (consts.g / thv_mean) * dthv_dz / dm_dz**2

    @jax.jit
    def _combine_K(K_bl, K_ent, K_loc, i_h_top):
        """Combine eddy diffusivities from boundary layer, entrainment zone, and free atmosphere into one array"""
        idx = jnp.arange(grid.Nz_h)
        is_bl = idx < i_h_top  # boundary layer indices
        is_ent = idx == i_h_top  # entrainment zone index
        is_fa = idx > i_h_top  # free atmosphere indices

        K = jnp.zeros(grid.Nz_h)
        K = jnp.where(is_bl, K_bl, K)
        K = jnp.where(is_ent, (K_ent * K_loc) ** (1 / 2), K)  # eq. A21, geometric mean in entrainment zone
        K = jnp.where(is_fa, K_loc, K)  # noqa: x and y in where provided, so no tuple output

        K = jnp.clip(K, 0.001 * grid.dz, 1000)  # final paragraph App. A

        return K

    @jax.jit
    def _closure(state: ProgVars, grads: ProgVars, mo_res: scm.mo.MOResult) -> DiagVars:
        """Main function"""
        u, v, th, q = state.u, state.v, state.th, state.q
        m = jnp.sqrt(u**2 + v**2)  # magnitude of wind vector
        th_v = th * (1 + 0.61 * q)  # virtual potential temperature profile

        # Gradient of virtual potential temperature needed later
        dthv_dz = jnp.zeros(grid.Nz_h)
        dthv_dz = dthv_dz.at[1:-1].set((th_v[1:] - th_v[:-1]) / grid.dz)
        dthv_dz = dthv_dz.at[0].set(grads.th[0])
        dthv_dz = dthv_dz.at[-1].set(grads.th[-1])

        # Dry and virtual potential temperature at lowest model level
        th_a = th[0]
        th_va = th_v[0]

        u_st, u_w_s, v_w_s, w_th_s, w_q_s, L_ob = mo_res.u_st, mo_res.u_w, mo_res.v_w, mo_res.w_th, mo_res.w_q, mo_res.L
        w_thv_s = w_th_s + 0.61 * th_a * w_q_s  # buoyancy flux at surface (virtual potential temperature)

        ## BLH (z < h)
        # Initial guess for h without th_T
        h = _get_blh_init(m=m, th_v=th_v, th_va=th_va)
        w_s0, _ = _get_w_s_Pr(
            h=h, w_thv_s=w_thv_s, u_st=u_st, L=L_ob, th_va=th_va, z=h / 2
        )  # at z=h/2, inline after (A3)

        # Enhanced h
        th_t = jnp.minimum(b * w_thv_s / w_s0, 3)  # eq. 2, capped at 3K, p. 2320
        h, (i_h_bot, i_h_top) = _get_blh_enhanced(m, th_v, th_va, th_t)

        # Recalculate velocity scales with final h
        w_s, Pr = _get_w_s_Pr(h=h, w_thv_s=w_thv_s, u_st=u_st, L=L_ob, th_va=th_va, z=zh)

        # Eddy diffusivities z < h
        Km_bl = consts.kappa * w_s * zh * (1 - zh / h) ** p  # eq. A1
        Kt_bl = Km_bl / Pr  # correct, see [2] l. 1151

        ## Entrainment zone (z = ca. h)
        # Entrainment fluxes, inline after eq. A8
        w_st_cubed = (consts.g / th_a) * w_th_s * h  # DRY air velocity scale
        w_m = (5 * u_st**3 + w_st_cubed) ** (1 / 3)  # inline after eq. A8
        w_thv_h = -0.15 * (th_va / consts.g) * w_m**3 / h  # eq. A9, bouyancy flux at top of bl (negative)

        w_e = w_thv_h / (th_v[i_h_top] - th_v[i_h_bot])  # eq. A11, Entrainment rate
        w_e = jnp.maximum(w_e, -w_m)  # Limit entrainment rate to magnitude of mixed-layer velocity
        Pr_h = 1  # inline after eq. A11

        w_th_h = w_e * (th[i_h_top] - th[i_h_bot])  # eq. A10a, DRY pot temp!
        w_q_h = w_e * (q[i_h_top] - q[i_h_bot])  # eq. A10b
        u_w_h = Pr_h * w_e * (u[i_h_top] - u[i_h_bot])  # eq. A10c
        v_w_h = Pr_h * w_e * (v[i_h_top] - v[i_h_bot])  # eq. A10d

        # Entrainment zone thickness
        d_thv_con = 0.001 * h  # theoretically, thv[h_idx] - thv[h_idx-1], but practically not. See after eq. A14
        Ri_con = ((consts.g / th_va) * d_thv_con * h) / w_m**2  # inline after eq. A14
        delta = (d1 + d2 / Ri_con) * h  # thickness of entrainment zone

        # Entrainment diffusivities
        x = jnp.exp(-((zh - h) ** 2) / delta**2)
        Kt_ent = -w_thv_h / dthv_dz[i_h_top] * x  # eq. A13a
        Km_ent = Pr_h * Kt_ent  # eq. A13b

        ## Free atmosphere (z > h)
        # Local diffusivities (no clouds)
        dm_dz = jnp.sqrt(grads.u**2 + grads.v**2)  # Mean shear
        Rig = _get_Rig(th_v=th_v, dm_dz=dm_dz, dthv_dz=dthv_dz)  # eq. A16a
        Rig = jnp.clip(Rig, -100, 100)  # todo: goes to inf when dm_dz -> 0, so I clip to +100

        f_t = jnp.where(
            Rig > 0,
            1 / (1 + 5 * Rig) ** 2,  # stable, eq. A18
            1 - ((8 * Rig) / (1 + 1.286 * jnp.sqrt(-Rig))),  # unstable, eq. A20a
        )
        f_m = jnp.where(
            Rig > 0,
            1 / (1 + 5 * Rig) ** 2,  # stable, eq. A18
            1 - ((8 * Rig) / (1 + 1.746 * jnp.sqrt(-Rig))),  # unstable, eq. A20a
        )

        l = (1 / (consts.kappa * zh) + 1 / lam0) ** (-1)  # eq. A17
        Km_loc = l**2 * f_m * dm_dz  # eq. A15, dm_dz is always positive
        Kt_loc = l**2 * f_t * dm_dz  # eq. A15, dm_dz is always positive

        ## Combine
        Km = _combine_K(Km_bl, Km_ent, Km_loc, i_h_top)
        Kt = _combine_K(Kt_bl, Kt_ent, Kt_loc, i_h_top)

        ## Compute countergradient correction
        # Only based on surface values
        gamma_th = b * w_th_s / (w_s0 * h)  # eq. A3
        gamma_q = b * w_q_h / (w_s0 * h)  # eq. A3
        gamma_u = b * u_w_s / (w_s0 * h)  # eq. A3
        gamma_v = b * v_w_s / (w_s0 * h)  # eq. A3

        ## Compute fluxes
        # Everywhere in the bl, local mixing
        u_w = Km * grads.u
        v_w = Km * grads.v
        w_th = Kt * grads.th
        w_q = Kt * grads.q

        # Non-local mixing and entrainment in the bl
        is_bl = zh <= h
        w_th = jnp.where(is_bl, w_th - Kt * gamma_th - w_th_h * (zh / h) ** 3, w_th)
        w_q = jnp.where(is_bl, w_q - Kt * gamma_q - w_q_h * (zh / h) ** 3, w_q)
        u_w = jnp.where(is_bl, u_w - Km * gamma_u - u_w_h * (zh / h) ** 3, u_w)
        v_w = jnp.where(is_bl, v_w - Km * gamma_v - v_w_h * (zh / h) ** 3, v_w)

        # Change sign of all fluxes. This is not stated in HND06, but only way that simulation doesn't blow up.
        u_w = -u_w
        v_w = -v_w
        w_th = -w_th
        w_q = -w_q

        return DiagVarsYSU(
            u_w=u_w,  # noqa: x and y in where provided
            v_w=v_w,  # noqa: x and y in where provided
            w_th=w_th,  # noqa: x and y in where provided
            w_q=w_q,  # noqa: x and y in where provided
            Km=Km,
            Kh=Kt,
            h=h,
            gamma_u=gamma_u,
            gamma_v=gamma_v,
            gamma_th=gamma_th,
            gamma_q=gamma_q,
            w_e=w_e,
            w_s0=w_s0,
            th_t=th_t,
        )

    return _closure
