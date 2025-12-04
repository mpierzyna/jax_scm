from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Callable, Tuple, Literal, Protocol

import jax
import jax.numpy as jnp

from scm import consts

logger = logging.getLogger("scm.mo")
SimFuncType = Callable[[jnp.ndarray], jnp.ndarray]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MOResult:
    """Result of Monin-Obukhov similarity model evaluation."""

    u_st: jnp.ndarray  # Friction velocity at the surface
    w_th: jnp.ndarray  # Sensible heat flux at the surface
    w_q: jnp.ndarray  # Moisture flux at the surface
    L: jnp.ndarray  # Obukhov length
    zeta: jnp.ndarray  # Stability parameter (z/L)
    zeta_err: jnp.ndarray  # Relative error in zeta convergence
    # du_dz: jnp.ndarray  # Vertical gradient of u component
    # dv_dz: jnp.ndarray  # Vertical gradient of v component
    # dth_dz: jnp.ndarray  # Vertical gradient of theta
    # dq_dz: jnp.ndarray  # Vertical gradient of specific humidity
    m10: jnp.ndarray  # 10m wind speed following MOST
    th2: jnp.ndarray  # 2m temperature following MOST
    th_s: jnp.ndarray  # Surface temperature
    u_w: jnp.ndarray  # Surface u-w stress
    v_w: jnp.ndarray  # Surface v-w stress


class MOFunc(Protocol):
    def __call__(
        self,
        *,
        u_0: jnp.ndarray | float,
        v_0: jnp.ndarray | float,
        th_0: jnp.ndarray | float,
        w_q_s: jnp.ndarray | float,
        w_th_s: jnp.ndarray | float | None = None,
        th_s: jnp.ndarray | float | None = None,
    ) -> MOResult: ...


class MOSimilarityFuncs(abc.ABC):
    @abc.abstractmethod
    def get_phi_m_fn(self) -> SimFuncType: ...

    @abc.abstractmethod
    def get_phi_h_fn(self) -> SimFuncType: ...

    @abc.abstractmethod
    def get_psi_m_fn(self) -> SimFuncType: ...

    @abc.abstractmethod
    def get_psi_h_fn(self) -> SimFuncType: ...

    def get_all_fns(self) -> Tuple[SimFuncType, SimFuncType, SimFuncType, SimFuncType]:
        """Convenience function: Get phi_m, phi_h, psi_m, psi_h functions all at once."""
        return self.get_phi_m_fn(), self.get_phi_h_fn(), self.get_psi_m_fn(), self.get_psi_h_fn()


class BusingerDyerSimFuncs(MOSimilarityFuncs):
    """Businger-Dyer flux-gradient relationships.

    References
    ----------
    - Businger
    - Paulson 1970
    """

    def __init__(self, gamma: float = 16, b: float = 5):
        self.gamma = gamma
        self.b = b

    def get_phi_m_fn(self) -> SimFuncType:

        @jax.jit
        def _phi_m(zeta):
            x = jnp.maximum((1 - self.gamma * zeta), 1e-10) ** (1 / 4)  # exponent is different for m and h
            return jnp.where(
                zeta < 0,
                1 / x,  # unstable
                1 + self.b * zeta,  # stable
            )

        return _phi_m

    def get_phi_h_fn(self) -> SimFuncType:

        @jax.jit
        def _phi_h(zeta):
            x = jnp.maximum((1 - self.gamma * zeta), 1e-10) ** (1 / 2)  # exponent is different for m and h
            return jnp.where(
                zeta < 0,
                1 / x,  # unstable
                1 + self.b * zeta,  # stable
            )

        return _phi_h

    def get_psi_m_fn(self) -> SimFuncType:
        @jax.jit
        def _psi_m(zeta):
            x = jnp.maximum((1 - self.gamma * zeta), 1e-10) ** (1 / 4)  # here, exponents are the same for m and h
            return jnp.where(
                zeta < 0,
                2 * jnp.log((1 + x) / 2) + jnp.log((1 + x**2) / 2) - 2 * jnp.atan(x) + jnp.pi / 2,  # unstable
                -self.b * zeta,  # stable
            )

        return _psi_m

    def get_psi_h_fn(self) -> SimFuncType:
        @jax.jit
        def _psi_h(zeta):
            x = jnp.maximum((1 - self.gamma * zeta), 1e-10) ** (1 / 4)  # here, exponents are the same for m and h
            return jnp.where(
                zeta < 0,
                2 * jnp.log((1 + x**2) / 2),  # unstable
                -self.b * zeta,  # stable
            )

        return _psi_h


@jax.jit
def get_L_obukhov(u_st: jnp.ndarray, w_th: jnp.ndarray, th: jnp.ndarray) -> jnp.ndarray:
    """Compute Obukhov length based on friction velocity and surface fluxes.
    For numerical stability, clip L to a reasonable range.
    """
    # Handle neutral case where w_th is zero
    L = jnp.where(w_th == 0, jnp.inf, -(th * u_st**3) / (consts.kappa * consts.g * w_th))
    return L


def init_mo_sfc(
    z0m: float,
    z0h: float,
    z: float,
    sim_funcs: MOSimilarityFuncs,
    prescribe: Literal["w_th_s", "th_s"],
    n_iter: int = 10,
) -> MOFunc:
    """Create a Monin-Obukhov similarity model for surface fluxes.

    Parameters
    ----------
    z0m: float
        Roughness length for momentum, m
    z0h: float
        Roughness length for heat, m
    z: float
        Height at which the model is evaluated, m
        (typically, height of lowest full level)
    sim_funcs: MOSimilarityFuncs
        Similarity functions to use for the model
    prescribe: Literal["w_th_s", "th_s"]
        Flag indicating if surface sensible heat flux (`w_th_s`) or surface temperature (`th_s`) is prescribed.
        The other variable will be computed based on the prescribed one.
    n_iter: int
        Number of iterations to use when solving for stability parameter zeta.

    Returns
    -------
    Callable
        A function that computes surface fluxes based on Monin-Obukhov similarity theory.
        The function signature is:
        `eval_mo(u_0: float, v_0: float, th_0: float, w_th_s: float | None = None, th_s: float | None = None) -> MOResult`
    """
    # Set up similarity functions
    phi_m_fn, phi_h_fn, psi_m_fn, psi_h_fn = sim_funcs.get_all_fns()

    @jax.jit
    def _eval_most(zeta, m_0, th_0, w_th, th_s):
        """Compute u_star and surface heatflux or surface temperature depending on what is prescribed.
        - if "th_s" is prescribed, `w_th` input gets ignored.
        - if "w_th" is prescribed, `th_z` input gets ignored.
        """
        # Evaluate similarity functions
        psi_m = psi_m_fn(zeta)
        psi_m0 = psi_m_fn(zeta * (z0m / z))

        psi_h = psi_h_fn(zeta)
        psi_h0 = psi_h_fn(zeta * (z0h / z))

        # Compute fluxes or surface temperature
        u_st = consts.kappa * m_0 / (jnp.log(z / z0m) - psi_m + psi_m0)
        if prescribe == "th_s":
            # prescribed surface temperature, so compute flux
            th_st = (th_0 - th_s) * consts.kappa / (jnp.log(z / z0h) - psi_h + psi_h0)
            w_th = -th_st * u_st
        elif prescribe == "w_th_s":
            # prescribed surface flux, so compute temperature
            th_st = -w_th / u_st
            th_s = th_0 - (th_st / consts.kappa) * (jnp.log(z / z0h) - psi_h + psi_h0)
        else:
            raise ValueError(f"Invalid mode: {prescribe}. Must be 'th_s' or 'w_st_h'.")

        return u_st, w_th, th_s

    @jax.jit
    def _get_zeta_fixed_iter(m_0, th_0, w_th, th_s) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get zeta using fixed number of iterations."""
        # zeta = jnp.where(w_th != 0.0, -jnp.sign(w_th) * 10.0, 0.0)
        zeta = 0
        zeta_err = jnp.nan

        for i in range(n_iter):
            u_st, w_th, th_s = _eval_most(zeta, m_0, th_0, w_th, th_s)
            L = get_L_obukhov(u_st=u_st, w_th=w_th, th=th_0)
            zeta_new = jnp.clip(z / L, min=-10.0, max=20.0)  # update zeta
            zeta_err = jnp.abs(zeta_new - zeta) / jnp.maximum(jnp.abs(zeta), 1e-10)
            zeta = zeta_new

        return zeta, zeta_err

    @jax.jit
    def _eval_mo(
        *,
        u_0: jnp.ndarray | float,
        v_0: jnp.ndarray | float,
        th_0: jnp.ndarray | float,
        w_q_s: jnp.ndarray | float,
        w_th_s: jnp.ndarray | float | None = None,
        th_s: jnp.ndarray | float | None = None,
    ) -> MOResult:
        """Compute surface fluxes using Monin-Obukhov similarity theory

        Parameters
        ----------
        u_0: float
            u velocity at lowest grid level, m/s
        v_0: float
            v velocity at lowest grid level, m/s
        th_0: float
            theta at lowest grid level, K
        w_q_s: float
            prescribed moisture flux at the surface, (kg/kg) m/s
        w_th_s: float
            prescribed sensible heat flux at the surface, K m/s
            (ignored if `prescribe` not is set to "w_th_s")
        th_s: float
            prescribed surface temperature, K
            (ignored if `prescribe` not is set to "th_s")

        Returns
        -------
        MOResult
            Result of Monin-Obukhov similarity evaluation
        """
        m_0 = jnp.sqrt(u_0**2 + v_0**2)  # wind magnitude
        w_th_s = w_th_s if prescribe == "w_th_s" else 0.0
        th_s = th_s if prescribe == "th_s" else 0.0

        # Solve for zeta given wind speed, temperature, and prescribed flux or surface temperature
        zeta, zeta_err = _get_zeta_fixed_iter(m_0=m_0, th_0=th_0, w_th=w_th_s, th_s=th_s)

        # Evaluate MOST with solved zeta
        u_st, w_th_s, th_s = _eval_most(zeta, m_0, th_0, w_th_s, th_s)
        th_st = -w_th_s / u_st
        # q_st = -w_q_s / u_st

        # Compute stresses at surface. Defined as -uw = ust^2
        u_w_s = -(u_st**2) * u_0 / m_0
        v_w_s = -(u_st**2) * v_0 / m_0

        # Compute gradients based on converged Obukhov length and fluxes
        # Claude suggests to evaluate gradients halfway between surface and lowest full level
        # dm_dz = phi_m_fn(zeta * z_grad / z) * u_st / (consts.kappa * z)
        # dth_dz = phi_h_fn(zeta * z_grad / z) * th_st / (consts.kappa * z)
        # dq_dz = phi_h_fn(zeta * z_grad / z) * q_st / (consts.kappa * z)

        # Split into u and v
        # du_dz = dm_dz * u_0 / m_0
        # dv_dz = dm_dz * v_0 / m_0

        # Compute m10 and th2 as aux outputs
        m10 = u_st * (jnp.log(10 / z0m) - psi_m_fn(zeta * (10 / z)) + psi_m_fn(zeta * (z0m / z))) / consts.kappa
        th2 = th_st * (jnp.log(2 / z0h) - psi_h_fn(zeta * (2 / z)) + psi_h_fn(zeta * (z0h / z))) / consts.kappa + th_s
        L = z / zeta

        return MOResult(
            u_st=u_st,
            w_th=w_th_s,
            w_q=w_q_s,
            L=L,
            zeta=zeta,
            zeta_err=zeta_err,
            # du_dz=du_dz,
            # dv_dz=dv_dz,
            # dth_dz=dth_dz,
            # dq_dz=dq_dz,
            m10=m10,
            th2=th2,
            th_s=th_s,
            u_w=u_w_s,
            v_w=v_w_s,
        )

    return _eval_mo


def _plot_phi_psi_diag(sim_funcs: MOSimilarityFuncs):
    """Diagnostic plots for implementation of similarity functions"""
    import matplotlib.pyplot as plt

    phi_m_fn, phi_h_fn, psi_m_fn, psi_h_fn = sim_funcs.get_all_fns()

    fig, ((ax_phi_m, ax_phi_h), (ax_psi_m, ax_psi_h)) = plt.subplots(ncols=2, nrows=2, sharex="all")

    zeta_space = jnp.linspace(-2, 2, 100)

    ax_phi_m.plot(zeta_space, phi_m_fn(zeta_space))
    ax_phi_h.plot(zeta_space, phi_h_fn(zeta_space))
    ax_psi_m.plot(zeta_space, psi_m_fn(zeta_space))
    ax_psi_h.plot(zeta_space, psi_h_fn(zeta_space))

    ax_phi_m.set_ylabel(r"$\phi_m$")
    ax_phi_h.set_ylabel(r"$\phi_h$")
    ax_psi_m.set_ylabel(r"$\psi_m$")
    ax_psi_h.set_ylabel(r"$\psi_h$")
    ax_psi_m.set_xlabel(r"$\zeta$")
    ax_psi_h.set_xlabel(r"$\zeta$")

    fig.suptitle(sim_funcs.__class__.__name__)
    fig.show()


if __name__ == "__main__":
    _plot_phi_psi_diag(BusingerDyerSimFuncs())
