"""Prognostic, diagnostic, and gradient variable containers for the MYNN closure scheme."""

from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp

from scm.metadata import meta_field


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ProgVarsMYNN:
    """Prognostic variables"""

    u: jnp.ndarray = meta_field(long_name="U velocity", units="m/s", level="full")
    v: jnp.ndarray = meta_field(long_name="V velocity", units="m/s", level="full")
    th: jnp.ndarray = meta_field(
        long_name="Potential temperature", units="K", level="full"
    )  # no condensation, so th_l = th compared to NN09
    qv: jnp.ndarray = meta_field(long_name="Specific humidity", units="kg/kg", level="full")  # vapor only
    qke: jnp.ndarray = meta_field(long_name="TWICE turbulent kinetic energy", units="m^2/s^2", level="full")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TendsVarsMYNN:
    """Tendencies of prognostic variables"""

    dudt: jnp.ndarray = meta_field(long_name="U velocity tendency", units="m/s^2", level="full")
    dvdt: jnp.ndarray = meta_field(long_name="V velocity tendency", units="m/s^2", level="full")
    dthdt: jnp.ndarray = meta_field(
        long_name="Potential temperature tendency", units="K/s", level="full"
    )  # no condensation, so th_l = th compared to NN09
    dqvdt: jnp.ndarray = meta_field(long_name="Specific humidity tendency", units="kg/kg/s", level="full")  # vapor only
    dqkedt: jnp.ndarray = meta_field(long_name="TWICE turbulent kinetic energy tendency", units="m^2/s^3", level="full")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class DiagVarsMYNN:
    """Turbulence diagnostics produced by the MYNN 2.5 closure.

    All array fields live on half-levels (``Nz+1`` elements) except
    ``th_th``, ``qke_P_S``, ``qke_P_B``, and ``qke_eps``, which are
    averaged to full levels (``Nz`` elements) for use in the tendency
    equations.  Scalar length scales (``L_T``) are domain-integrated
    values returned as 0-d arrays.
    """

    # Parameterized fluxes and variances
    u_w: jnp.ndarray = meta_field(long_name="Momentum flux (<uw>)", units="m^2/s^2", level="half")
    v_w: jnp.ndarray = meta_field(long_name="Momentum flux (<vw>)", units="m^2/s^2", level="half")
    w_th: jnp.ndarray = meta_field(long_name="Sensible heat flux", units="K m/s", level="half")
    w_thv: jnp.ndarray = meta_field(long_name="Buoyancy flux ", units="K m/s", level="half")
    w_qv: jnp.ndarray = meta_field(long_name="Moisture flux", units="kg/kg m/s", level="half")
    th_th: jnp.ndarray = meta_field(long_name="Potential temperature variance", units="K^2", level="full")

    # Length scales
    L: jnp.ndarray = meta_field(long_name="Turbulent length scale", units="m", level="half")
    L_S: jnp.ndarray = meta_field(long_name="Surface length scale", units="m", level="half")
    L_T: jnp.ndarray = meta_field(long_name="Turbulent length scale", units="m", level="half")
    L_B: jnp.ndarray = meta_field(long_name="Buoyancy length scale", units="m", level="half")

    # Eddy diffusivities
    Km: jnp.ndarray = meta_field(long_name="Momentum diffusivity", units="m^2/s", level="half")
    Kh: jnp.ndarray = meta_field(long_name="Heat diffusivity", units="m^2/s", level="half")
    Kq: jnp.ndarray = meta_field(long_name="QKE diffusivity", units="m^2/s", level="half")  # (= L * q * Sq)

    # TKE terms
    w_qke: jnp.ndarray = meta_field(long_name="QKE flux", units="m^3/s^3", level="half")
    qke_P_S: jnp.ndarray = meta_field(long_name="QKE shear production rate", units="m^2/s^3", level="half")
    qke_P_B: jnp.ndarray = meta_field(long_name="QKE buoyancy production rate", units="m^2/s^3", level="half")
    qke_eps: jnp.ndarray = meta_field(long_name="QKE dissipation rate", units="m^2/s^3", level="half")

    # Auxiliary parameters
    ct2: jnp.ndarray = meta_field(long_name="CT2", units="K/m^(2/3)", level="half")


GradVarsMYNN = ProgVarsMYNN
"""Alias for :class:`ProgVarsMYNN` representing vertical gradients of the MYNN prognostic variables.

Each field holds ``Nz+1`` values on half-levels rather than ``Nz`` values on full levels.
The alias avoids duplicating the dataclass definition while keeping call sites self-documenting.
"""
