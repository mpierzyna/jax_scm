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
class DiagVarsMYNN:
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


# Gradients of prognostic variables share the same field structure as ProgVarsMYNN but live
# on half-levels (Nz+1 elements per field) rather than full levels (Nz elements). The alias
# makes call sites self-documenting without duplicating the dataclass definition.
GradVarsMYNN = ProgVarsMYNN
