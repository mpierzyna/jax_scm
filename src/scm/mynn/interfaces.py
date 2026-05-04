from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp
from scm.metadata import meta_field


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ProgVarsMYNN:
    """Prognostic variables"""

    u: jnp.ndarray = meta_field(long_name="u velocity", units="m/s", level="full")
    v: jnp.ndarray = meta_field(long_name="v velocity", units="m/s", level="full")
    th: jnp.ndarray = meta_field(
        long_name="potential temperature", units="K", level="full"
    )  # no condensation, so th_l = th compared to NN09
    qv: jnp.ndarray = meta_field(long_name="specific humidity", units="kg/kg", level="full")  # vapor only
    qke: jnp.ndarray = meta_field(long_name="TWICE turbulent kinetic energy", units="m^2/s^2", level="full")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class DiagVarsMYNN:
    # Parameterized fluxes and variances
    u_w: jnp.ndarray = meta_field(long_name="momentum flux (<uw>)", units="m^2/s^2", level="half")
    v_w: jnp.ndarray = meta_field(long_name="momentum flux (<vw>)", units="m^2/s^2", level="half")
    w_th: jnp.ndarray = meta_field(long_name="sensible heat flux", units="K m/s", level="half")
    w_thv: jnp.ndarray = meta_field(long_name="buoyancy flux ", units="K m/s", level="half")
    w_qv: jnp.ndarray = meta_field(long_name="moisture flux", units="kg/kg m/s", level="half")
    th_th: jnp.ndarray = meta_field(long_name="potential temperature variance", units="K^2", level="full")

    # Length scales
    L: jnp.ndarray = meta_field(long_name="turbulent length scale", units="m", level="half")
    L_S: jnp.ndarray = meta_field(long_name="surface length scale", units="m", level="half")
    L_T: jnp.ndarray = meta_field(long_name="turbulent length scale", units="m", level="half")
    L_B: jnp.ndarray = meta_field(long_name="buoyancy length scale", units="m", level="half")

    # Eddy diffusivities
    Km: jnp.ndarray = meta_field(long_name="momentum diffusivity", units="m^2/s", level="half")
    Kh: jnp.ndarray = meta_field(long_name="heat diffusivity", units="m^2/s", level="half")
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MYNNParams:
    """MYNN level-2.5 closure constants (Nakanishi & Niino 2009, eq 66).

    All fields are JAX pytree leaves. For gradient-based optimization, construct
    with ``jnp.array`` values or convert via ``jax.tree_util.tree_map(jnp.asarray, MYNNParams())``.
    """

    A1: float = 1.18
    A2: float = 0.665
    B1: float = 24.0
    B2: float = 15.0
    C1: float = 0.137
    C2: float = 0.75
    C3: float = 0.352
    C4: float = 0.0
    C5: float = 0.2
    gamma1: float = 0.235  # below eq A4, NN09
    g_m_min: float = 1e-12  # numerical stabilizer for G_M denominator
