from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ProgVarsMYNN:
    """Prognostic variables"""

    u: jnp.ndarray
    v: jnp.ndarray
    th: jnp.ndarray  # potential temperature (no condensation, so th_l = th)
    qv: jnp.ndarray  # specific humidity (vapor only, no condensation)
    qke: jnp.ndarray  #  qke = q^2 = uu + vv + ww = 2*TKE


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DiagVarsMYNN:
    # Parameterized fluxes and variances
    u_w: jnp.ndarray
    v_w: jnp.ndarray
    w_th: jnp.ndarray  # sensible heat flux
    w_thv: jnp.ndarray  # buoyancy flux (virtual potential temperature flux)
    w_qv: jnp.ndarray  # moisture flux
    th_th: jnp.ndarray  # potential temperature variance

    # Length scales
    L: jnp.ndarray  # turbulent length scale
    L_S: jnp.ndarray  # surface length scale
    L_T: jnp.ndarray  # turbulent length scale
    L_B: jnp.ndarray  # buoyancy length scale

    # Eddy diffusivities
    Km: jnp.ndarray
    Kh: jnp.ndarray
    Kq: jnp.ndarray  # QKE turbulent transport diffusivity (= L * q * Sq)

    # TKE terms
    w_qke: jnp.ndarray  # TKE flux (turbulent transport)
    qke_P_S: jnp.ndarray  # TKE production by shear
    qke_P_B: jnp.ndarray  # TKE production by buoyancy
    qke_eps: jnp.ndarray  # TKE dissipation

    # Auxiliary parameters
    ct2: jnp.ndarray  # temperature structure function coefficient


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
