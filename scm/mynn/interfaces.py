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
