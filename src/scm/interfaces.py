"""Simulation interfaces.

Concrete types (ProgVarsMYNN, DiagVarsMYNN, etc.) are defined in the closure-specific
module (scm.mynn.interfaces) and imported here so the rest of the codebase has a single
import location. Swapping the closure scheme means updating these imports.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Tuple, TypeVar

import jax.numpy as jnp
import jax.tree_util
import pandas as pd

from scm.grid import StaggeredGrid
from scm.mo import MOResult, MOSettings
from scm.mynn.interfaces import ProgVarsMYNN, DiagVarsMYNN, GradVarsMYNN, MYNNParams  # noqa: F401

ParamsT = TypeVar("ParamsT")


@dataclasses.dataclass
class Simulation:
    """Simulation setup container."""

    name: str
    grid: StaggeredGrid
    mo_settings: MOSettings
    init: ProgVarsMYNN
    forcing: Forcing
    th_ref: float  # Reference potential temperature for buoyancy terms (K)

    t_start_s: int
    t_end_s: int
    t_index: pd.DatetimeIndex | None = None  # Optional time index for output


@dataclasses.dataclass(frozen=True, kw_only=True)
class Forcing:
    # Geostrophic wind components
    u_geo: ForceSingleFn  # Unit: m/s; must return (Nz,)
    v_geo: ForceSingleFn  # Unit: m/s; must return (Nz,)

    # Coriolis parameter
    f_c: float  # Unit: (1/s); remains static

    # Surface heat flux or temperature
    w_th_s: ForceSingleFn | None = None  # Unit: (K m/s); must return scalar
    th_s: ForceSingleFn | None = None  # Unit: K, must return scalar

    # Surface Latent heat flux
    w_qv_s: ForceSingleFn  # Unit: (kg/kg m/s); must return scalar

    # Capping inversion at domain top
    dth_dz_top: float = 0.01  # Unit: (K/m)

    # Large scale tendencies
    ls_tends: ForceTendsFn | None = None

    def __post_init__(self):
        if not ((self.w_th_s is None) or (self.th_s is None)):
            raise ValueError("Exactly one of w_th_s and th_s must be provided.")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Output:
    """Simulation output container."""

    state_traj: ProgVarsMYNN
    diag_traj: DiagVarsMYNN
    mo_traj: MOResult
    t_s: jnp.ndarray


class ModelFn(Protocol[ParamsT]):
    def __call__(
        self, t_s: jnp.ndarray, state: ProgVarsMYNN, params: ParamsT
    ) -> Tuple[ProgVarsMYNN, DiagVarsMYNN, MOResult]:
        """Compute tendencies, i.e., right-hand side of ODEs."""


class ClosureFn(Protocol[ParamsT]):
    def __call__(self, state: ProgVarsMYNN, grads: GradVarsMYNN, mo_res: MOResult, params: ParamsT) -> DiagVarsMYNN:
        """Compute closure terms for prognostic variables.

        Parameters
        ----------
        state : ProgVarsMYNN
            Prognostic state on full levels (Nz elements per field).
        grads : GradVarsMYNN
            Vertical gradients on half-levels (Nz+1 elements).
        mo_res : MOResult
            Monin-Obukhov surface-layer result for lower boundary conditions.
        params : ParamsT
            Closure parameters (physical constants or ML weights). Explicit so
            they are visible to ``jax.grad`` for optimization.
        """


class ForceSingleFn(Protocol):
    def __call__(self, t_s: jnp.ndarray) -> jnp.ndarray:
        """Compute time-dependent forcing at time `t_s` for a single variable."""


class ForceTendsFn(Protocol):
    def __call__(self, t_s: jnp.ndarray, state: ProgVarsMYNN, grads: GradVarsMYNN, diag: DiagVarsMYNN) -> ProgVarsMYNN:
        """Compute large-scale tendencies at time `t_s` (seconds after simulation start)."""
