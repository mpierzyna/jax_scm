"""General interfaces are defined here.
Specific interfaces for each model (containing extra variables) should be defined in their respective files.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Tuple, TypeVar, Generic

import jax.numpy as jnp
import jax.tree_util
import pandas as pd

from scm.grid import StaggeredGrid
from scm.mo import MOResult, MOSettings

# Placeholders for concrete implementations of ProgVars and DiagVars per closure scheme
ProgVarsT = TypeVar("ProgVarsT")
DiagVarsT = TypeVar("DiagVarsT")


@dataclasses.dataclass
class Simulation(Generic[ProgVarsT, DiagVarsT]):
    """Simulation container.
    For correct type hinting, return, e.g., `Simulation[ProgVarsMYNN]`
    """

    name: str
    grid: StaggeredGrid
    mo_settings: MOSettings
    init: ProgVarsT
    forcing: Forcing[ProgVarsT, DiagVarsT]

    t_start_s: int
    t_end_s: int
    t_index: pd.DatetimeIndex | None = None  # Optional time index for output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Forcing(Generic[ProgVarsT, DiagVarsT]):
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
    ls_tends: ForceTendsFn[ProgVarsT, DiagVarsT] | None = None

    def __post_init__(self):
        if not ((self.w_th_s is None) or (self.th_s is None)):
            raise ValueError("Exactly one of w_th_s and th_s must be provided.")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Output(Generic[ProgVarsT, DiagVarsT]):
    """Simulation output container."""

    state_hist: ProgVarsT
    diag_hist: DiagVarsT
    mo_hist: MOResult
    time: jnp.ndarray


class ModelFn(Protocol[ProgVarsT, DiagVarsT]):
    def __call__(self, t_s: jnp.ndarray, state: ProgVarsT) -> Tuple[ProgVarsT, DiagVarsT, MOResult]:
        """Compute tendencies, i.e., right-hand side of ODEs."""


class ClosureFn(Protocol[ProgVarsT, DiagVarsT]):
    def __call__(self, state: ProgVarsT, grads: ProgVarsT, mo_res: MOResult) -> DiagVarsT:
        """Compute closure terms for prognostic variables."""


class ForceSingleFn(Protocol):
    def __call__(self, t_s: jnp.ndarray) -> jnp.ndarray:
        """Compute time-dependent forcing at time `t_s` for a single variable.

        Parameters
        ----------
        t_s : jnp.ndarray
            Time in seconds

        Returns
        -------
        jnp.ndarray
            Forcing at time t_s. Must be 1D if forcing for all vertical levels or scalar if surface forcing.
        """


class ForceTendsFn(Protocol[ProgVarsT, DiagVarsT]):
    def __call__(self, t_s: jnp.ndarray, state: ProgVarsT, grads: ProgVarsT, diag: DiagVarsT) -> ProgVarsT:
        """Compute time-dependent forcing directly as tendencies at time `t_s`.

        Parameters
        ----------
        t_s : jnp.ndarray
            Time in seconds AFTER start of simulation.

        Returns
        -------
        ProgVarsT
                Tendencies to be added to model tendencies at time t_s.
        """
