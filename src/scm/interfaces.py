"""Simulation interfaces.

Concrete types (ProgVarsMYNN, DiagVarsMYNN, etc.) are defined in the closure-specific
module (scm.mynn.interfaces) and imported here so the rest of the codebase has a single
import location. Swapping the closure scheme means updating these imports.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Protocol, Tuple, TypeVar, Union

import jax.numpy as jnp
import jax.tree_util
import pandas as pd

from scm.grid import StaggeredGrid
from scm.metadata import meta_field
from scm.mo import MOResult, MOSettings
from scm.mynn.interfaces import DiagVarsMYNN, GradVarsMYNN, ProgVarsMYNN

ParamsT = TypeVar("ParamsT")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
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

    # Optional function to convert time array to datatime index or scaled index for output.
    t_index_fn: Callable[[jnp.ndarray], Union[pd.DatetimeIndex, jnp.ndarray]] | None = None

    def update_init(
        self,
        *,
        new_t_start_s: int,
        new_init: ProgVarsMYNN = None,
        **new_init_fields: jnp.ndarray,
    ) -> Simulation:
        """Return a copy of this simulation with updated initial conditions.

        Provide either full `new_init` or individual fields which are used to update the existing initial condtions.
        Both options require a `new_t_start_s` between the original `t_start_s` and `t_end_s` to ensure consistency
        with forcing.

        Parameters
        ----------
        new_t_start_s : int
            New simulation start time in seconds.
        new_init : ProgVarsMYNN, optional
            New initial conditions for all prognostic variables.
            If not provided, `new_init_fields` are used to update the existing initial conditions.
        **new_init_fields : jnp.ndarray
            Individual prognostic variable fields to update in the existing initial conditions.
            Ignored if `new_init` is provided.

        Returns
        -------
        Simulation
            A new Simulation object with updated initial conditions and start time.
        """
        # Validate correct start time
        if not (self.t_start_s <= new_t_start_s < self.t_end_s):
            raise ValueError(
                f"`new_t_start_s` must be between original "
                f"`t_start_s` ({self.t_start_s}) and `t_end_s` ({self.t_end_s})."
            )

        if new_init is not None and new_init_fields:
            raise ValueError("Provide either `new_init` or `new_init_fields`, not both.")

        # Simply swap in new initial conditions.
        if new_init is not None:
            return dataclasses.replace(self, init=new_init, t_start_s=new_t_start_s)

        # Update specified fields in the existing initial conditions.
        updated_init = dataclasses.replace(self.init, **new_init_fields)
        return dataclasses.replace(self, init=updated_init, t_start_s=new_t_start_s)

    def update(self, **kwargs) -> Simulation:
        """Convenience method to update any Simulation field EXCEPT init"""
        if "init" in kwargs:
            raise ValueError(
                "Use `update_init` to update initial conditions, which ensures consistency with t_start_s."
            )
        return dataclasses.replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Forcing:
    # Geostrophic wind components
    u_geo: ForceSingleFn = meta_field("u geostrophic wind", "m/s", level="full")  # must return (Nz,)
    v_geo: ForceSingleFn = meta_field("v geostrophic wind", "m/s", level="full")  # must return (Nz,)

    # Coriolis parameter
    f_c: float = meta_field("Coriolis parameter", "1/s", level="full")  # static

    # Surface heat flux or temperature
    w_th_s: ForceSingleFn | None = meta_field(
        long_name="surface potential temperature flux (forcing)",
        units="K m/s",
        level="surface",
        default=None,
    )  # must return scalar

    th_s: ForceSingleFn | None = meta_field(
        long_name="surface potential temperature (forcing)",
        units="K",
        level="surface",
        default=None,
    )  # must return scalar

    # Surface Latent heat flux
    w_qv_s: ForceSingleFn = meta_field(
        long_name="surface specific humidity flux (forcing)",
        units="kg/kg m/s",
        level="surface",
    )  # must return scalar

    # Capping inversion at domain top
    dth_dz_top: float = meta_field(
        long_name="potential temperature gradient at domain top (forcing)",
        units="K/m",
        level="full",
        default=0.01,
    )

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

    def __len__(self) -> int:
        # If time dim is removed, no length
        if self.t_s.ndim == 0:
            return 0
        return len(self.t_s)

    def __getitem__(self, item) -> Output:
        """Subset output"""
        return jax.tree_util.tree_map(lambda x: x[item], self)

    def __iter__(self):
        """Iterate over time steps, yielding Output objects for each time step."""
        if len(self) == 0:
            raise ValueError("Output has no time dimension to iterate over.")

        for i in range(len(self)):
            yield self[i]


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
