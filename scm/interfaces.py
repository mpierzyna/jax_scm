"""General interfaces are defined here.
Specific interfaces for each model (containing extra variables) should be defined in their respective files.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Self, TypeVar, Callable, Generic
import dataclasses
import jax.numpy as jnp
import jax.tree_util

from scm.grid import StaggeredGrid
from scm.mo import MOResult


ProgVarsT = TypeVar("ProgVarsT")
DiagVarsT = TypeVar("DiagVarsT")


@dataclasses.dataclass
class Simulation(Generic[ProgVarsT]):
    name: str
    grid: StaggeredGrid
    init: ProgVarsT
    forcing: TransientForcing
    t_start_s: int
    t_end_s: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ProgVars:
    """Prognostic variables"""

    u: jnp.ndarray
    v: jnp.ndarray
    th: jnp.ndarray
    q: jnp.ndarray

    def as_tensor(self) -> jnp.ndarray:
        """Return as tensor where variables are stacked along the first dimension."""
        return jnp.stack(dataclasses.astuple(self), axis=0)

    @classmethod
    def from_tensor(cls, tensor: jnp.ndarray) -> Self:
        """Create ProgVars from a tensor where variables are stacked along the first dimension."""
        return cls(*tensor)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DiagVars:
    """Diagnosed variables, e.g., fluxes, gradients, etc.
    todo: maybe rename to ClosureVars
    """

    u_w: jnp.ndarray  # Horizontal wind stress
    v_w: jnp.ndarray  # Horizontal wind stress
    w_th: jnp.ndarray  # Vertical heat flux
    w_q: jnp.ndarray  # Vertical moisture flux

    def as_tensor(self) -> jnp.ndarray:
        """Return as tensor where variables are stacked along the first dimension."""
        return jnp.stack(dataclasses.astuple(self), axis=0)

    @classmethod
    def from_tensor(cls, tensor: jnp.ndarray) -> Self:
        """Create ProgVars from a tensor where variables are stacked along the first dimension."""
        return cls(*tensor)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StaticForcing:
    """Static forcing using geostrophic wind and surface heat flux or temperature."""

    # Geostrophic wind components
    u_geo: jnp.ndarray  # Unit: m/s; Shape: (1,) or (Nz,)
    v_geo: jnp.ndarray  # Unit: m/s; Shape: (1,) or (Nz,)

    # Coriolis parameter
    f_c: float  # Unit: (1/s); Shape: (1,)

    # Surface heat flux or temperature
    w_th_s: jnp.ndarray | None = None  # Unit: (K m/s); Shape: (1,)
    th_s: jnp.ndarray | None = None  # Unit: (K); Shape: (1,)

    # Latent heat flux
    w_q_s: jnp.ndarray | None = None  # Unit: (kg/kg m/s); Shape: (1,)

    # Capping inversion at domain top
    dth_dz_top: float = 0.01  # Unit: (K/m)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TransientForcing:
    # Geostrophic wind components
    u_geo: ForcingFn  # Unit: m/s; must return (Nz,)
    v_geo: ForcingFn  # Unit: m/s; must return (Nz,)

    # Coriolis parameter
    f_c: float  # Unit: (1/s); remains static

    # Surface heat flux or temperature
    w_th_s: ForcingFn | None = None  # Unit: (K m/s); must return scalar
    th_s: ForcingFn | None = None  # Unit: K, must return scalar

    # Latent heat flux
    w_q_s: ForcingFn  # Unit: (kg/kg m/s); must return scalar

    # Capping inversion at domain top
    dth_dz_top: float = 0.01  # Unit: (K/m)

    def get_eval_fn(self) -> Callable[[jnp.ndarray], StaticForcing]:
        """Evaluate the transient forcing at time t_s to produce a StaticForcing instance."""
        if self.w_th_s is None and self.th_s is None:
            raise ValueError("At least one of w_th_s or th_s must be set.")
        if self.w_th_s is not None and self.th_s is not None:
            raise ValueError("Only one of w_th_s or th_s can be set.")

        @jax.jit
        def _eval_fn(t_s: jnp.ndarray) -> StaticForcing:
            return StaticForcing(
                u_geo=self.u_geo(t_s),
                v_geo=self.v_geo(t_s),
                f_c=self.f_c,
                w_th_s=self.w_th_s(t_s) if self.w_th_s is not None else None,
                th_s=self.th_s(t_s) if self.th_s is not None else None,
                w_q_s=self.w_q_s(t_s),
                dth_dz_top=self.dth_dz_top,
            )

        return _eval_fn




class ModelFn(Protocol[ProgVarsT, DiagVarsT]):
    def __call__(self, state: ProgVarsT, **kwargs) -> Tuple[ProgVarsT, DiagVarsT, MOResult]:
        """Compute tendencies, i.e., right-hand side of ODEs."""


class ClosureFn(Protocol[ProgVarsT, DiagVarsT]):
    def __call__(self, state: ProgVarsT, grads: ProgVarsT, mo_res: MOResult) -> DiagVarsT:
        """Compute closure terms for prognostic variables."""


class ForcingFn(Protocol):
    def __call__(self, t_s: jnp.ndarray) -> jnp.ndarray:
        """Compute time-dependent forcing at time t_s.

        Parameters
        ----------
        t_s : jnp.ndarray
            Time in seconds AFTER start of simulation.

        Returns
        -------
        jnp.ndarray
            Forcing at time t_s. Must be 1D if forcing for all vertical levels or scalar if surface forcing.

        """
