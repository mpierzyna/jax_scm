from __future__ import annotations

import pathlib
import warnings
from enum import StrEnum
from typing import Self

import pydantic

from scm.config.yaml import yaml_to_dict


class LogLevel(StrEnum):
    """Verbosity for `simulate`."""

    SILENT = "silent"  # nothing is printed
    BOUNDARY = "boundary"  # only "begin" and "complete" messages
    PROGRESS = "progress"  # boundary + per-outer-step progress with ETA


class TimeIntMethod(StrEnum):
    """Time integration method."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class Namelist(pydantic.BaseModel):
    """Main configuration

    Always set default values and add comment to each field for documentation.
    """

    # Implicit or explicit time integration.
    time_int: TimeIntMethod = TimeIntMethod.EXPLICIT

    # If explicit, use adaptive time stepping or constant time step.
    adaptive_timestep: AdaptiveTimestepConfig | None = None

    # Integration time step, seconds
    dt_s: float = 0.01

    # Output time step, seconds
    dt_s_out: float = 5 * 60.0  # 5 mins

    # Verbosity of the simulation loop. SILENT is required when wrapping `simulate`
    # in transforms that disallow host callbacks (e.g. some grad/vmap setups).
    log_level: LogLevel = LogLevel.PROGRESS

    mo_n_iter: int = 10  # Number of iterations for MO solver. Increase for more accuracy, but also more runtime.

    @pydantic.model_validator(mode="after")
    def implicit_no_adaptive(self) -> Self:
        """If time_int is implicit, adaptive_timestep must be None."""
        if self.time_int == TimeIntMethod.IMPLICIT and self.adaptive_timestep is not None:
            self.adaptive_timestep = None
            warnings.warn("Implicit time stepping set, ignoring adaptive_timestep config.")
        return self

    @property
    def is_implicit(self) -> bool:
        return self.time_int == TimeIntMethod.IMPLICIT


class AdaptiveTimestepConfig(pydantic.BaseModel):
    cfl_max: float = 0.5  # Max CFL number for diffusion
    dt_s_max: float = 10.0  # Maximum time step, seconds

    @pydantic.field_validator("cfl_max")
    @classmethod
    def cfl_max_stable_for_ab2(cls, v: float) -> float:
        if v > 0.5:
            raise ValueError(
                f"cfl_max={v} exceeds 0.5, which is outside the AB2 stability region "
                "for diffusion.  Use cfl_max ≤ 0.5 or switch to implicit time stepping."
            )
        return v


def load_namelist(f: str | pathlib.Path) -> Namelist:
    """Load namelist from YAML file."""
    f = pathlib.Path(f)
    f_dict = yaml_to_dict(f.read_text())
    return Namelist(**f_dict)
