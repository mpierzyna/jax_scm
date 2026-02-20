from __future__ import annotations

import pathlib
import warnings
from typing import Self
from enum import StrEnum

import pydantic
from scm.config.yaml import yaml_to_dict


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

    @pydantic.model_validator(mode="after")
    def implicit_no_adaptive(self) -> Self:
        """If time_int is implicit, adaptive_timestep must be None."""
        if self.time_int == TimeIntMethod.IMPLICIT and self.adaptive_timestep is not None:
            self.adaptive_timestep = None
            warnings.warn("Implicit time stepping set, ignoring adaptive_timestep config.")
        return self


class AdaptiveTimestepConfig(pydantic.BaseModel):
    cfl_max: float = 0.5  # Max CFL number for diffusion
    dt_s_max: float = 10.0  # Maximum time step, seconds


def load_namelist(f: str | pathlib.Path) -> Namelist:
    """Load namelist from YAML file."""
    f = pathlib.Path(f)
    f_dict = yaml_to_dict(f.read_text())
    return Namelist(**f_dict)
