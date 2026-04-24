from __future__ import annotations

import pathlib
import warnings
from enum import StrEnum
from typing import Self

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

    # Whether to print advanced timing information and progress during simulation.
    # Needs to be disabled when `simulate` is jitted, e.g., for ML training.
    # todo: implement timer outside simulate function and only pass callback. Then jitting should work.
    print_advanced_status: bool = True

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
