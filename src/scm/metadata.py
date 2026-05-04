from __future__ import annotations

import dataclasses
from typing import Literal, Type

TLevelLiteral = Literal["full", "half", "surface", "2m", "10m"]


def meta_field(long_name: str, units: str | None, level: TLevelLiteral, **field_kwargs) -> dataclasses.Field:
    """Helper function to create field with variable metadata."""
    meta = {
        "long_name": long_name,
        "units": units if units is not None else "-",  # `None` causes problems with netcdf serialization
        "level": level,
    }
    return dataclasses.field(metadata=meta, **field_kwargs)
