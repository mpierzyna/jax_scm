from __future__ import annotations

import dataclasses
from typing import Literal

TLevelLiteral = Literal["full", "half", "surface", "2m", "10m"]


def meta_field(long_name: str, units: str | None, level: TLevelLiteral, **field_kwargs) -> dataclasses.Field:
    """Create a :func:`dataclasses.field` with embedded variable metadata.

    The ``metadata`` dict attached to the returned field contains
    ``long_name``, ``units``, and ``level`` keys, which are consumed by
    ``out_to_ds()`` when writing xarray / NetCDF output.

    Parameters
    ----------
    long_name : str
        Human-readable variable name (CF ``long_name`` convention).
    units : str or None
        Physical units string.  ``None`` is stored as ``"-"`` to avoid
        NetCDF serialisation issues.
    level : TLevelLiteral
        Vertical coordinate type: ``"full"``, ``"half"``, ``"surface"``,
        ``"2m"``, or ``"10m"``.
    **field_kwargs
        Additional keyword arguments forwarded to :func:`dataclasses.field`
        (e.g. ``default``, ``default_factory``).

    Returns
    -------
    dataclasses.Field
        A dataclass field descriptor with the metadata dict populated.
    """
    meta = {
        "long_name": long_name,
        "units": units if units is not None else "-",  # `None` causes problems with netcdf serialization
        "level": level,
    }
    return dataclasses.field(metadata=meta, **field_kwargs)
