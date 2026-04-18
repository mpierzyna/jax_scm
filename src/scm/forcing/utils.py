from __future__ import annotations

import warnings
from typing import Dict

import jax
import jax.numpy as jnp

from scm.interfaces import Forcing


def sample_forcing(f: Forcing, t_s: jnp.ndarray) -> Dict[str, jnp.ndarray | None]:
    """Sample forcing at a given time."""

    def _sample(fn, ndim_expected: int) -> jnp.ndarray | None:
        """Expand scalar or 1D output to expected dimensions.
        If, e.g., simulation forced with constant geostrophic wind, forcing may not broadcast to (time, z) shape.
        """
        # Handle missing forcing
        if fn is None:
            return None

        return jax.vmap(fn)(t_s)

        # Sample forcing
        res = fn(t_s)
        if res.ndim == 0 and ndim_expected == 1:
            res = jnp.full_like(t_s, res)
        elif res.ndim == 1 and ndim_expected == 2:
            res = jnp.tile(res, (t_s.size, 1))
        return res

    if f.ls_tends is not None:
        warnings.warn("Sampling of large-scale tendencies not implemented yet. Ignoring ls_tends!")

    return {
        # Expect 2D (time, z) for geostrophic wind
        "u_geo": _sample(f.u_geo, 2),
        "v_geo": _sample(f.v_geo, 2),
        # Expect 1D (time,) for surface forcing
        "w_th_s": _sample(f.w_th_s, 1),
        "th_s": _sample(f.th_s, 1),
        "w_qv_s": _sample(f.w_qv_s, 1),
        # Constants
        "f_c": f.f_c,
        "dth_dz_top": f.dth_dz_top,
    }
