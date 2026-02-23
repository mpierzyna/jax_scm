from typing import Dict

import jax.numpy as jnp

from scm.interfaces import Forcing


def sample_forcing(f: Forcing, t_s: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Sample forcing at a given time."""
    return {
        "u_geo": f.u_geo(t_s),
        "v_geo": f.v_geo(t_s),
        "f_c": f.f_c,
        "w_th_s": f.w_th_s(t_s) if f.w_th_s is not None else None,
        "th_s": f.th_s(t_s) if f.th_s is not None else None,
        "w_qv_s": f.w_qv_s(t_s),
        "dth_dz_top": f.dth_dz_top,
    }
