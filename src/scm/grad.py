from __future__ import annotations

from jax import numpy as jnp


def d_dz(a: jnp.ndarray, dz: float, bot: jnp.ndarray | float | str, top: jnp.ndarray | float | str) -> jnp.ndarray:
    """Compute the vertical gradient of ``a`` at half levels.

    Uses first-order centred finite differences between adjacent full levels.
    The boundary values at the bottom and top half levels are supplied
    explicitly or replicated from the nearest interior difference.

    Parameters
    ----------
    a : jnp.ndarray
        Array of shape ``(Nz,)`` defined on full levels.
    dz : float
        Uniform grid spacing (m).
    bot : array-like or float or ``"edge"``
        Gradient value imposed at the bottom half level (``zh[0]``).
        Pass ``"edge"`` to copy the lowest interior difference.
    top : array-like or float or ``"edge"``
        Gradient value imposed at the top half level (``zh[-1]``).
        Pass ``"edge"`` to copy the highest interior difference.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(Nz+1,)`` containing ``da/dz`` at half levels.
    """
    da_dz_inner = (a[1:] - a[:-1]) / dz

    if bot == "edge":
        bot_val = da_dz_inner[0]
    else:
        bot_val = bot

    if top == "edge":
        top_val = da_dz_inner[-1]
    else:
        top_val = top

    return jnp.concatenate([jnp.atleast_1d(bot_val), da_dz_inner, jnp.atleast_1d(top_val)])
