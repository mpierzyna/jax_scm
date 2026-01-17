from __future__ import annotations

from jax import numpy as jnp


def d_dz(a: jnp.ndarray, dz: float, bot: jnp.ndarray | float | str, top: jnp.ndarray | float | str) -> jnp.ndarray:
    """Compute vertical gradient of a at half levels using first-order finite differences."""
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
