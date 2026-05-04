"""Safe math functions with gradient clipping to prevent NaNs.

According to Claude, advantage of defining custom_vjps is that explicit gradient clipping
keeps gradients alive for multiple repeated applications of the safe function whereas simple
clipping of the argument sets gradients to zero.

"""

import jax
import jax.numpy as jnp

GRAD_MIN, GRAD_MAX = -1e6, 1e6


@jax.custom_vjp
def safe_root(x: jnp.ndarray, p: float, eps: float = 1e-6) -> jnp.ndarray:
    """Safe root function with argument and gradient clipping to prevent NaNs in backprop."""
    # Default forward: gets called outside gradient computation
    x_safe = jnp.maximum(x, eps)
    return jnp.pow(x_safe, p)


def safe_root_fwd(x: jnp.ndarray, p: float, eps: float):
    # Forward in gradient computation. Same as default forward.
    x_safe = jnp.maximum(x, eps)
    return safe_root(x_safe, p, eps), (x_safe, p)  # pass x_safe and p as residuals for backward


def safe_root_bwd(res, g):
    # Backward in gradient computation: compute safe gradient
    x_safe, p = res
    raw = g * p * jnp.power(x_safe, p - 1)
    return (jnp.clip(raw, GRAD_MIN, GRAD_MAX), None, None)  # p and eps are non diffable, so return None.


safe_root.defvjp(safe_root_fwd, safe_root_bwd)
