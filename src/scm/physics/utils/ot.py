"""Utility functions for optical turbulence"""

from __future__ import annotations

from scm import consts
from scm.physics.utils.base import ArrayT


def bowen_ratio(w_th: ArrayT, w_qv: ArrayT) -> ArrayT:
    """Calculate the Bowen ratio.

    Parameters
    ----------
    w_th : jnp.ndarray
        Kinematic sensible heat flux (K m/s).
    w_qv : jnp.ndarray
        Kinematic latent heat flux (kg/kg m/s).

    Returns
    -------
    jnp.ndarray
        Bowen ratio (dimensionless).
    """
    return consts.cp * w_th / (consts.L_v * w_qv)


def cn2_tk(ct2: ArrayT, p: ArrayT, tk: ArrayT, bowen: ArrayT | None) -> ArrayT:
    """Compute Cn2, assuming CT2 is computed from absolute temperature.

    NOT the case for CT2 from MYNN as it is parameterized by POTENTIAL temperature variance.
    See `get_cn2_th` for that case.
    """
    a = 7.9e-5  # K hPa^(-1) for optical wave lenghts
    cn2 = (a * p / tk**2) ** 2 * ct2
    if bowen is not None:
        cn2 = cn2 * (1 + 0.03 / bowen) ** 2
    return cn2


def cn2_th(ct2: ArrayT, p: ArrayT, tk: ArrayT, th: ArrayT, bowen: ArrayT | None) -> ArrayT:
    """UNVALIDATED: Compute Cn2, assuming CT2 is computed from potential temperature.

    Masciadri et al. (2017) mention this modification of the Gladstone equation, which accounts for potential temperature.
    This equation may be the correct to use if CT2 is computed from potential temperature variance, as is the case for MYNN.
    However, this has not been validated.

    References
    ----------
    Masciadri, Elena, et al. “Optical Turbulence Forecast: Ready for an Operational Application.”
    Monthly Notices of the Royal Astronomical Society, vol. 466, no. 1, Apr. 2017, pp. 520–39. arXiv.org,
    https://doi.org/10.1093/mnras/stw3111.


    """
    a = 7.9e-5  # K hPa^(-1) for optical wave lenghts
    cn2 = (a * p / (tk * th)) ** 2 * ct2
    if bowen is not None:
        cn2 = cn2 * (1 + 0.03 / bowen) ** 2
    return cn2
