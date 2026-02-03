from __future__ import annotations

import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from scm.reporter import BaseReport

plot_kwargs = {
    "color": "C1",
    "linewidth": 2,
    "marker": "o",
    "markevery": 10,
    "label": "jax-scm",
}


def get_ref_ax(
    img_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    trim: Tuple[int, int, int, int] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # Load image and trim if needed
    img = Image.open(img_path)
    if trim is not None:
        w, h = img.size
        left, bottom, right, top = trim
        img = img.crop(
            (
                0 + left,
                0 + top,
                w - right,
                h - bottom,
            )
        )  # (left, top) to  (right, bottom)

    fig, ax = plt.subplots()

    # format: [xmin, xmax, ymin, ymax]
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto")

    return fig, ax


if __name__ == "__main__":
    # res_file = pathlib.Path("out_64.nc")
    # res_file = pathlib.Path("out_128.nc")
    res_file = pathlib.Path("out_400.nc")
    res = xr.open_dataset(res_file)

    t_min = res["time"] * 60  # hours to minutes
    m = np.sqrt(res["u"] ** 2 + res["v"] ** 2)
    tau = (res["u_w"] ** 2 + res["v_w"] ** 2) ** 0.5
    blh = (tau / tau.isel(zh=0)).where(lambda x: x < 0.05).idxmax("zh")  # blh where stress < 5% of surface stress

    with BaseReport(title="GABLS1 Validation", path=f"{res_file.stem}.html") as r:
        r.add_text("This report compares the jax-scm model against GABLS1 reference results from Cuxart et al. (2006).")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig02_blh.png",
            (0, 540),
            (0, 400),
            trim=(134, 100, 36, 19),
        )
        ax.plot(t_min, blh, **plot_kwargs)
        ax.set_xlabel("Time, mins")
        ax.set_ylabel("BLH, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Boundary layer height over time")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig02_ust.png",
            (0, 540),
            (0.2, 0.5),
            trim=(109, 70, 20, 15),
        )
        ax.plot(t_min, res["mo_u_st"], **plot_kwargs)
        ax.set_xlabel("Time, mins")
        ax.set_ylabel("$u_*$, m/s")
        ax.legend()
        r.add_mpl_fig(fig, caption="Friction velocity over time")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig03_m.png",
            (0, 11),
            (0, 400),
            trim=(112, 66, 13, 15),
        )
        ax.plot(m.isel(time=-1), res["z"], **plot_kwargs)
        ax.set_xlabel("Wind speed, m/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind speed profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig03_th.png",
            (262.5, 268),
            (0, 400),
            trim=(114, 81, 21, 9),
        )
        ax.plot(res["th"].isel(time=-1), res["z"], **plot_kwargs)
        ax.set_xlabel(r"$\theta$, K")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig04_hfx.png",
            (-0.03, 0),
            (0, 400),
            trim=(118, 84, 40, 21),
        )
        ax.plot(res["w_th"].isel(time=-1), res["zh"], **plot_kwargs)
        ax.set_xlabel(r"$w'\theta'$, K m/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Sensible heat flux profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig04_momentum.png",
            (0, 0.14),
            (0, 400),
            trim=(133, 84, 59, 32),
        )
        ax.plot(tau.isel(time=-1), res["zh"], **plot_kwargs)
        ax.set_xlabel(r"Momentum flux, m^2/s^2")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Momentum flux profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig06_Km.png",
            (0, 6),
            (0, 400),
            trim=(122, 96, 47, 26),
        )
        ax.plot(res["Km"].isel(time=-1), res["zh"], **plot_kwargs)
        ax.set_xlabel(r"$K_m$, m^2/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Momentum diffusivity profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig06_Kh.png",
            (0, 6),
            (0, 400),
            trim=(118, 96, 55, 21),
        )
        ax.plot(res["Kh"].isel(time=-1), res["zh"], **plot_kwargs)
        ax.set_xlabel(r"$K_h$, m^2/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Heat diffusivity profile at 9h")

        fig, ax = get_ref_ax(
            "ref_cuxart06/fig06_KmKh.png",
            (0, 4),
            (0, 400),
            trim=(132, 116, 49, 33),
        )
        ax.plot((res["Km"] / res["Kh"]).isel(time=-1), res["zh"], **plot_kwargs)
        ax.set_xlabel(r"$K_m/K_h$, -")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Turbulent Prandtl number profile at 9h")
