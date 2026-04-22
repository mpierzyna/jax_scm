from __future__ import annotations

import pathlib

import jax
import xarray as xr
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scm.config import load_namelist, Namelist
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate

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


def make_report(ds: xr.Dataset, fname: str):
    t_min = ds["time"] * 60  # hours to minutes
    m = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    tau = (ds["u_w"] ** 2 + ds["v_w"] ** 2) ** 0.5
    blh = (tau / tau.isel(zh=0)).where(lambda x: x < 0.05).idxmax("zh")  # blh where stress < 5% of surface stress

    with BaseReport(title="GABLS1 Validation", path=fname) as r:
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
        ax.plot(t_min, ds["mo_u_st"], **plot_kwargs)
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
        ax.plot(m.isel(time=-1), ds["z"], **plot_kwargs)
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
        ax.plot(ds["th"].isel(time=-1), ds["z"], **plot_kwargs)
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
        ax.plot(ds["w_th"].isel(time=-1), ds["zh"], **plot_kwargs)
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
        ax.plot(tau.isel(time=-1), ds["zh"], **plot_kwargs)
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
        ax.plot(ds["Km"].isel(time=-1), ds["zh"], **plot_kwargs)
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
        ax.plot(ds["Kh"].isel(time=-1), ds["zh"], **plot_kwargs)
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
        ax.plot((ds["Km"] / ds["Kh"]).isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlabel(r"$K_m/K_h$, -")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Turbulent Prandtl number profile at 9h")


def run(cfg: Namelist, name: str):
    # Run simulation
    sim = get_gabls1(Nz=64, plot=False)
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    out_file = pathlib.Path(f"out_{name}.nc")
    ds = out_to_ds(out=out, sim=sim, time=out.t_s / 60 / 60)
    ds.to_netcdf(out_file)
    print("Written to disk.")

    # Make report
    make_report(ds, fname=f"report_gabls1_{name}.html")


if __name__ == "__main__":
    with jax.enable_x64():
        run(cfg=load_namelist("namelist_cn.yaml"), name="cn")
        run(cfg=load_namelist("namelist_ab2.yaml"), name="ab2")
