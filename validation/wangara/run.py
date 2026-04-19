from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

from scm import convert
from scm.config import load_namelist
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.io.local import out_to_ds
from scm.mo import MOSettings
from scm.mynn.interfaces import ProgVarsMYNN
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


def get_wangara_33(Nz: int = 50) -> Simulation:
    ## Grid
    grid = StaggeredGrid(H=2000, Nz=Nz)

    ## Initial conditions
    df = pd.read_csv("ref/day33_0900.csv")
    tk = df["tc"] + 273.15  # Convert to K
    p_hPa = df["p"]
    th = convert.tk_to_th(tk=tk, p_hPa=p_hPa)
    th = np.interp(grid.z, df["z"], th)  # Interpolate to model grid

    u = np.interp(grid.z, df["z"], df["u"])
    v = np.interp(grid.z, df["z"], df["v"])

    init = ProgVarsMYNN(
        u=jnp.array(u),
        v=jnp.array(v),
        th=jnp.array(th),
        qke=0.01 * jnp.ones_like(th),  # small initial turbulence
        qv=jnp.zeros_like(th),  # No initial moisture
    )

    ## Forcing
    # t_s = 0 corresponds to 00 local time
    w_thl_fn = lambda t_s: 2.16e-1 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  # K m/s
    w_qw_fn = lambda t_s: 2.29e-5 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  #  m/s
    dthl_dz_top = 0.0075  # K/m

    u_g = jnp.where(
        grid.z < 1000,
        -5.5 + 2.9e-3 * grid.z,  # linear decrease from -5.5m/s to -2.6m/s at 1000m
        -2.6 + 1.4e-3 * (grid.z - 1000),  # linear decrease from -2.6m/s to -1.2m/s at 2000m
    )
    v_g = jnp.zeros(grid.Nz)

    forcing = Forcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=convert.get_fc(lat_deg=np.abs(-34.5)),  # 34.5°S latitude
        w_th_s=w_thl_fn,
        w_qv_s=w_qw_fn,
        dth_dz_top=dthl_dz_top,
    )

    ## Simulation
    sim = Simulation(
        name="Wangara_Day33",
        grid=grid,
        init=init,
        forcing=forcing,
        th_ref=277.0,  # Pot. temp. close to surface from soundings
        mo_settings=MOSettings(z0m=0.1, z0h=0.1),  # todo: check if agrees with paper
        t_start_s=9 * 3600,
        t_end_s=16 * 3600,
    )
    return sim


def get_ref_ax(
    img_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    trim: Tuple[int, int, int, int] | None = None,
    **fig_kwargs,
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

    fig, ax = plt.subplots(**fig_kwargs)

    # format: [xmin, xmax, ymin, ymax]
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto")

    return fig, ax


def make_report(ds: xr.Dataset, fname: str):

    t_short = ["09:00", "10:00", "12:00", "14:00", "16:00"]
    t_long = [f"1967-08-16T{t}" for t in t_short]
    ds = ds.sel(time=t_long)

    with BaseReport(title="GABLS1 Validation", path=fname) as r:
        r.add_text("This report compares the jax-scm model against Wangara Day 33 reference results from NN09.")

        # Potential temperature
        fig, ax = get_ref_ax(
            "ref/nn09_fig3.png",
            (2, 18),
            (-50, 2050),
            trim=(662, 199, 45, 14),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(5):
            ax.plot(ds["th"].isel(time=i) - 273.15, ds["z"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("Pot. temp, C")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="Potential temperature over time.")

        # Heatflux
        fig, ax = get_ref_ax(
            "ref/nn09_fig4.png",
            (-6e-2, 24e-2),
            (-50, 2050),
            trim=(688, 122, 52, 14),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(1, 5):
            ax.plot(ds["w_thv"].isel(time=i), ds["zh"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("Sensible heat flux, K m / s")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="Sensible heat flux over time")

        # TKE
        fig, ax = get_ref_ax(
            "ref/nn09_fig5.png",
            (0, 3),
            (-50, 2050),
            trim=(678, 117, 43, 16),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(1, 5):
            ax.plot(ds["qke"].isel(time=i) / 2, ds["z"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("TKE, m^2/s^2")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="Turbulent kinetic energy over time")

        # Length scale
        fig, ax = get_ref_ax(
            "ref/nn09_fig7.png",
            (0, 250),
            (-50, 2050),
            trim=(725, 125, 30, 14),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(1, 5):
            ax.plot(ds["L"].isel(time=i), ds["zh"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("Length scale, m")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="MYNN length scale over time")

        # Water vapor
        fig, ax = get_ref_ax(
            "ref/nn09_fig8.png",
            (0, 5),
            (-50, 2050),
            trim=(730, 124, 34, 20),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(5):
            ax.plot(ds["qv"].isel(time=i) * 1000, ds["z"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("Water vapor, g/kg")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="Water vapor over time")

        # Moisture flux
        fig, ax = get_ref_ax(
            "ref/nn09_fig9.png",
            (-2e-5, 8e-5),
            (-50, 2050),
            trim=(738, 127, 36, 17),  # left, bottom, right, top
            figsize=(3, 5),
        )
        for i in range(1, 5):
            ax.plot(ds["w_qv"].isel(time=i), ds["zh"], label=t_short[i], color=f"C{i}")
        ax.set_xlabel("Moisture flux, g/kg")
        ax.set_ylabel("Height, m")
        ax.legend()
        fig.show()
        r.add_mpl_fig(fig, caption="Moisture flux over time")


if __name__ == "__main__":
    # sim = get_wangara_33()
    # cfg = load_namelist("namelist_cn.yaml")
    # model = init_model(sim, cfg)
    # out = simulate(model=model, sim=sim, cfg=cfg)
    # ds = out_to_ds(
    #     out,
    #     sim,
    #     time=pd.date_range(
    #         "1967-08-16T09:00",
    #         freq=f"{cfg.dt_s_out:.0f}s",
    #         periods=out.n_steps,
    #     ),
    # )
    # ds.to_netcdf("wangara_day33.nc")

    ds = xr.open_dataset("wangara_day33.nc")
    make_report(ds, "report.html")
