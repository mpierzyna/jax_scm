from __future__ import annotations

from typing import Tuple

import jax
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
from scm.mynn.interfaces import ProgVarsMYNN, MYNNParams
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate
from scm import consts
from scm import convert

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

    ## Surface and model parameters — defined early so they can be reused below
    mo_settings = MOSettings(z0m=0.01, z0h=0.01)  # Wangara site: flat open paddock, ~0.01 m
    params = MYNNParams()

    ## Initial conditions
    df = pd.read_csv("ref/day33_0900.csv")
    tk = df["tc"] + 273.15  # Convert to K
    p_hPa = df["p"]
    th = convert.tk_to_th(tk=tk, p_hPa=p_hPa)
    th = np.interp(grid.z, df["z"], th)  # Interpolate to model grid

    r = df["r"] / 1000  # kg/kg
    qv = r / (1 + r)
    qv = np.interp(grid.z, df["z"], qv)

    u = np.interp(grid.z, df["z"], df["u"])
    v = np.interp(grid.z, df["z"], df["v"])

    # Estimate surface TKE from initial wind via neutral log-law, then decay exponentially.
    u_st_est = np.sqrt(u[0] ** 2 + v[0] ** 2) * consts.kappa / np.log(grid.z[0] / mo_settings.z0m)
    qke_sfc_est = params.B1 ** (2 / 3) * u_st_est**2
    qke_init = np.maximum(qke_sfc_est * np.exp(-grid.z / 100.0), consts.qke_min)

    init = ProgVarsMYNN(
        u=jnp.array(u),
        v=jnp.array(v),
        th=jnp.array(th),
        qv=jnp.array(qv),
        qke=jnp.array(qke_init),
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
        mo_settings=mo_settings,
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
    t_1400 = "1967-08-16T14:00"
    ds = ds.sel(time=t_long)

    # Single value validation
    zi = ds["zh"].isel(zh=ds["w_thv"].argmin("zh"))  # tab 1 caption
    w_thv_s = ds["mo_w_thv"]
    R = ds["w_thv"].sel(zh=zi) / w_thv_s  # m, tab 1 caption
    w_st = (consts.g / ds.attrs["th_ref"] * w_thv_s * zi) ** (1 / 3)  # m/s, tab 1 caption

    # Prepare 1400 TKE budget
    tke_scale = 2 * w_st**3 / zi  # qke_P_S/B/eps are in QKE (=2*TKE) units; divide by 2 for TKE normalization
    tke_P_S = (ds["qke_P_S"] / tke_scale).sel(time=t_1400)
    tke_P_B = (ds["qke_P_B"] / tke_scale).sel(time=t_1400)
    tke_eps = (ds["qke_eps"] / tke_scale).sel(time=t_1400)

    # Transport term: divergence of q² flux, normalized by tke_scale.
    # w_qke is the q²=2*TKE flux; tke_scale already carries the factor of 2,
    # so ∂w_qke/∂z / tke_scale = ∂w_TKE/∂z / (w_st³/zi) — no extra /2 needed.
    div_w_tke = ds["w_qke"].diff("zh") / ds["zh"].diff("zh")
    div_w_tke = (div_w_tke / tke_scale).sel(time=t_1400)

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
        r.add_mpl_fig(fig, caption="Sensible heat flux over time")
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
        r.add_mpl_fig(fig, caption="Moisture flux over time")

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
        r.add_mpl_fig(fig, caption="Turbulent kinetic energy over time")

        # TKE budget
        fig, ax = get_ref_ax(
            "ref/nn09_fig6.png",
            (-1, 1),
            (-50, 2050),
            trim=(123, 189, 472, 21),  # left, bottom, right, top
            figsize=(4, 4),
        )
        ax.plot(tke_P_S, ds["z"], c="red", lw=1.5)
        ax.plot(tke_P_B, ds["z"], c="red", lw=1.5, ls="--")
        ax.plot(div_w_tke.values, ds["z"].values, c="red", lw=1.5, ls=":")
        ax.plot(-tke_eps, ds["z"], c="red", lw=1.5, ls="-.")  # here negative because sink
        ax.set_xlim(-1, 1)
        r.add_mpl_fig(fig, caption="TKE budget at 14:00")

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
        r.add_mpl_fig(fig, caption="MYNN length scale over time")

        # Table 1
        def _annotate_scatter(ax: plt.Axes, label: str):
            # 1:1 line
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            vmin = min(xmin, ymin)
            vmax = max(xmax, ymax)
            ax.plot([vmin, vmax], [vmin, vmax], "k--")

            # Annotate axis
            ax.set_xlabel(f"Ref {label}")
            ax.set_ylabel(f"jax-scm {label}")

        df = pd.read_csv("ref/nn09_tab1.csv")

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(6, 2), constrained_layout=True)
        c = np.linspace(0, 1, len(df))
        ax1.scatter(df["neg_R"], -R[1:], c=c)
        _annotate_scatter(ax1, "-R, -")
        ax2.scatter(df["zi"], zi[1:], c=c)
        _annotate_scatter(ax2, "zi, m")
        ax3.scatter(df["w_st"], w_st[1:], c=c)
        _annotate_scatter(ax3, "w_st, m/s")
        r.add_mpl_fig(fig, caption="Mixed layer parameters")


def run():
    sim = get_wangara_33(Nz=100)
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(
        out,
        sim,
        time=pd.date_range(
            "1967-08-16T09:00",
            freq=f"{cfg.dt_s_out:.0f}s",
            periods=out.n_steps,
        ),
    )
    ds.to_netcdf("wangara_day33.nc")


if __name__ == "__main__":
    # with jax.enable_x64():
    #     run()
    # run()

    ds = xr.open_dataset("wangara_day33.nc")
    make_report(ds, "report.html")
