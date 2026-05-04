from __future__ import annotations

import jax
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scm import consts
from scm.config import load_namelist, Namelist
from scm.examples.wangara import get_wangara_day33, postproc_wangara
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


def read_ref_csv(path: str) -> dict:
    """Read digitized reference CSV (label row, X/Y row, data...). Returns dict of label -> (x, y) sorted by y."""
    raw = pd.read_csv(path, header=None)
    labels = raw.iloc[0].dropna().tolist()
    data = raw.iloc[2:].astype(float)
    result = {}
    for i, label in enumerate(labels):
        x = data.iloc[:, i * 2].dropna().values
        y = data.iloc[:, i * 2 + 1].dropna().values
        order = np.argsort(y)
        result[label] = (x[order], y[order])
    return result


def _add_ref_legend(ax: plt.Axes):
    """Append a dashed proxy entry for NN09 reference to the existing legend."""
    handles, labels = ax.get_legend_handles_labels()
    proxy_ref = mlines.Line2D([], [], color="k", lw=1.5, ls="--", alpha=0.8, label="NN09 (ref)")
    ax.legend(handles=handles + [proxy_ref], labels=labels + ["NN09 (ref)"], fontsize=7)


def make_report(ds: xr.Dataset, fname: str):
    t_short = ["09:00", "10:00", "12:00", "14:00", "16:00"]
    t_long = [f"1967-08-16T{t}" for t in t_short]
    t_1400 = "1967-08-16T14:00"

    ds = ds.sel(time=t_long)
    ds_pp = postproc_wangara(ds)
    ds_tke_budget = ds_pp.sel(time=t_1400)

    ref_kw = {"linestyle": "--", "linewidth": 1.5, "alpha": 0.8}

    with BaseReport(title="Wangara Day 33 Validation", path=fname) as r:
        r.add_text("This report compares the jax-scm model against Wangara Day 33 reference results from NN09.")

        # Potential temperature
        ref = read_ref_csv("ref/nn09_fig3.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i, t in enumerate(t_short):
            ax.plot(*ref[t], color=f"C{i}", **ref_kw)
            ax.plot(ds["th"].isel(time=i) - 273.15, ds["z"], color=f"C{i}", label=t, lw=1.5)
        ax.set_xlabel("Pot. temp, C")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="Potential temperature over time.")

        # Heatflux
        ref = read_ref_csv("ref/nn09_fig4.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i in range(1, 5):
            ax.plot(*ref[t_short[i]], color=f"C{i}", **ref_kw)
            ax.plot(ds["w_th"].isel(time=i) * 100, ds["zh"], color=f"C{i}", label=t_short[i], lw=1.5)
        ax.set_xlabel("Sensible heat flux, 1e-2 K m / s")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="Sensible heat flux over time")

        # Water vapor
        ref = read_ref_csv("ref/nn09_fig8.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i, t in enumerate(t_short):
            ax.plot(*ref[t], color=f"C{i}", **ref_kw)
            ax.plot(ds["qv"].isel(time=i) * 1000, ds["z"], color=f"C{i}", label=t, lw=1.5)
        ax.set_xlabel("Water vapor, g/kg")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="Water vapor over time")

        # Moisture flux
        ref = read_ref_csv("ref/nn09_fig9.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i in range(1, 5):
            ax.plot(*ref[t_short[i]], color=f"C{i}", **ref_kw)
            ax.plot(ds["w_qv"].isel(time=i) * 1e5, ds["zh"], color=f"C{i}", label=t_short[i], lw=1.5)
        ax.set_xlabel("Moisture flux, 1e-5 g/kg")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="Moisture flux over time")

        # TKE
        ref = read_ref_csv("ref/nn09_fig5.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i in range(1, 5):
            ax.plot(*ref[t_short[i]], color=f"C{i}", **ref_kw)
            ax.plot(ds["qke"].isel(time=i) / 2, ds["z"], color=f"C{i}", label=t_short[i], lw=1.5)
        ax.set_xlabel("TKE, m^2/s^2")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="Turbulent kinetic energy over time")

        # TKE budget — CSV columns: S (shear), B (buoyancy), T+P (transport), D (dissipation)
        ref = read_ref_csv("ref/nn09_fig6.csv")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(*ref["S"], color="C0", **ref_kw)
        ax.plot(*ref["B"], color="C1", **ref_kw)
        ax.plot(*ref["T+P"], color="C2", **ref_kw)
        ax.plot(*ref["D"], color="C3", **ref_kw)
        ax.plot(ds_tke_budget["tke_P_S"].values, ds["z"], color="C0", lw=1.5, label="Shear (S)")
        ax.plot(ds_tke_budget["tke_P_B"].values, ds["z"], color="C1", lw=1.5, label="Buoyancy (B)")
        ax.plot(ds_tke_budget["div_w_tke"], ds["z"].values, color="C2", lw=1.5, label="Transport (T)")
        ax.plot(-ds_tke_budget["tke_eps"], ds["z"], color="C3", lw=1.5, label="Dissipation (-D)")
        ax.set_xlim(-1, 1)
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="TKE budget at 14:00")

        # Length scale
        ref = read_ref_csv("ref/nn09_fig7.csv")
        fig, ax = plt.subplots(figsize=(3, 5))
        for i in range(1, 5):
            ax.plot(*ref[t_short[i]], color=f"C{i}", **ref_kw)
            ax.plot(ds["L"].isel(time=i), ds["zh"], color=f"C{i}", label=t_short[i], lw=1.5)
        ax.set_xlabel("Length scale, m")
        ax.set_ylabel("Height, m")
        _add_ref_legend(ax)
        r.add_mpl_fig(fig, caption="MYNN length scale over time")

        # Surface fluxes vs Hicks (1981) observations
        df_sfc = pd.read_csv("ref/day33_sfc_fluxes_Hicks81.csv")  # todo: probably off
        df_sfc = df_sfc[(df_sfc["Time"] >= 9) & (df_sfc["Time"] <= 16)]
        ref_times = pd.to_datetime([f"1967-08-16T{h:02d}:00" for h in df_sfc["Time"]])
        ref_ust = df_sfc["ust"].values / 100  # cm/s → m/s
        ref_H = df_sfc["H"].values / (consts.rho_0 * consts.cp)  # W/m² → K m/s

        fig, (ax_ust, ax_hfx) = plt.subplots(nrows=2, sharex=True, figsize=(5, 4), constrained_layout=True)
        ax_ust.plot(ds["time"], ds["mo_u_st"], **plot_kwargs)
        ax_ust.plot(ref_times, ref_ust, "ko-", label="Hicks (1981)")
        ax_ust.set_ylabel(r"$u_*$, m/s")
        ax_ust.legend()

        ax_hfx.plot(ds["time"], ds["mo_w_th"], **plot_kwargs)
        ax_hfx.plot(ref_times, ref_H, "ko-", label="Hicks (1981)")
        ax_hfx.set_ylabel(r"$\overline{w'\theta'}$, K m/s")
        ax_hfx.set_xlabel("Time (LST)")
        ax_hfx.legend()
        r.add_mpl_fig(fig, caption="Surface friction velocity and sensible heat flux vs Hicks (1981)")

        # Table 1
        def _annotate_scatter(ax: plt.Axes, label: str):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            vmin = min(xmin, ymin)
            vmax = max(xmax, ymax)
            ax.plot([vmin, vmax], [vmin, vmax], "k--")
            ax.set_xlabel(f"Ref {label}")
            ax.set_ylabel(f"jax-scm {label}")

        df = pd.read_csv("ref/nn09_tab1.csv")

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(6, 2), constrained_layout=True)
        c = np.linspace(0, 1, len(df))
        ax1.scatter(df["neg_R"], -ds_pp["R"][1:], c=c)
        _annotate_scatter(ax1, "-R, -")
        ax2.scatter(df["zi"], ds_pp["zi"][1:], c=c)
        _annotate_scatter(ax2, "zi, m")
        ax3.scatter(df["w_st"], ds_pp["w_st"][1:], c=c)
        _annotate_scatter(ax3, "w_st, m/s")
        r.add_mpl_fig(fig, caption="Mixed layer parameters")


def run(cfg: Namelist, name: str):
    sim = get_wangara_day33(Nz=100)
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    out_file = f"out_{name}.nc"
    ds.to_netcdf(out_file)
    print(f"Written to {out_file}")

    ds = xr.open_dataset(out_file)  # for whatever reason, we need to reopen the file to avoid `VectorIndexing` error.
    make_report(ds, fname=f"report_wangara_{name}.html")


if __name__ == "__main__":
    with jax.enable_x64():
        run(cfg=load_namelist("namelist_cn.yaml"), name="cn")
        run(cfg=load_namelist("namelist_ab2.yaml"), name="ab2")
