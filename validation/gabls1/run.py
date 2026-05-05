from __future__ import annotations

import pathlib
from typing import Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scm.config import Namelist, load_namelist
from scm.examples.gabls1 import get_gabls1, postproc_gabls1
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


def read_ref_csv(path: str, sort: str = "x") -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read digitized reference CSV (label row, X/Y row, data...). Returns dict of label -> (x, y) sorted by x or y."""
    raw = pd.read_csv(path, header=None)
    labels = raw.iloc[0].dropna().tolist()
    data = raw.iloc[2:].astype(float)
    result = {}
    for i, label in enumerate(labels):
        x = data.iloc[:, i * 2].dropna().values
        y = data.iloc[:, i * 2 + 1].dropna().values
        order = np.argsort(x) if sort == "x" else np.argsort(y)
        result[label] = (x[order], y[order])
    return result


def make_report(ds: xr.Dataset, fname: str):
    t_min = ds["time"] * 60  # hours to minutes
    ds_pp = postproc_gabls1(ds)

    ref_kw = {"color": "k", "lw": 1, "alpha": 0.6}

    with BaseReport(title="GABLS1 Validation", path=fname) as r:
        r.add_text("This report compares the jax-scm model against GABLS1 reference results from Cuxart et al. (2006).")

        ref = read_ref_csv("ref_cuxart06/fig02_blh.csv", sort="x")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(t_min, ds_pp["blh"], **plot_kwargs)
        ax.set_xlim(0, 540)
        ax.set_ylim(0, 400)
        ax.set_xlabel("Time, mins")
        ax.set_ylabel("BLH, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Boundary layer height over time")

        ref = read_ref_csv("ref_cuxart06/fig02_ust.csv", sort="x")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(t_min, ds["mo_u_st"], **plot_kwargs)
        ax.set_xlim(0, 540)
        ax.set_ylim(0.2, 0.5)
        ax.set_xlabel("Time, mins")
        ax.set_ylabel("$u_*$, m/s")
        ax.legend()
        r.add_mpl_fig(fig, caption="Friction velocity over time")

        ref = read_ref_csv("ref_cuxart06/fig03_m.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds_pp["m"].isel(time=-1), ds["z"], **plot_kwargs)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 400)
        ax.set_xlabel("Wind speed, m/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind speed profile at 9h")

        ref = read_ref_csv("ref_cuxart06/fig03_th.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds["th"].isel(time=-1), ds["z"], **plot_kwargs)
        ax.set_xlim(262.5, 268)
        ax.set_ylim(0, 400)
        ax.set_xlabel(r"$\theta$, K")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 9h")

        ref = read_ref_csv("ref_cuxart06/fig04_hfx.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds["w_th"].isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlim(-0.03, 0)
        ax.set_ylim(0, 400)
        ax.set_xlabel(r"$w'\theta'$, K m/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Sensible heat flux profile at 9h")

        ref = read_ref_csv("ref_cuxart06/fig04_momentum.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds_pp["tau"].isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlim(0, 0.14)
        ax.set_ylim(0, 400)
        ax.set_xlabel(r"Momentum flux, m^2/s^2")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Momentum flux profile at 9h")

        ref = read_ref_csv("ref_cuxart06/fig06_Km.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds["Km"].isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 400)
        ax.set_xlabel(r"$K_m$, m^2/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Momentum diffusivity profile at 9h")

        ref = read_ref_csv("ref_cuxart06/fig06_Kh.csv", sort="y")
        fig, ax = plt.subplots()
        for x, y in ref.values():
            ax.plot(x, y, **ref_kw)
        ax.plot(ds["Kh"].isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 400)
        ax.set_xlabel(r"$K_h$, m^2/s")
        ax.set_ylabel("Height, m")
        ax.legend()
        r.add_mpl_fig(fig, caption="Heat diffusivity profile at 9h")

        fig, ax = plt.subplots()
        ax.plot((ds["Km"] / ds["Kh"]).isel(time=-1), ds["zh"], **plot_kwargs)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 400)
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
    ds = out_to_ds(out=out, sim=sim)
    ds.to_netcdf(out_file)
    print("Written to disk.")

    # Make report
    make_report(ds, fname=f"report_gabls1_{name}.html")


if __name__ == "__main__":
    with jax.enable_x64():
        run(cfg=load_namelist("namelist_cn.yaml"), name="cn")
        run(cfg=load_namelist("namelist_ab2.yaml"), name="ab2")
