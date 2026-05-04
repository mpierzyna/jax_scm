from typing import Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scm import consts
from scm.config import load_namelist, Namelist
from scm.examples.andren1994.andren1994 import get_andren1994
from scm.io.local import out_to_ds
from scm.mo import MOSettings
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate

plot_kwargs = {
    "color": "C1",
    "linewidth": 2,
    "label": "jax-scm",
}

scatter_kwargs = {
    "color": "C1",
    "s": 10,
    "label": "jax-scm",
    "zorder": 5,
}


def read_ref_csv(path: str, sort: str = "x") -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read digitized reference CSV (label row, X/Y row, data...). Returns dict of label -> (x, y) sorted by y."""
    raw = pd.read_csv(path, header=None)
    labels = raw.iloc[0].dropna().tolist()
    data = raw.iloc[2:].astype(float)
    result = {}
    for i, label in enumerate(labels):
        x = data.iloc[:, i * 2].dropna().values
        y = data.iloc[:, i * 2 + 1].dropna().values
        if sort == "x":
            order = np.argsort(x)
        else:
            order = np.argsort(y)
        result[label] = (x[order], y[order])
    return result


def make_report(ds: xr.Dataset, fname: str):
    f = ds["frc_f_c"].item()
    tf = ds["time"]  # already in inertial periods (t_s * f_c)
    mo_settings = MOSettings.deserialize(ds.attrs["mo_settings"])

    with BaseReport(title="Andren 1994 Validation", path=fname) as r:
        r.add_text("Comparison of jax-scm against Andren et al. (1994) for neutral boundary layer.")

        # Fig 2: Normalized vertically integrated TKE
        tke_int = np.trapezoid(y=ds["qke"] / 2, x=ds["z"])
        tke_norm = f / ds["mo_u_st"] ** 3 * tke_int

        data = read_ref_csv("ref/a94_fig2.csv")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(tf, tke_norm, **plot_kwargs)

        ax.set_xlabel("tf")
        ax.set_xlim(0, 10)
        ax.set_ylabel(r"$f \int q^2/2 \, dz \; / \; u_*^3$")
        ax.set_yticks(np.arange(0, 1.6, 0.25))
        ax.set_ylim(0, 1.25)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 2: Normalized vertically integrated TKE")

        # Fig 3a: C_u deviation from steady state
        uw0 = ds["mo_u_w"]
        C_u = -f / uw0 * np.trapezoid(y=ds["v"] - ds["frc_v_geo"], x=ds["z"])

        data = read_ref_csv("ref/a94_fig3a.csv")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(tf, C_u, **plot_kwargs)

        ax.set_xlabel("tf")
        ax.set_xlim(0, 10)
        ax.set_ylabel(r"$C_u$")
        ax.set_ylim(0, 1.75)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 3a: C_u deviation from steady state (x-momentum)")

        # Fig 3b: C_v deviation from steady state
        vw0 = ds["mo_v_w"]
        C_v = f / vw0 * np.trapezoid(y=ds["u"] - ds["frc_u_geo"], x=ds["z"])

        data = read_ref_csv("ref/a94_fig3b.csv")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(tf, C_v, **plot_kwargs)
        ax.set_xlabel("tf")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.set_ylabel(r"$C_v$")
        ax.set_ylim(0, 3)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 3b: C_v deviation from steady state (y-momentum)")

        # Time-averaged statistics over last 3/f (Andren 1994)
        ds_sub = ds.sel(time=slice(7.0, None)).mean("time")
        u_st = float(ds_sub["mo_u_st"])

        # Fig 4a: Surface layer phi_m gradient function
        # Add u=v=0 at roughness height for better gradient calculation near the surface
        u = ds_sub["u"].values
        u = np.insert(u, 0, 0)
        v = ds_sub["v"].values
        v = np.insert(v, 0, 0)
        z = ds_sub["z"].values
        z = np.insert(z, 0, mo_settings.z0h)
        zh = np.diff(z) / np.log(z[1:] / z[:-1])  # log-mean height: correct for log-law profiles near surface

        dz = np.diff(z)
        du_dz = (u[1:] - u[:-1]) / dz
        dv_dz = (v[1:] - v[:-1]) / dz
        phi_m = consts.kappa * zh / u_st * np.sqrt(du_dz**2 + dv_dz**2)

        data = read_ref_csv("ref/a94_fig4a.csv", sort="y")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(phi_m, zh * f / u_st, **plot_kwargs)
        ax.axvline(1, color="k", ls="--", lw=1)

        ax.set_xlabel(r"$\Phi_M$")
        ax.set_xlim(0, 2)

        ax.set_ylabel(r"$zf/u_*$")
        ax.set_ylim(0, 0.1)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 4a: Phi_M gradient function in the surface layer")

        # Fig 6a: Normalized u-momentum flux profile
        zh_norm = ds_sub["zh"] * f / u_st
        uw_norm = ds_sub["u_w"] / u_st**2

        data = read_ref_csv("ref/a94_fig6a.csv", sort="y")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(uw_norm, zh_norm, **plot_kwargs)
        ax.axvline(0, color="k", ls="--", lw=1)

        ax.set_xlabel(r"$\overline{uw}/u_*^2$")

        ax.set_ylabel(r"$zf/u_*$")
        ax.set_ylim(0, 0.35)

        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 6a: Normalized u-momentum flux profile")

        # Fig 6b: Normalized v-momentum flux profile
        vw_norm = ds_sub["v_w"] / u_st**2

        data = read_ref_csv("ref/a94_fig6b.csv", sort="y")

        fig, ax = plt.subplots()
        for l, (x, y) in data.items():
            ax.plot(x, y, label=l, color="k", lw=1)
        ax.plot(vw_norm, zh_norm, **plot_kwargs)
        ax.axvline(0, color="k", ls="--", lw=1)

        ax.set_xlabel(r"$\overline{vw}/u_*^2$")
        ax.set_ylabel(r"$zf/u_*$")
        ax.set_ylim(0, 0.35)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 6b: Normalized v-momentum flux profile")


def run(cfg: Namelist, name: str):
    sim = get_andren1994(Nz=100)
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    out_file = f"out_{name}.nc"
    ds.to_netcdf(out_file)
    print(f"Written to {out_file}")

    ds = xr.open_dataset(out_file)
    make_report(ds, fname=f"report_andren1994_{name}.html")


if __name__ == "__main__":
    with jax.enable_x64():
        run(cfg=load_namelist("namelist_cn.yaml"), name="cn")
        run(cfg=load_namelist("namelist_ab2.yaml"), name="ab2")
