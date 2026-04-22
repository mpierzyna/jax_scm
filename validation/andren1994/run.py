from typing import Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from PIL import Image
from scm.config import load_namelist
from scm.examples.andren1994.andren1994 import get_andren1994
from scm.io.local import out_to_ds
from scm.mo import MOSettings
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate
from scm import consts


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


def get_ref_ax(
    img_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    trim: Tuple[int, int, int, int] | None = None,
    rot_deg: float = 0,
) -> Tuple[plt.Figure, plt.Axes]:
    img = Image.open(img_path)
    if rot_deg != 0:
        img = img.rotate(rot_deg, expand=True)
    if trim is not None:
        w, h = img.size
        left, bottom, right, top = trim
        img = img.crop((left, top, w - right, h - bottom))

    fig, ax = plt.subplots()
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto")
    return fig, ax


def run():
    sim = get_andren1994(Nz=100)
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    ds.to_netcdf("andren1994.nc")


def make_report(ds: xr.Dataset):
    f = ds["frc_f_c"].item()
    tf = ds["time"] * f
    mo_settings = MOSettings.deserialize(ds.attrs["mo_settings"])

    with BaseReport(title="Andren 1994 Validation", path="report.html") as r:
        r.add_text("Comparison of jax-scm against Andren et al. (1994) for neutral boundary layer.")

        # Fig 2: Normalized vertically integrated TKE
        tke_int = np.trapezoid(y=ds["qke"] / 2, x=ds["z"])
        tke_norm = f / ds["mo_u_st"] ** 3 * tke_int

        fig, ax = get_ref_ax(
            "ref/a94_fig2.png",
            x_lims=(0, 14),
            y_lims=(0, 1.5),
            trim=(227, 156, 207, 14),  # left bottom right top
            rot_deg=-0.2,
        )
        ax.plot(tf, tke_norm, **plot_kwargs)
        ax.set_xlabel("tf")
        ax.set_ylabel(r"$f \int q^2/2 \, dz \; / \; u_*^3$")
        ax.set_yticks(np.arange(0, 1.6, 0.25))
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 2: Normalized vertically integrated TKE")

        # Fig 3a: C_u deviation from steady state
        uw0 = ds["mo_u_w"]
        C_u = -f / uw0 * np.trapezoid(y=ds["v"] - ds["frc_v_geo"], x=ds["z"])

        fig, ax = get_ref_ax(
            "ref/a94_fig3a.png",
            x_lims=(0, 14),
            y_lims=(0, 2),
            trim=(92, 106, 50, 19),  # left bottom right top
            rot_deg=0.2,
        )
        ax.plot(tf, C_u, **plot_kwargs)
        ax.set_xlabel("tf")
        ax.set_ylabel(r"$C_u$")
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 3a: C_u deviation from steady state (x-momentum)")

        # Fig 3b: C_v deviation from steady state
        vw0 = ds["mo_v_w"]
        C_v = f / vw0 * np.trapezoid(y=ds["u"] - ds["frc_u_geo"], x=ds["z"])

        fig, ax = get_ref_ax(
            "ref/a94_fig3b.png",
            x_lims=(0, 14),
            y_lims=(0, 3),
            trim=(247, 177, 217, 21),  # left bottom right top
        )
        ax.plot(tf, C_v, **plot_kwargs)
        ax.set_xlabel("tf")
        ax.set_ylim(0, 3)
        ax.set_ylabel(r"$C_v$")
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 3b: C_v deviation from steady state (y-momentum)")

        # Time-averaged statistics over last 3/f (Andren 1994)
        ds_sub = ds.sel(time=slice(7 / f, None)).mean("time")
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

        fig, ax = get_ref_ax(
            "ref/a94_fig4a.png",
            x_lims=(0, 2),
            y_lims=(0, 0.1),
            trim=(100, 92, 68, 20),  # left bottom right top
            rot_deg=-0.2,
        )
        ax.scatter(phi_m, zh * f / u_st, **scatter_kwargs)
        ax.axvline(1, color="k", ls="--", lw=1)
        ax.set_ylim(0, 0.1)
        ax.set_xlabel(r"$\Phi_M$")
        ax.set_ylabel(r"$zf/u_*$")
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 4a: Phi_M gradient function in the surface layer")

        # Fig 6a: Normalized u-momentum flux profile
        zh_norm = ds_sub["zh"] * f / u_st
        uw_norm = ds_sub["u_w"] / u_st**2

        fig, ax = get_ref_ax(
            "ref/a94_fig6a.png",
            x_lims=(-1, 0.2),
            y_lims=(0, 0.35),
            trim=(133, 82, 53, 25),  # left bottom right top
        )
        ax.scatter(uw_norm, zh_norm, **scatter_kwargs)
        ax.set_xlabel(r"$\overline{uw}/u_*^2$")
        ax.set_ylabel(r"$zf/u_*$")
        ax.set_ylim(0, 0.35)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 6a: Normalized u-momentum flux profile")

        # Fig 6b: Normalized v-momentum flux profile
        vw_norm = ds_sub["v_w"] / u_st**2

        fig, ax = get_ref_ax(
            "ref/a94_fig6b.png",
            x_lims=(-0.7, 0.3),
            y_lims=(0, 0.35),
            trim=(300, 170, 237, 23),  # left bottom right top
        )
        ax.scatter(vw_norm, zh_norm, **scatter_kwargs)
        ax.set_xlabel(r"$\overline{vw}/u_*^2$")
        ax.set_ylabel(r"$zf/u_*$")
        ax.set_ylim(0, 0.35)
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 6b: Normalized v-momentum flux profile")


if __name__ == "__main__":
    with jax.enable_x64():
        run()

    ds = xr.open_dataset("andren1994.nc")
    make_report(ds)
