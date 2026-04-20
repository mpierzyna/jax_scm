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
from scm.mynn.interfaces import ProgVarsMYNN
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


def get_a94(Nz: int = 40) -> Simulation:
    grid = StaggeredGrid(Nz=Nz, H=1500)

    f_c = convert.get_fc(lat_deg=45)
    u_g = jnp.ones(Nz) * 10
    v_g = jnp.zeros(Nz)
    mo_settings = MOSettings(z0h=0.1, z0m=0.1)

    forcing = Forcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=f_c,
        w_th_s=lambda t_s: jnp.array(0.0),
        w_qv_s=lambda t_s: jnp.array(0.0),
        dth_dz_top=0.0,
    )

    df = pd.read_csv("ref/andren94_tab_A1.csv")
    u = jnp.interp(grid.z, df["z"].values, df["u"].values)
    v = jnp.interp(grid.z, df["z"].values, df["v"].values)
    qke = jnp.interp(grid.z, df["z"].values, df["tke"].values) * 2
    init = ProgVarsMYNN(
        u=u,
        v=v,
        th=jnp.ones(Nz) * 273.15,
        qv=jnp.zeros(Nz),
        qke=qke,
    )

    return Simulation(
        name="Andren1994",
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        grid=grid,
        th_ref=273.15,
        t_start_s=0,
        t_end_s=int(10 / f_c),
    )


def run():
    sim = get_a94(Nz=100)
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    ds.to_netcdf("andren1994.nc")


def make_report(ds: xr.Dataset):
    f = ds["frc_f_c"].item()
    tf = ds["time"] * f

    with BaseReport(title="Andren 1994 Validation", path="report.html") as r:
        r.add_text("Comparison of jax-scm against Andren et al. (1994) neutral Ekman layer LES reference results.")

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
        zh_norm = ds_sub["zh"] * f / u_st
        z_norm = ds_sub["z"] * f / u_st

        # Fig 4a: Surface layer phi_m gradient function
        du_dz = np.gradient(ds_sub["u"], ds_sub["z"])
        dv_dz = np.gradient(ds_sub["v"], ds_sub["z"])
        phi_m = consts.kappa * ds_sub["z"] / u_st * np.sqrt(du_dz**2 + dv_dz**2)

        fig, ax = get_ref_ax(
            "ref/a94_fig4a.png",
            x_lims=(0, 2),
            y_lims=(0, 0.1),
            trim=(100, 92, 68, 20),  # left bottom right top
            rot_deg=-0.2,
        )
        ax.scatter(phi_m, z_norm, **scatter_kwargs)
        ax.axvline(1, color="k", ls="--", lw=1)
        ax.set_ylim(0, 0.1)
        ax.set_xlabel(r"$\Phi_M$")
        ax.set_ylabel(r"$zf/u_*$")
        ax.legend()
        r.add_mpl_fig(fig, caption="Fig 4a: Phi_M gradient function in the surface layer")

        # Fig 6a: Normalized u-momentum flux profile
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
