from __future__ import annotations

from typing import Tuple
import dataclasses
from PIL import Image
import matplotlib.pyplot as plt
from scm.forcing.era5 import get_era5_sim
import xarray as xr
from scm.grid import StaggeredGrid
from scm.forcing.interp import interp_dtindex

import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from scm.config import load_namelist
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
    img = ImageOps.grayscale(img)

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
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto", cmap="Greys_r")

    return fig, ax


def run():
    # Setup simulation for GABLS3 case from ERA5
    sim = get_era5_sim(
        name="ERA5 Test Simulation",
        lat_deg=52.0,
        lon_deg=5.0,
        time_slice=("2006-07-01T11:00", "2006-07-02T12:00"),
        grid=StaggeredGrid(Nz=200, H=3000.0),
        source="cds",
        cache_dir="./era5",
    )
    # sim.forcing = dataclasses.replace(sim.forcing, ls_tends=None)  # disable large-scale tendencies

    # Load config and run simulation
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    out_file = pathlib.Path(f"out.nc")
    ds = out_to_ds(
        out=out,
        sim=sim,
        time=interp_dtindex(t_s=np.array(out.t_s), idx=sim.t_index).round("1min"),
    )
    ds.to_netcdf(out_file)
    print("Written to disk.")
    return ds


if __name__ == "__main__":
    ds = run()
    # ds = xr.open_dataset("out.nc")
    ds = ds.sel(time=slice("2006-07-01T12:00", "2006-07-02T12:00"))  # exclude spinup

    ds["m"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds["d"] = np.rad2deg(np.arctan2(-ds["u"], -ds["v"]))
    z = ds["z"].values
    t_h = np.linspace(0, 24, ds.sizes["time"])

    with BaseReport(title="GABLS3 Validation", path=f"val_gabls3.html") as r:
        r.add_text(
            "This report compares the jax-scm model against GABLS3 reference results from Bosveld et al. (2014). "
            "Instead of using the prescribed initial and boundary conditions from the paper, we use ERA5 data."
        )

        r.add_heading("Profiles after init (12:10 UTC)", level=2)
        ds_1210UTC = ds.sel(time="2006-07-01T12:10")

        # th
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (296, 308),
            (0, 3000),
            trim=(98, 636, 535, 17),
        )
        ax.plot(ds_1210UTC["th"], z, **plot_kwargs)
        ax.set_xlabel("Potential temperature, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 12:10 UTC")

        # qv
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (0, 0.01),
            (0, 3000),
            trim=(604, 636, 28, 17),
        )
        ax.plot(ds_1210UTC["qv"], z, **plot_kwargs)
        ax.set_xlabel("Specific humidity, kg/kg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Specific humidity profile at 12:10 UTC")

        # m
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (1, 6),
            (0, 3000),
            trim=(98, 132, 527, 511),
        )
        ax.plot(ds_1210UTC["m"], z, **plot_kwargs)
        ax.set_xlabel("Wind speed, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind profile at 12:10 UTC")

        # d
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (80, 140),
            (0, 3000),
            trim=(604, 132, 20, 511),
        )
        ax.plot(ds_1210UTC["d"], z, **plot_kwargs)
        ax.set_xlabel("Wind direction, deg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind direction at 12:10 UTC")

        r.add_heading("Profiles at 00:00 UTC", level=2)
        ds_000UTC = ds.sel(time="2006-07-02T00:00", z=slice(0, 500))

        # th
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (286, 300),
            (0, 500),
            trim=(92, 616, 530, 13),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["th"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Potential temperature, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 00:00 UTC")

        # qv
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (0.007, 0.012),
            (0, 500),
            trim=(602, 616, 24, 15),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["qv"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Specific humidity, kg/kg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Specific humidity profile at 00:00 UTC")

        # m
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (0, 14),
            (0, 500),
            trim=(93, 106, 525, 519),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["m"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Wind speed, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind profile at 00:00 UTC")

        # d
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (70, 140),
            (0, 500),
            trim=(602, 106, 20, 522),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["d"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Wind direction, deg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind direction at 00:00 UTC")

        # Time series
        # T2m
        r.add_heading("Time series", level=2)
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig3.png",
            (0, 24),
            (12, 30),
            trim=(82, 631, 558, 12),  # left, bottom, right, top
        )
        ax.plot(t_h, ds["mo_th2"] - 273.15, **plot_kwargs)  # todo: convert to air
        ax.set_xlabel("Time, h")
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_ylabel("2m air temperature, deg C")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 00:00 UTC")

        # # qv
        # fig, ax = get_ref_ax(
        #     "ref_bosveld14/fig2.png",
        #     (0.007, 0.012),
        #     (0, 500),
        #     trim=(602, 616, 24, 15),  # left, bottom, right, top
        # )
        # ax.plot(ds_000UTC["qv"], ds_000UTC.z, **plot_kwargs)
        # ax.set_xlabel("Specific humidity, kg/kg")
        # ax.set_ylabel("Height, z")
        # ax.legend()
        # r.add_mpl_fig(fig, caption="Specific humidity profile at 00:00 UTC")

        # m
        m200 = ds["m"].interp(z=200)
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig3.png",
            (0, 24),
            (0, 14),
            trim=(82, 135, 558, 508),  # left, bottom, right, top
        )
        ax.plot(t_h, m200, **plot_kwargs)
        ax.set_xlabel("Time, h")
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_ylabel("Wind speed at 200m, m/s")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind speed at 200m")

        # d
        u200, v200 = ds["u"].interp(z=200), ds["v"].interp(z=200)
        d200 = np.rad2deg(np.arctan2(-u200, -v200))
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig3.png",
            (0, 24),
            (60, 200),
            trim=(594, 136, 41, 502),  # left, bottom, right, top
        )
        ax.plot(t_h, d200, **plot_kwargs)
        ax.set_xlabel("Time, h")
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_ylabel("Wind speed at 200m, m/s")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind direction at 200m")

        # shfx
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig4.png",
            (0, 24),
            (-80, 200),
            trim=(97, 139, 542, 24),  # left, bottom, right, top
        )
        ax.plot(t_h, ds["mo_w_th"] * 1.216e3, **plot_kwargs)
        ax.set_xlabel("Time, h")
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_ylabel("Sensible heat flux, W/m^2")
        ax.set_yticks(np.arange(-80, 201, 40))
        ax.legend()
        r.add_mpl_fig(fig, caption="Surface sensible heat flux")

        # ust
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig4.png",
            (0, 24),
            (0.0, 0.7),
            trim=(593, 138, 42, 22),  # left, bottom, right, top
        )
        ax.plot(t_h, ds["mo_u_st"], **plot_kwargs)
        ax.set_xlabel("Time, h")
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_ylabel("Surface friction velocity, m/s")
        ax.set_yticks(np.arange(0, 0.71, 0.1))
        ax.legend()
        r.add_mpl_fig(fig, caption="Surface friction velocity")
