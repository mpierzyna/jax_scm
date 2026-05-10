from __future__ import annotations

import dataclasses
import pathlib
from shutil import which
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from scm.examples import get_andren1994, get_gabls1, get_wangara_day33
from scm.examples.andren1994 import postproc_andren1994
from scm.examples.gabls1 import postproc_gabls1
from scm.examples.wangara import postproc_wangara
from scm.interfaces import Simulation

sns.set_palette("colorblind")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "figure.dpi": 300,
        # "figure.labelsize": 8,
        "lines.linewidth": 1.0,
        "hatch.linewidth": 0.5,
    }
)


FIG_WIDTH = 7  # Full-page width for figure. Use as base.

REF_KW = {"color": "lightgrey"}
JAX_SCM_KW = {"color": "C1", "linewidth": 1.5}

LABELS_PRETTY = {
    # Mean
    "u": "$U$",
    "ug": "$U_g$",
    "v": "$V$",
    "m": r"$M$",
    "th": r"$\Theta$",
    "th_s": r"$\Theta_s$",
    "qv": r"$Q_v$",
    # TKE/QKE
    "qke": r"$q^2$",
    "tke": r"$q^2/2$",
    # Turb stats
    "w_th": r"$\langle w \theta \rangle$",
    "w_qv": r"$\langle w {q_v} \rangle$",
    "u_w": r"$\langle u w \rangle$",
    "v_w": r"$\langle v w \rangle$",
    "u_st": r"$u_*$",
    "L": "$L_M$",
}
UNITS = {
    "u": "m/s",
    "v": "m/s",
    "th": "K",
    "thC": r"$^{\circ}$C",
    "qv": "g/kg",  # ATTENTION! Native unit: kg/kg
    "qke": "m$^2$/s$^2$",
    "th_s": "K",
    "w_th": "K m/s",
    "w_qv": "(g/kg) (m/s)",  # ATTENTION! Native unit: kg/kg m/s
    "u_w": "m$^2$/s$^2$",
    "v_w": "m$^2$/s$^2$",
    "u_st": "m/s",
    "K_m": "m$^2$/s",
}

FIG_ROOT = pathlib.Path("figures")
VAL_ROOT = pathlib.Path(__file__).parent.parent


def _read_ref_csv(path: pathlib.Path, sort: str) -> dict:
    """Read digitized reference CSV (label row, X/Y row, data...). Returns dict label -> (x, y)."""
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


@dataclasses.dataclass
class SimPlotSpec:
    """Plotting specifications for a given simulation"""

    sim: Simulation
    short_name: str
    time_formatter: Callable[[float], str]
    time_label: str = "Time, s"
    time_n_ticks: int = 5
    time_n_ticks_short: int = 3
    ref_dir: pathlib.Path | None = None
    out_file: pathlib.Path | None = None


# Global simulation objects for plotting
sim_a94 = get_andren1994()
sim_gab1 = get_gabls1()
sim_wg33 = get_wangara_day33()
sims = [
    SimPlotSpec(
        sim=sim_a94,
        short_name="A94",
        time_formatter=lambda t: f"{t * sim_a94.forcing.f_c:.0f}",
        time_n_ticks=6,
        time_label="$t f$, --",
        ref_dir=VAL_ROOT / "andren1994" / "ref",
        out_file=VAL_ROOT / "andren1994" / "out_cn.nc",
    ),
    SimPlotSpec(
        sim=sim_gab1,
        short_name="GAB1",
        time_formatter=lambda t: f"{t / 60 / 60:.0f}",
        time_label="$t$, h",
        time_n_ticks=4,
        time_n_ticks_short=4,
        ref_dir=VAL_ROOT / "gabls1" / "ref_cuxart06",
        out_file=VAL_ROOT / "gabls1" / "out_cn.nc",
    ),
    SimPlotSpec(
        sim=sim_wg33,
        short_name="WG33",
        time_formatter=lambda t: f"{t / 3600:02.0f}",
        time_label="Time, LST",
        time_n_ticks=8,
        ref_dir=VAL_ROOT / "wangara" / "ref",
        out_file=VAL_ROOT / "wangara" / "out_cn.nc",
    ),
]


def _add_is_const(v: jnp.ndarray, ax: plt.Axes, x: float = 0.95, y: float = 0.95, color: str = "grey") -> None:
    """Add 'constant' label if plotted variable is constant"""
    if v.mean() == 0:
        label = "zero"
    elif jnp.abs(v.std() / v.mean()) < 1e-5:
        label = "constant"
    else:
        return

    if x == 0.5:
        ha = "center"
    elif x < 0.5:
        ha = "left"
    else:
        ha = "right"

    ax.text(x, y, label, transform=ax.transAxes, ha=ha, va="top", fontsize=6, color=color)


def _xticks_only(ax: plt.Axes) -> None:
    """Keep y ticks, but turn labels off"""
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)


def _yticks_only(ax: plt.Axes) -> None:
    """Keep y ticks, but turn labels off"""
    ax.tick_params(axis="y", which="both", left=True, labelleft=False)


def _add_subplot_label(ax: plt.Axes, l: str, dy: float = 0, dx: float = 0) -> None:
    """Add subplot label (a), (b), etc. in top left corner of ax."""
    ax.text(0.025 + dx, 0.975 + dy, f"({l})", ha="left", va="top", transform=ax.transAxes)


def plot_ic(sps: SimPlotSpec, *, axes: Tuple[plt.Axes, ...], labels: Tuple[str, ...]) -> None:
    """Plot initial conditions."""
    # Unpack axes
    ax_u_v, ax_th, ax_qv, ax_qke = axes
    for ax, l in zip(axes, labels):
        _add_subplot_label(ax, l)

    sim = sps.sim
    ax_u_v.plot(sim.init.u, sim.grid.z, label=LABELS_PRETTY["u"], color="C0")
    ax_u_v.plot(sim.init.v, sim.grid.z, label=LABELS_PRETTY["v"], color="C0", ls="dashed")
    _add_is_const(v=sim.init.u, ax=ax_u_v, x=0.5, y=0.5)
    ax_u_v.legend(fontsize=6, loc="upper right")
    ax_u_v.set_xlabel(f"Wind, {UNITS['u']}")
    ax_u_v.set_ylabel("$z$, m")
    ax_u_v.set_ylim(0, sim.grid.H)

    ax_th.plot(sim.init.th, sim.grid.z, color="C0")
    _add_is_const(v=sim.init.th, ax=ax_th, x=0.5, y=0.5)
    ax_th.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['th']}")
    _yticks_only(ax_th)

    ax_qv.plot(sim.init.qv * 100, sim.grid.z, color="C0")
    _add_is_const(v=sim.init.qv, ax=ax_qv, x=0.5, y=0.5)
    ax_qv.set_xlim(0, None)
    ax_qv.set_xlabel(f"{LABELS_PRETTY['qv']}, {UNITS['qv']}")
    _yticks_only(ax_qv)

    ax_qke.plot(sim.init.qke / 2, sim.grid.z, color="C0")
    ax_qke.set_xlim(0, None)
    ax_qke.set_xlabel(f"{LABELS_PRETTY['tke']}, {UNITS['qke']}")
    _yticks_only(ax_qke)


def plot_bc(sps: SimPlotSpec, *, fig: plt.Figure, axes: Tuple[plt.Axes, ...], labels: Tuple[str, ...]) -> None:
    """Plot boundary conditions (i.e. time-varying forcing)."""
    # Unpack axes
    ax_ug, ax_heat, ax_w_qv = axes
    for ax, l in zip(axes, labels):
        _add_subplot_label(ax, l)

    sim = sps.sim
    t = jnp.linspace(sim.t_start_s, sim.t_end_s)
    t_ticks = jnp.linspace(sim.t_start_s, sim.t_end_s, sps.time_n_ticks)
    t_ticks_ug = jnp.linspace(sim.t_start_s, sim.t_end_s, sps.time_n_ticks_short)

    # Geostrophic forcing
    ug = jax.vmap(sim.forcing.u_geo)(t)
    pc = ax_ug.pcolormesh(t, sim.grid.z, ug.T, shading="auto", cmap="Blues", rasterized=True)
    _add_is_const(v=ug, ax=ax_ug, x=0.5, y=0.5, color="white")
    ax_ug.set_xticks(t_ticks_ug)
    ax_ug.set_xticklabels([sps.time_formatter(tick) for tick in t_ticks_ug])
    ax_ug.set_xlabel(sps.time_label)
    _yticks_only(ax_ug)
    cb = fig.colorbar(pc, ax=ax_ug, pad=0.01, shrink=0.65)
    cb.ax.set_title(f"{LABELS_PRETTY['ug']},\n{UNITS['u']}", loc="left", fontsize=7)

    # Heat forcing
    if sim.forcing.w_th_s is None:
        # Surface temperature forcing
        heat = jax.vmap(sim.forcing.th_s)(t)
        label = LABELS_PRETTY["th_s"]
        unit = UNITS["th_s"]
        color = "C3"
    else:
        # Sensible heat flux forcing
        heat = jax.vmap(sim.forcing.w_th_s)(t)
        label = LABELS_PRETTY["w_th"]
        unit = UNITS["w_th"]
        color = "C6"

    ax_heat.plot(t, heat, color=color)
    ax_heat.set_ylabel(f"{label}, {unit}")
    _xticks_only(ax_heat)
    _add_is_const(v=heat, ax=ax_heat)

    # Moisture forcing
    w_qv = jax.vmap(sim.forcing.w_qv_s)(t) * 1e3
    ax_w_qv.plot(t, w_qv, color="C0")
    _add_is_const(v=w_qv, ax=ax_w_qv)

    ax_w_qv.margins(x=0)
    ax_w_qv.set_xlabel(sps.time_label)
    ax_w_qv.set_xticks(t_ticks)
    ax_w_qv.set_xticklabels([sps.time_formatter(tick) for tick in t_ticks])

    ax_w_qv.set_ylabel(f"{LABELS_PRETTY['w_qv']},\n{UNITS['w_qv']}")


def plot_ic_bc(sps: SimPlotSpec) -> plt.Figure:
    """Plot initial conditions and boundary conditions for a given simulation."""
    fig = plt.figure(constrained_layout=True, figsize=(FIG_WIDTH, 1.75))
    gs = fig.add_gridspec(nrows=1, ncols=6, width_ratios=(1, 1, 1, 1, 1, 3))
    gs_ts = gs[0, -1].subgridspec(nrows=2, ncols=1)

    ax_u_v = fig.add_subplot(gs[0, 0])
    ax_th = fig.add_subplot(gs[0, 1], sharey=ax_u_v)
    ax_qv = fig.add_subplot(gs[0, 2], sharey=ax_u_v)
    ax_qke = fig.add_subplot(gs[0, 3], sharey=ax_u_v)
    ax_ug = fig.add_subplot(gs[0, 4], sharey=ax_u_v)

    ax_heat = fig.add_subplot(gs_ts[0, 0])
    ax_w_qv = fig.add_subplot(gs_ts[1, 0], sharex=ax_heat)

    plot_ic(sps, axes=(ax_u_v, ax_th, ax_qv, ax_qke), labels=("a", "b", "c", "d"))
    plot_bc(sps, fig=fig, axes=(ax_ug, ax_heat, ax_w_qv), labels=("e", "f", "g"))

    return fig


def plot_a94_res(sps: SimPlotSpec) -> plt.Figure:
    """Plot Andren 1994 results against digitized reference data."""

    def _plot_ref(ax: plt.Axes, path: pathlib.Path, sort: str = "x") -> None:
        """Overplot all digitized reference curves on ax (A94 multi-model style)."""
        for label, (x, y) in _read_ref_csv(path, sort=sort).items():
            ax.plot(x, y, label=label, **REF_KW)

    ds = xr.open_dataset(sps.out_file)
    ds_pp = postproc_andren1994(ds)
    tf = ds["time"]  # time in dataframe is normalized with f, so tf = t * f

    fig = plt.figure(figsize=(FIG_WIDTH * 0.45, FIG_WIDTH * 0.65), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=(1, 1, 1, 2))

    # Time series plots in first row
    ax = fig.add_subplot(gs[0, 0])
    _add_subplot_label(ax, "a", dx=-0.015)
    _plot_ref(ax, sps.ref_dir / "a94_fig2.csv")
    ax.plot(tf, ds_pp["tke_int_norm"], **JAX_SCM_KW)
    ax.set_xlim(0, 10)
    _xticks_only(ax)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel(r"$f \int q^2/2\,dz / u_*^3$")

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    _add_subplot_label(ax, "b", dx=-0.015)
    _plot_ref(ax, sps.ref_dir / "a94_fig3a.csv")
    ax.plot(tf, ds_pp["C_u"], **JAX_SCM_KW)
    ax.axhline(1, color="k", ls="--", lw=0.75)
    ax.set_xlim(0, 10)
    _xticks_only(ax)
    ax.set_ylim(0, 2)
    ax.set_ylabel(r"$C_u$")

    ax = fig.add_subplot(gs[2, 0], sharex=ax)
    _add_subplot_label(ax, "c", dx=-0.015)
    _plot_ref(ax, sps.ref_dir / "a94_fig3b.csv")
    ax.plot(tf, ds_pp["C_v"], **JAX_SCM_KW)
    ax.axhline(1, color="k", ls="--", lw=0.75)
    ax.set_xlabel("$t f$, -")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_ylabel(r"$C_v$")

    # Normalied profiles in second row
    gs_sub = gs[3, :].subgridspec(nrows=1, ncols=2, width_ratios=(1, 1))

    label_ust = LABELS_PRETTY["u_st"][1:-1]
    label_z_norm = rf"$z\,f/{label_ust}$"

    ax_uw = fig.add_subplot(gs_sub[0, 0])
    _add_subplot_label(ax_uw, "d")
    _plot_ref(ax_uw, sps.ref_dir / "a94_fig6a.csv", sort="y")
    ax_uw.plot(ds_pp["uw_norm"], ds_pp["zh_"], **JAX_SCM_KW)
    ax_uw.axvline(0, color="k", ls="--", lw=0.75)
    label_uw = LABELS_PRETTY["u_w"][1:-1]
    ax_uw.set_xlabel(rf"${label_uw}/{label_ust}^{{2}}$")
    ax_uw.set_ylabel(label_z_norm)
    ax_uw.set_ylim(0, 0.35)

    ax_vw = fig.add_subplot(gs_sub[0, 1], sharey=ax_uw)
    _add_subplot_label(ax_vw, "e")
    _plot_ref(ax_vw, sps.ref_dir / "a94_fig6b.csv", sort="y")
    ax_vw.plot(ds_pp["vw_norm"], ds_pp["zh_"], **JAX_SCM_KW)
    ax_vw.axvline(0, color="k", ls="--", lw=0.75)

    label_vw = LABELS_PRETTY["v_w"][1:-1]
    ax_vw.set_xlabel(rf"${label_vw}/{label_ust}^{{2}}$")
    _yticks_only(ax_vw)

    handles = [
        mlines.Line2D([], [], color=REF_KW["color"], lw=1, label="Andren et al. (1994)"),
        mlines.Line2D([], [], color=JAX_SCM_KW["color"], lw=1, label="JAX-SCM"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=7)

    return fig


def plot_wg33_res(sps: SimPlotSpec) -> plt.Figure:
    """Plot Wangara Day 33 results against NN09 reference."""
    cmap = "viridis"
    t_short = ["09:00", "10:00", "12:00", "14:00", "16:00"]
    t_long = [f"1967-08-16T{t}" for t in t_short]
    t_1400 = "1967-08-16T14:00"
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(t_short)))

    ds_full = xr.open_dataset(sps.out_file)
    ds = ds_full.sel(time=t_long)
    ds_pp = postproc_wangara(ds)
    ds_tke_budget = ds_pp.sel(time=t_1400)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=6, height_ratios=(1, 1))
    ref_kw = {**REF_KW, "ls": "--"}
    ref_kw.pop("color")  # coloring by time for WG33

    # --- Potential temperature ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig3.csv", sort="y")
    ax = fig.add_subplot(gs[0, 0])
    _add_subplot_label(ax, "a")
    for i, t in enumerate(t_short):
        ax.plot(*ref[t], color=colors[i], **ref_kw)
        ax.plot(ds["th"].isel(time=i) - 273.15, ds["z"], color=colors[i], label=t)
    ax.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['thC']}")
    ax.set_ylabel("z, m")
    ax.set_ylim(0, 2000)

    # --- Sensible heat flux ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig4.csv", sort="y")
    ax = fig.add_subplot(gs[0, 1], sharey=ax)
    _add_subplot_label(ax, "b")
    for i in range(1, 5):
        x, y = ref[t_short[i]]
        ax.plot(x / 100, y, color=colors[i], **ref_kw)
        ax.plot(ds["w_th"].isel(time=i), ds["zh"], color=colors[i], label=t_short[i])
    ax.axvline(0, color="k", ls="dotted", lw=0.75)
    ax.set_xlabel(f"{LABELS_PRETTY['w_th']}, {UNITS['w_th']}")
    _yticks_only(ax)

    # --- Water vapor ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig8.csv", sort="y")
    ax = fig.add_subplot(gs[0, 2], sharey=ax)
    _add_subplot_label(ax, "c")
    for i, t in enumerate(t_short):
        ax.plot(*ref[t], color=colors[i], **ref_kw)
        ax.plot(ds["qv"].isel(time=i) * 1000, ds["z"], color=colors[i], label=t)
    ax.set_xlabel(f"{LABELS_PRETTY['qv']}, {UNITS['qv']}")
    ax.set_xlim(0, None)
    _yticks_only(ax)

    # --- Moisture flux ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig9.csv", sort="y")
    ax = fig.add_subplot(gs[0, 3], sharey=ax)
    _add_subplot_label(ax, "d")
    for i in range(1, 5):
        x, y = ref[t_short[i]]
        ax.plot(x / 1e2, y, color=colors[i], **ref_kw)
        ax.plot(ds["w_qv"].isel(time=i) * 1e3, ds["zh"], color=colors[i], label=t_short[i])
    ax.axvline(0, color="k", ls="dotted", lw=0.75)
    ax.set_xlabel(f"{LABELS_PRETTY['w_qv']}, {UNITS['w_qv']}")
    _yticks_only(ax)

    # --- TKE ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig5.csv", sort="y")
    ax = fig.add_subplot(gs[0, 4], sharey=ax)
    _add_subplot_label(ax, "e")
    for i in range(1, 5):
        ax.plot(*ref[t_short[i]], color=colors[i], **ref_kw)
        ax.plot(ds["qke"].isel(time=i) / 2, ds["z"], color=colors[i], label=t_short[i])
    ax.set_xlabel(f"{LABELS_PRETTY['tke']}, {UNITS['qke']}")
    _yticks_only(ax)

    # --- Length scale ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig7.csv", sort="y")
    ax = fig.add_subplot(gs[0, 5], sharey=ax)
    _add_subplot_label(ax, "f")
    for i in range(1, 5):
        ax.plot(*ref[t_short[i]], color=colors[i], **ref_kw)
        ax.plot(ds["L"].isel(time=i), ds["zh"], color=colors[i], label=t_short[i])
    ax.set_xlabel(f"{LABELS_PRETTY['L']}, m")
    _yticks_only(ax)

    # Figure-level legend: time colors + NN09 reference marker
    proxy_ref = mlines.Line2D([], [], color="k", ls=ref_kw["ls"], label="Nakanishi and Niino (2009)")
    handles = [mlines.Line2D([], [], color=colors[i], label=t) for i, t in enumerate(t_short)]
    handles.append(proxy_ref)
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=7)

    gs_sub = gs[1, :].subgridspec(nrows=1, ncols=4)

    # --- Mixed layer parameters scatter (Table 1) ---
    def _annotate_scatter(ax: plt.Axes, label: str) -> None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        vmin, vmax = min(xmin, ymin), max(xmax, ymax)
        ax.plot([vmin, vmax], [vmin, vmax], color="k", ls="dotted", lw=0.75)
        ax.set_xlabel(f"{label} (NN09)")
        ax.set_ylabel(f"{label} (JAX-SCM)")

    df = pd.read_csv(sps.ref_dir / "nn09_tab1.csv")

    ax = fig.add_subplot(gs_sub[0, 0])
    _add_subplot_label(ax, "g")
    ax.scatter(df["zi"], ds_pp["zi"][1:], c=colors[1:], s=10)
    ax.set_aspect("equal")
    _annotate_scatter(ax, "$z_i$, m")

    ax = fig.add_subplot(gs_sub[0, 1])
    _add_subplot_label(ax, "h")
    ax.scatter(df["neg_R"], -ds_pp["R"][1:], c=colors[1:], s=10)
    ax.set_aspect("equal")
    _annotate_scatter(ax, "$R$")

    ax = fig.add_subplot(gs_sub[0, 2])
    _add_subplot_label(ax, "i")
    ax.scatter(df["w_st"], ds_pp["w_st"][1:], c=colors[1:], s=10)
    ax.set_aspect("equal")
    _annotate_scatter(ax, rf"$w_*$, {UNITS['u_st']}")

    # --- TKE budget at 14:00 ---
    ref = _read_ref_csv(sps.ref_dir / "nn09_fig6.csv", sort="y")
    ax = fig.add_subplot(gs_sub[0, 3])
    _add_subplot_label(ax, "j")
    ax.plot(*ref["S"], color="C0", **ref_kw)
    ax.plot(*ref["B"], color="C1", **ref_kw)
    ax.plot(*ref["T+P"], color="C2", **ref_kw)
    ax.plot(*ref["D"], color="C3", **ref_kw)
    ax.plot(ds_tke_budget["tke_P_S"].values, ds["z"], color="C0", label="$P_S$")
    ax.plot(ds_tke_budget["tke_P_B"].values, ds["z"], color="C1", label="$P_B$")
    ax.plot(
        -ds_tke_budget["div_w_tke"].values,
        ds["z"].values,
        color="C2",
        label=r"div.~$\langle w q^2 \rangle / 2$",
    )
    ax.plot(-ds_tke_budget["tke_eps"].values, ds["z"], color="C3", label=r"$-\epsilon$")
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$\square\, z_i / w_*^3$, --")
    ax.set_ylabel("z, m")
    ax.legend(fontsize=6, loc="upper center", ncols=2)

    return fig


def plot_gabls1_res(sps: SimPlotSpec) -> plt.Figure:
    """Plot GABLS1 results against Cuxart et al. (2006) multi-model reference."""

    def _plot_ref(ax: plt.Axes, path: pathlib.Path, x_scale: float = 1, sort: str = "x") -> None:
        for x, y in _read_ref_csv(path, sort=sort).values():
            ax.plot(x * x_scale, y, **REF_KW)

    ds = xr.open_dataset(sps.out_file)
    ds_pp = postproc_gabls1(ds)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / 2), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=6)

    # --- Profiles at 9 h ---
    ax_m = fig.add_subplot(gs[0, 0])
    _add_subplot_label(ax_m, "a")
    _plot_ref(ax_m, sps.ref_dir / "fig03_m.csv", sort="y")
    ax_m.plot(ds_pp["m"].isel(time=-1), ds["z"], **JAX_SCM_KW)
    ax_m.set_xlim(0, 11)
    ax_m.set_xticks([0, 5, 10])
    ax_m.set_ylim(0, 400)
    ax_m.set_xlabel(f"{LABELS_PRETTY['m']}, {UNITS['u']}")
    ax_m.set_ylabel("z, m")

    ax = fig.add_subplot(gs[0, 1], sharey=ax_m)
    _add_subplot_label(ax, "b")
    _plot_ref(ax, sps.ref_dir / "fig03_th.csv", sort="y")
    ax.plot(ds["th"].isel(time=-1), ds["z"], **JAX_SCM_KW)
    ax.set_xlim(262.5, 268)
    ax.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['th']}")
    _yticks_only(ax)

    ax = fig.add_subplot(gs[0, 2], sharey=ax_m)
    _add_subplot_label(ax, "c")
    _plot_ref(ax, sps.ref_dir / "fig04_hfx.csv", sort="y")
    ax.plot(ds["w_th"].isel(time=-1), ds["zh"], **JAX_SCM_KW)
    ax.set_xlim(-0.02, 0)
    ax.set_xlabel(f"{LABELS_PRETTY['w_th']}, {UNITS['w_th']}")
    _yticks_only(ax)

    ax = fig.add_subplot(gs[0, 3], sharey=ax_m)
    _add_subplot_label(ax, "d")
    _plot_ref(ax, sps.ref_dir / "fig04_momentum.csv", sort="y")
    ax.plot(ds_pp["tau"].isel(time=-1), ds["zh"], **JAX_SCM_KW)
    ax.set_xlim(0, 0.15)
    ax.set_xlabel(rf"$\tau$, {UNITS['u_w']}")
    _yticks_only(ax)

    ax = fig.add_subplot(gs[0, 4], sharey=ax_m)
    _add_subplot_label(ax, "e")
    _plot_ref(ax, sps.ref_dir / "fig06_Km.csv", sort="y")
    ax.plot(ds["Km"].isel(time=-1), ds["zh"], **JAX_SCM_KW)
    ax.set_xlim(0, 6)
    ax.set_xlabel(rf"$K_m$, {UNITS['K_m']}")
    _yticks_only(ax)

    ax = fig.add_subplot(gs[0, 5], sharey=ax_m)
    _add_subplot_label(ax, "f")
    _plot_ref(ax, sps.ref_dir / "fig06_Kh.csv", sort="y")
    ax.plot(ds["Kh"].isel(time=-1), ds["zh"], **JAX_SCM_KW)
    ax.set_xlim(0, 6)
    ax.set_xlabel(rf"$K_h$, {UNITS['K_m']}")
    _yticks_only(ax)

    # --- Time series ---
    gs_sub = gs[1, :].subgridspec(nrows=1, ncols=2, width_ratios=(1, 1))

    ax = fig.add_subplot(gs_sub[0, 0])
    _add_subplot_label(ax, "g", dx=-0.015)
    _plot_ref(ax, sps.ref_dir / "fig02_blh.csv", sort="x", x_scale=1 / 60)
    ax.plot(ds["time"], ds_pp["blh"], **JAX_SCM_KW)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xlabel(sps.time_label)
    ax.set_ylim(0, 400)
    ax.set_ylabel("BLH, m")

    ax = fig.add_subplot(gs_sub[0, 1])
    _add_subplot_label(ax, "h", dx=-0.015)
    _plot_ref(ax, sps.ref_dir / "fig02_ust.csv", sort="x", x_scale=1 / 60)
    ax.plot(ds["time"], ds["mo_u_st"], **JAX_SCM_KW)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xlabel(sps.time_label)
    ax.set_ylim(0.2, 0.5)
    ax.set_ylabel(f"{LABELS_PRETTY['u_st']}, {UNITS['u_st']}")

    handles = [
        mlines.Line2D([], [], color=REF_KW["color"], lw=1, label="Cuxart et al. (2006) SCMs"),
        mlines.Line2D([], [], color=JAX_SCM_KW["color"], lw=1, label="JAX-SCM"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=7)

    return fig


if __name__ == "__main__":
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for sim in sims:
        fig_ic_bc = plot_ic_bc(sim)
        # Use bbox_inches='tight' and pad_inches=0 to remove extra padding around the
        # figure when exporting to PDF. Some PDF viewers still show a hairline margin,
        # but this produces the smallest possible white border from Matplotlib.
        fig_ic_bc.savefig(
            FIG_ROOT / f"ic_bc_{sim.short_name}.pdf",
            bbox_inches="tight",
            pad_inches=0.0,
        )

    sps_a94, sps_gab1, sps_wg33 = sims

    fig_a94 = plot_a94_res(sps_a94)
    fig_a94.savefig(FIG_ROOT / "res_A94.pdf")

    fig_gab1 = plot_gabls1_res(sps_gab1)
    fig_gab1.savefig(FIG_ROOT / "res_GAB1.pdf")

    fig_wg33 = plot_wg33_res(sps_wg33)
    fig_wg33.savefig(FIG_ROOT / "res_WG33.pdf")
