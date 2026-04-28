import matplotlib.pyplot as plt
import seaborn as sns
from scm.examples import get_andren1994, get_wangara_day33, get_gabls1
from scm.interfaces import Simulation
import jax
import jax.numpy as jnp

sns.set_palette("colorblind")

COLORS = {
    "u": "C0",
    "v": "C1",
    "th": "C2",
    "qv": "C3",
    "qke": "C4",
}

LABELS_PRETTY = {
    "u": "$u$",
    "v": "$v$",
    "th": r"$\theta$",
    "qv": r"$q_v$",
    "qke": r"$q^2$",
    "th_s": r"$\theta_s$",
    "w_th": r"$w'\theta'$",
    "w_qv": r"$w'{q_v}'$",
}
UNITS = {
    "u": "m s$^{-1}$",
    "v": "m s$^{-1}$",
    "th": "K",
    "qv": "g/kg",
    "qke": "m$^2$ s$^{-2}$",
    "th_s": "K",
    "w_th": "K m s$^{-1}$",
    "w_qv": "g/kg m s$^{-1}$",
}

# Global simulation objects for plotting
sim_a94 = get_andren1994()
sim_gab1 = get_gabls1()
sim_wg33 = get_wangara_day33()
sims = [sim_a94, sim_gab1, sim_wg33]


def plot_ic(sim: Simulation, fig: plt.Figure, gs: plt.SubplotSpec) -> None:
    """Plot initial conditions."""
    ic_gs = gs.subgridspec(nrows=1, ncols=4)
    ax_uv = fig.add_subplot(ic_gs[0, 0])
    ax_th = fig.add_subplot(ic_gs[0, 1], sharey=ax_uv)
    ax_qv = fig.add_subplot(ic_gs[0, 2], sharey=ax_uv)
    ax_qke = fig.add_subplot(ic_gs[0, 3], sharey=ax_uv)

    ax_uv.plot(sim.init.u, sim.grid.z, label="u", color=COLORS["u"])
    ax_uv.plot(sim.init.v, sim.grid.z, label="v", color=COLORS["v"])
    ax_uv.legend()
    ax_uv.set_xlabel(f"Wind, {UNITS['u']}")

    ax_th.plot(sim.init.th, sim.grid.z, label="th", color=COLORS["th"])
    ax_th.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['th']}")

    ax_qv.plot(sim.init.qv * 100, sim.grid.z, label="qv", color=COLORS["qv"])
    ax_qv.set_xlim(0, None)
    ax_qv.set_xlabel(f"{LABELS_PRETTY['qv']}, {UNITS['qv']}")

    ax_qke.plot(sim.init.qke, sim.grid.z, label="qke", color=COLORS["qke"])
    ax_qke.set_xlim(0, None)
    ax_qke.set_xlabel(f"{LABELS_PRETTY['qke']}, {UNITS['qke']}")

    ax_uv.set_ylabel("Height, m")
    ax_uv.set_ylim(0, sim.grid.H)


def plot_bc(sim: Simulation, fig: plt.Figure, gs: plt.SubplotSpec) -> None:
    """Plot boundary conditions (i.e. time-varying forcing)."""
    row_gs = gs.subgridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1, 3),
        height_ratios=(1, 1),
    )
    ax_ug = fig.add_subplot(row_gs[:, 0])
    ax_heat = fig.add_subplot(row_gs[0, 1])
    ax_w_qv = fig.add_subplot(row_gs[1, 1], sharex=ax_heat)

    t = jnp.linspace(sim.t_start_s, sim.t_end_s)

    # Geostrophic forcing
    ug = jax.vmap(sim.forcing.u_geo)(t)
    pc = ax_ug.pcolormesh(t, sim.grid.z, ug.T, shading="auto", cmap="Blues")
    ax_ug.set_ylabel("Height, m")
    ax_ug.set_ylim(0, sim.grid.H)
    fig.colorbar(pc, ax=ax_ug)

    # Heat forcing
    if sim.forcing.w_th_s is None:
        # Surface temperature forcing
        heat = jax.vmap(sim.forcing.th_s)(t)
        label = LABELS_PRETTY["th_s"]
        unit = UNITS["th_s"]
    else:
        # Sensible heat flux forcing
        heat = jax.vmap(sim.forcing.w_th_s)(t)
        label = LABELS_PRETTY["w_th"]
        unit = UNITS["w_th"]

    ax_heat.plot(t, heat, color=COLORS["th"])
    ax_heat.set_ylabel(f"{label}, {unit}")

    # Moisture forcing
    ax_w_qv.plot(t, jax.vmap(sim.forcing.w_qv_s)(t), label="w_qv", color=COLORS["qv"])
    ax_w_qv.set_xlabel("Time, s")
    ax_w_qv.set_ylabel(f"{LABELS_PRETTY['w_qv']}, {UNITS['w_qv']}")


def plot_ic_bc(sim: Simulation) -> plt.Figure:
    """Plot initial conditions and boundary conditions for a given simulation."""
    fig = plt.figure(constrained_layout=True, figsize=(6, 3))
    outer_gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1, 1))

    plot_ic(sim, fig, outer_gs[0])
    plot_bc(sim, fig, outer_gs[1])

    return fig


if __name__ == "__main__":
    figs = [plot_ic_bc(sim) for sim in sims]
    plt.show()
