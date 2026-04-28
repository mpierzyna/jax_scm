import matplotlib.pyplot as plt
import seaborn as sns
from scm.examples import get_andren1994, get_wangara_day33, get_gabls1
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
}
UNITS = {
    "u": "m s$^{-1}$",
    "v": "m s$^{-1}$",
    "th": "K",
    "qv": "g/kg",
    "qke": "m$^2$ s$^{-2}$",
}

# Global simulation objects for plotting
sim_a94 = get_andren1994()
sim_gab1 = get_gabls1()
sim_wg33 = get_wangara_day33()
sims = [sim_a94, sim_gab1, sim_wg33]


def plot_ic() -> plt.Figure:
    fig, axarr = plt.subplots(
        nrows=3,
        ncols=4,
        constrained_layout=True,
        sharey="row",
        figsize=(6, 8),
    )
    for i, (axrow, sim) in enumerate(zip(axarr, sims)):
        ax_uv, ax_th, ax_qv, ax_qke = axrow
        ax_uv.plot(sim.init.u, sim.grid.z, label="u", color=COLORS["u"])
        ax_uv.plot(sim.init.v, sim.grid.z, label="v", color=COLORS["v"])
        ax_uv.legend()
        if i == 2:
            ax_uv.set_xlabel(f"Wind, {UNITS['u']}")

        ax_th.plot(sim.init.th, sim.grid.z, label="th", color=COLORS["th"])
        if i == 2:
            ax_th.set_xlabel(f"{LABELS_PRETTY['th']}, {UNITS['th']}")

        ax_qv.plot(sim.init.qv * 100, sim.grid.z, label="qv", color=COLORS["qv"])
        ax_qv.set_xlim(0, None)
        if i == 2:
            ax_qv.set_xlabel(f"{LABELS_PRETTY['qv']}, {UNITS['qv']}")

        ax_qke.plot(sim.init.qke, sim.grid.z, label="qke", color=COLORS["qke"])
        ax_qke.set_xlim(0, None)
        if i == 2:
            ax_qke.set_xlabel(f"{LABELS_PRETTY['qke']}, {UNITS['qke']}")

    # y ax labels
    for ax, sim in zip(axarr[:, 0], sims):
        ax.set_ylabel("Height, m")
        ax.set_ylim(0, sim.grid.H)

    return fig


def plot_bc() -> plt.Figure:
    fig = plt.figure(constrained_layout=True, figsize=(8, 9))
    outer_gs = fig.add_gridspec(nrows=3, ncols=1)

    for i, sim in enumerate(sims):
        row_gs = outer_gs[i].subgridspec(
            nrows=2,
            ncols=2,
            width_ratios=(1, 3),
            height_ratios=(1, 1),
            # wspace=0.25,
            # hspace=0.18,
        )
        ax_ug = fig.add_subplot(row_gs[:, 0])
        ax_heat = fig.add_subplot(row_gs[0, 1])
        ax_w_qv = fig.add_subplot(row_gs[1, 1], sharex=ax_heat)

        t = jnp.linspace(sim.t_start_s, sim.t_end_s)

        # Geostrophic forcing
        ug = jax.vmap(sim.forcing.u_geo)(t)
        pc = ax_ug.pcolormesh(t, sim.grid.z, ug.T, shading="auto", cmap="Blues")
        fig.colorbar(pc, ax=ax_ug)

        # Heat forcing
        if sim.forcing.w_th_s is None:
            # Surface temperature forcing
            heat = jax.vmap(sim.forcing.th_s)(t)
        else:
            # Sensible heat flux forcing
            heat = jax.vmap(sim.forcing.w_th_s)(t)

        ax_heat.plot(t, heat, color=COLORS["th"])

        # Moisture forcing
        ax_w_qv.plot(t, jax.vmap(sim.forcing.w_qv_s)(t), label="w_qv", color=COLORS["qv"])

        ax_ug.set_ylabel("Height, m")
        ax_ug.set_ylim(0, sim.grid.H)
        ax_heat.set_xlabel("Time, s")
        ax_w_qv.set_xlabel("Time, s")

    return fig


if __name__ == "__main__":
    plot_bc().show()
    plt.show()
