from __future__ import annotations

import dataclasses
import logging
from typing import Tuple, List, Literal, TypeVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cases
from scm.closures.ysu import init_ysu_closure
from scm.grid import StaggeredGrid
from scm.interfaces import DiagVars, ProgVars, StaticForcing, ModelFn, ClosureFn, TransientForcing
from scm.mo import MOSimilarityFuncs, init_mo_sfc, MOResult, BusingerDyerSimFuncs
from scm.odeint import METHODS as ODE_METHODS
from scm.odeint import init_time_stepper
from scm.utils import make_dataset

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("scm")

T = TypeVar("T")


@dataclasses.dataclass
class SurfaceProperties:
    """Surface properties for the model."""

    z0m: float
    z0h: float
    sim_funcs: MOSimilarityFuncs
    prescribe: Literal["th_s", "w_th_s"]  # todo: I don't like this here

    @property
    def mh_ratio(self):
        return self.z0m / self.z0h


def init_model(grid: StaggeredGrid, sfc: SurfaceProperties, closure_fn: ClosureFn) -> ModelFn:

    # Create MO model
    z_mo = float(grid.z[0])
    eval_mo = init_mo_sfc(
        z0m=sfc.z0m,
        z0h=sfc.z0h,
        z=z_mo,
        z_grad=z_mo / 2,  # Halfway between surface and first full level
        sim_funcs=sfc.sim_funcs,
        prescribe=sfc.prescribe,
    )

    @jax.jit
    def _model(state: ProgVars, forcing: StaticForcing) -> Tuple[ProgVars, DiagVars]:
        # Unpack state
        u, v, th, q = state.u, state.v, state.th, state.q

        # Unpack forcing
        f_c = forcing.f_c
        u_geo, v_geo = forcing.u_geo, forcing.v_geo
        w_th_s, th_s, w_q_s = forcing.w_th_s, forcing.th_s, forcing.w_q_s

        # Run MO for surface coupling
        mo_res: MOResult = eval_mo(u_0=u[0], v_0=v[0], th_0=th[0], w_th_s=w_th_s, th_s=th_s, w_q_s=w_q_s)

        # Compute vertical gradients of state for fluxes (half levels, 1st order finite differences)
        du_dz = jnp.zeros(grid.Nz_h)
        du_dz = du_dz.at[1:-1].set((u[1:] - u[:-1]) / grid.dz)
        du_dz = du_dz.at[0].set(mo_res.du_dz)

        dv_dz = jnp.zeros(grid.Nz_h)
        dv_dz = dv_dz.at[1:-1].set((v[1:] - v[:-1]) / grid.dz)
        dv_dz = dv_dz.at[0].set(mo_res.dv_dz)

        dth_dz = jnp.zeros(grid.Nz_h)
        dth_dz = dth_dz.at[1:-1].set((th[1:] - th[:-1]) / grid.dz)
        dth_dz = dth_dz.at[0].set(mo_res.dth_dz)
        dth_dz = dth_dz.at[-1].set(forcing.dth_dz_top)

        dq_dz = jnp.zeros(grid.Nz_h)
        dq_dz = dq_dz.at[1:-1].set((q[1:] - q[:-1]) / grid.dz)
        dq_dz = dq_dz.at[0].set(mo_res.dq_dz)

        # PBL SCHEME on half levels
        grads = ProgVars(u=du_dz, v=dv_dz, th=dth_dz, q=dq_dz)
        diag = closure_fn(state, grads, mo_res)
        u_w, v_w, w_th, w_q = diag.u_w, diag.v_w, diag.w_th, diag.w_q  # unpack

        # Update fluxes with MO results
        u_w = u_w.at[0].set(mo_res.u_w)
        v_w = v_w.at[0].set(mo_res.v_w)
        w_th = w_th.at[0].set(mo_res.w_th)

        # Compute flux divergence (half levels -> full levels)
        div_u_w = (u_w[1:] - u_w[:-1]) / grid.dz
        div_v_w = (v_w[1:] - v_w[:-1]) / grid.dz
        div_w_th = (w_th[1:] - w_th[:-1]) / grid.dz
        div_w_q = (w_q[1:] - w_q[:-1]) / grid.dz

        # Compute tendencies
        u_tend = f_c * v - f_c * v_geo - div_u_w
        v_tend = -f_c * u + f_c * u_geo - div_v_w
        th_tend = -div_w_th
        q_tend = -div_w_q

        # Gather tendencies and updated diagnostics
        tends = ProgVars(u=u_tend, v=v_tend, th=th_tend, q=q_tend)
        diag = update_dc_obj(diag, u_w=u_w, v_w=v_w, w_th=w_th)
        return tends, diag

    return _model


def update_dc_obj(d: T, **updates) -> T:
    """Update dataclass object with new values."""
    d_dict = dataclasses.asdict(d)
    d_dict.update(updates)
    return d.__class__(**d_dict)


def simulate(
    model: ModelFn,
    ic: ProgVars,
    forcing: TransientForcing,
    dt_s: float,
    t_end_s: float,
    dt_out_s: float,
    ode_int: ODE_METHODS,
) -> Tuple[ProgVars, DiagVars, jnp.ndarray]:
    # Setup time arrays
    t_outer = jnp.arange(0, t_end_s, dt_out_s)
    rel_t_inner = jnp.arange(0, dt_out_s, dt_s)
    jax.debug.print(
        f"Inner steps: {len(rel_t_inner)}, "
        f"Outer steps: {len(t_outer)}, "
        f"Total steps: {len(t_outer) * len(rel_t_inner)}"
    )

    # Create time stepper
    model_stepper = init_time_stepper(model, dt=dt_s, method=ode_int)

    # Create forcing evaluation function
    get_forcing = forcing.get_eval_fn()

    @jax.jit
    def _scan_inner(carry, t):
        """Advance model by one step but don't accumulate outputs"""
        (state, _) = carry
        state_next, diag_next = model_stepper(state, forcing=get_forcing(t))
        # jax.debug.print("{t}", t=t)
        return (state_next, diag_next), None

    @jax.jit
    def _scan_outer(carry, t):
        """Advance model by inner steps and accumulate outputs"""
        (state, _) = carry
        (state_next, diag_next), _ = jax.lax.scan(_scan_inner, init=carry, xs=t + rel_t_inner)
        jax.debug.print("t={t} ({frac_done:.2f}%)", t=t + dt_out_s, frac_done=100 * (t + dt_out_s) / t_end_s)
        return (state_next, diag_next), (state_next, diag_next)

    # Perform one step to get init DiagVars object, which we can use to initialize the scan
    _, diag_init = model(ic, forcing=get_forcing(jnp.array(0.0)))

    jax.debug.print("Begin simulation...")
    _, (state_hist, diag_hist) = jax.lax.scan(_scan_outer, init=(ic, diag_init), xs=t_outer)
    return state_hist, diag_hist, t_outer


def plot_state(state: ProgVars, grid: StaggeredGrid):
    """Plot initial conditions."""
    fig, (ax_uv, ax_th, ax_q) = plt.subplots(ncols=3, figsize=(8, 3), constrained_layout=True)
    ax_uv.plot(state.u, grid.z)
    ax_uv.plot(state.v, grid.z)
    ax_th.plot(state.th, grid.z)
    ax_q.plot(state.q, grid.z)
    fig.show()


def plot_hist(hist: List, t: jnp.ndarray, grid: StaggeredGrid, plot_sfc_val: bool, cmap: str = "viridis"):
    """Plot history of diagnostics."""

    def _plot_profiles(keys):
        fig, axarr = plt.subplots(ncols=len(keys), figsize=(len(keys) * 1.5, 3), sharey="all", constrained_layout=True)
        colors = plt.get_cmap(cmap)(jnp.linspace(0, 1, len(hist)))
        for item, c in zip(hist, colors):
            for ax, k in zip(axarr, keys):
                vals = getattr(item, k)
                z = grid.z if len(vals) == grid.Nz else grid.zh
                if not plot_sfc_val:
                    vals = vals[1:]
                    z = z[1:]
                ax.plot(vals, z, color=c)

        for ax, k in zip(axarr, keys):
            ax.set_xlabel(k)
            ax.margins(y=0)
        axarr[0].set_ylabel("Height (m)")

        fig.show()

    hist_dict = dataclasses.asdict(hist[0])
    keys_profile = []
    keys_ts = []
    for k, v in hist_dict.items():
        if v.ndim == 1:
            keys_profile.append(k)
        elif v.ndim == 0:
            keys_ts.append(k)
        else:
            raise ValueError(f"Unexpected dimension for key '{k}': {v.ndim}")

    if keys_profile:
        _plot_profiles(keys_profile)
    if keys_ts:
        plot_sfc_hist(hist, t=t, keys=keys_ts)


def plot_sfc_hist(hist: List[DiagVars | ProgVars], t: jnp.ndarray, keys: List[str] = None):
    """Plot history of diagnostics at surface."""
    if keys is None:
        keys = list(dataclasses.asdict(hist[0]).keys())

    sfc_vals = {k: [] for k in keys}
    for item in hist:
        for k in keys:
            v = getattr(item, k)
            if v.ndim == 0:  # Single value
                sfc_vals[k].append(v)
            elif v.ndim == 1:  # Profile
                sfc_vals[k].append(v[0])  # Take the first value (surface value)
            else:
                raise ValueError(f"Unexpected dimension for key '{k}': {v.ndim}")

    fig, axarr = plt.subplots(nrows=len(keys), figsize=(5, len(keys) * 1), sharex="all", constrained_layout=True)
    for ax, k in zip(axarr, keys):
        ax.plot(t, sfc_vals[k])
        ax.set_xlabel("Time, s")
        ax.set_ylabel(k)
        ax.margins(x=0)

    fig.show()


def unstack_hist(v: T) -> List[T]:
    v_dict = dataclasses.asdict(v)
    v_class = v.__class__
    n, _ = v_dict[next(iter(v_dict))].shape  # Get number of time steps
    return [v_class(**{k: v_dict[k][i] for k in v_dict}) for i in range(n)]


if __name__ == "__main__":
    # Ekman spiral
    # grid, init, forcing = cases.get_ekman(Nz=100)

    # YSU test case
    grid, init, forcing = cases.get_ysu()
    sfc = SurfaceProperties(z0m=0.1, z0h=0.1, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")

    # Init and run model
    model = init_model(grid, sfc, closure_fn=init_ysu_closure(grid=grid))
    state_hist, diag_hist, t = simulate(
        model, init, forcing, dt_s=0.1, t_end_s=60 * 60 * 10, dt_out_s=10, ode_int="euler"
    )
    # state_hist, diag_hist, t = simulate(model, init, forcing, dt_s=1, t_end_s=60 * 60 * 5, dt_out_s=60, ode_int="euler")

    # Save output
    ds = make_dataset(state_hist, diag_hist, time=t, grid=grid)
    ds.to_netcdf("out.nc")

    # Unstack for plotting
    state_hist = unstack_hist(state_hist)
    diag_hist = unstack_hist(diag_hist)

    plot_hist(state_hist, t, grid, plot_sfc_val=True)
    plot_hist(diag_hist, t, grid, plot_sfc_val=True)
