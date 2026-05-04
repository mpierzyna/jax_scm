from typing import List

import jax
import jax.numpy as jnp
import xarray as xr

from scm.config import AdaptiveTimestepConfig, Namelist
from scm.examples.gabls1 import get_gabls1
from scm.interfaces import Output
from scm.io.local import out_to_ds
from scm.mynn.closure import MYNNParams
from scm.mynn.model import init_model
from scm.time_stepping import simulate


def unstack(batched_out: Output) -> List[Output]:
    leaves, treedef = jax.tree_util.tree_flatten(batched_out)
    batch_size = leaves[0].shape[0]
    return [treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(batch_size)]


def run() -> xr.Dataset:
    # Setup simulation and model
    sim = get_gabls1(Nz=64, plot=False)
    cfg = Namelist(
        adaptive_timestep=AdaptiveTimestepConfig(cfl_max=0.05),
        print_advanced_status=False,  # needs to be off for `simulate` to compile.
    )
    model = init_model(sim, cfg=cfg)

    def simulate_member(b1) -> Output:
        """Run one member with a given B1 value."""
        return simulate(model=model, sim=sim, cfg=cfg, params=MYNNParams(B1=b1))

    # Space to explore needs to be jax array
    B1_space = jnp.array([10, 15, 20, 25, 30, 35, 40, 45])

    # Run ensemble in parallel using vmap
    ens_out = jax.vmap(simulate_member)(B1_space)
    ens_out = unstack(ens_out)

    # Convert to xarray
    ens_ds = xr.concat(
        [out_to_ds(out=out, sim=sim).expand_dims(B1=[b1]) for out, b1 in zip(ens_out, B1_space)],
        dim="B1",
    )
    ens_ds.to_netcdf("ens_out.nc")
    return ens_ds


if __name__ == "__main__":
    # Run ensemble
    ens_ds = run()

    # Plot example variables for each ensemble member
    print("Plotting results...")

    plot_vars = ["Km", "u", "th"]
    for v in plot_vars:
        z_dim = "zh" if "zh" in ens_ds[v].dims else "z"
        p = ens_ds[v].plot.contourf(
            x="time",
            y=z_dim,
            col="B1",
        )

        # Save figure
        p.fig.savefig(f"ens_{v}.png")

    print("Plotting complete.")
