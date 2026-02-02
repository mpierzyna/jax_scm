import dataclasses

import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclasses.dataclass
class StaggeredGrid:
    """Staggered vertical grid with cell centers and faces."""

    H: float  # Domain height, m
    Nz: int  # Number of full levels

    @property
    def dz(self) -> float:
        """Vertical grid spacing."""
        return self.H / self.Nz

    @property
    def z(self) -> jnp.ndarray:
        """Vertical positions of full levels (cell centers)."""
        return self.dz * (jnp.arange(self.Nz) + 0.5)

    @property
    def zh(self) -> jnp.ndarray:
        """Vertical positions of half levels (cell faces)."""
        return self.dz * jnp.arange(self.Nz + 1)

    @property
    def Nz_h(self) -> int:
        """Number of half levels."""
        return self.Nz + 1


def plot_grid(grid: StaggeredGrid) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2, 4), constrained_layout=True)

    # Plot cell centers
    ax.scatter(
        jnp.ones_like(grid.z) * 0.5,
        grid.z,
        marker="o",
        label="Cell centers (z_c)",
    )

    # Plot cell faces
    for z_i in grid.zh:
        ax.plot([0, 1], [z_i, z_i], color="C0", linestyle="--")

    ax.set_ylim(-grid.dz / 2, grid.H + grid.dz / 2)
    ax.set_xticks([])

    return fig


if __name__ == "__main__":
    grid = StaggeredGrid(H=1000, Nz=16)
    plot_grid(grid).show()
