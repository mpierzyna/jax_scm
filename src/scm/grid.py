import dataclasses

import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclasses.dataclass(frozen=True)
class StaggeredGrid:
    """Staggered vertical grid with cell centers and faces.

    Full levels (cell centers) are located at ``z = dz * (i + 0.5)`` for
    ``i = 0, …, Nz-1``; half levels (cell faces) at ``zh = dz * i`` for
    ``i = 0, …, Nz``.

    Attributes
    ----------
    H : float
        Domain height (m).
    Nz : int
        Number of full levels (cell centers).
    """

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
    """Plot the staggered grid, showing cell centers and face positions.

    Parameters
    ----------
    grid : StaggeredGrid
        Grid to visualise.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with a single axes showing full-level dots and half-level
        dashed lines.
    """
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
