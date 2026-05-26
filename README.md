# JAX-SCM

![coverage](docs/coverage-badge.svg)
![tests](docs/test-badge.svg)

JAX-SCM is a modern single-column model (SCM) for atmospheric boundary layer simulation, implemented
in [JAX](https://github.com/jax-ml/jax). It implements the MYNN 2.5 turbulence closure with Monin-Obukhov surface layer
coupling, and is designed for research use.

For technical details and to cite JAX-SCM, see the preprint:

> Pierzyna, Maximilian. “JAX-SCM v1.0: A Modern Atmospheric Single-Column Model for Boundary Layer Research.”
> arXiv:2605.24544, arXiv, May 2026. https://doi.org/10.48550/arXiv.2605.24544.



> [!TIP]
> To get a quick taste of JAX-SCM, you can **run the GABLS1 stable boundary layer case in your browser**!
> Just click the "Open in Colab" button below, which will open an [example notebook](examples/GABLS1_interactive.ipynb)
> in [Google Colab](https://colab.research.google.com/).
>
> <a target="_blank" href="https://colab.research.google.com/github/mpierzyna/jax_scm/blob/main/examples/GABLS1_interactive.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
>
> No local installation is required!

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for package management. Install it first if you don't have it.
Then clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd jax-scm
uv sync  # CPU only
```

To install JAX with CUDA GPU support, run `uv sync --extra cuda` instead.

Verify the installation:

```bash
uv run python -c "import scm; print('OK')"
```

## Simulation Setup

Run your own simulations from the `workspaces/` directory. Create a new file, e.g., `my_run.py` in a subdirectory and
set up the simulation as follows.

All simulation parameters (initial conditions, forcing, time stepping, ...) are controlled by the `Simulation`
object. See examples in [`scm.examples`](src/scm/examples). Once the `Simulation` is built, initialize a `Model` and
run the time-stepping loop with `simulate()`. The output is a dictionary of arrays, which can be converted to an
`xarray.Dataset` for analysis and export.

```python
from scm.config import Namelist, TimeIntMethod
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

sim = get_gabls1(Nz=64)  # build Simulation object
cfg = Namelist(time_int=TimeIntMethod.IMPLICIT)  # choose time integration
model = init_model(sim, cfg=cfg)
out = simulate(model=model, sim=sim, cfg=cfg)

ds = out_to_ds(out=out, sim=sim)  # convert to xarray Dataset
ds.to_netcdf("out.nc")
```

Run the simulation as

```bash
uv run python my_run.py
```

### The `Simulation` object

`Simulation` is a frozen dataclass that fully specifies a model run. Build one directly or use a pre-built example from
`scm.examples`:

| Field         | Type                 | Description                                                               |
|---------------|----------------------|---------------------------------------------------------------------------|
| `name`        | `str`                | Identifier for this run                                                   |
| `grid`        | `StaggeredGrid`      | Vertical grid (Nz, dz)                                                    |
| `mo_settings` | `MOSettings`         | Monin-Obukhov solver settings (z0, stability functions)                   |
| `init`        | `ProgVarsMYNN`       | Initial profiles of u, v, θ, q_v, qke                                     |
| `forcing`     | `Forcing`            | Time-dependent boundary conditions and large-scale tendencies (see below) |
| `th_ref`      | `float`              | Reference potential temperature for buoyancy terms (K)                    |
| `t_start_s`   | `int`                | Simulation start time (seconds)                                           |
| `t_end_s`     | `int`                | Simulation end time (seconds)                                             |
| `t_index_fn`  | `Callable` or `None` | Optional mapping from seconds to, e.g., hours forr xarray output          |

**`Forcing`** specifies all time-varying boundary conditions and external tendencies. Every `ForceSingleFn` is a
callable `f(t_s) → scalar or (Nz,) array`:

| Field            | Type                      | Description                                                                                                          |
|------------------|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| `u_geo`, `v_geo` | `ForceSingleFn`           | Geostrophic wind profiles, must return `(Nz,)`                                                                       |
| `f_c`            | `float`                   | Coriolis parameter (1/s)                                                                                             |
| `w_th_s`         | `ForceSingleFn` or `None` | Surface sensible heat flux (K m/s); mutually exclusive with `th_s`                                                   |
| `th_s`           | `ForceSingleFn` or `None` | Prescribed skin temperature (K); mutually exclusive with `w_th_s`                                                    |
| `w_qv_s`         | `ForceSingleFn`           | Surface moisture flux (kg/kg m/s)                                                                                    |
| `dth_dz_top`     | `float`                   | Potential temperature lapse rate at the upper boundary (K/m)                                                         |
| `ls_tends`       | `ForceTendsFn` or `None`  | Large-scale advective tendencies and subsidence; signature `(t_s, state, grads, diag) → ProgVarsMYNN`  (unvalidated) | 

### Namelist configuration

The `Namelist` controls time integration and output. The two most common configurations, also used by the validation
cases, are:

**Crank-Nicolson (semi-implicit, recommended):**

```yaml
# namelist_cn.yaml
time_int: "implicit"
dt_s: 1.0
dt_s_out: 300.0
logging:
  log_every_n: 10
```

**Adams-Bashforth 2 (explicit with adaptive timestep):**

```yaml
# namelist_ab2.yaml
time_int: "explicit"
adaptive_timestep:
  cfl_max: 0.025
  dt_s_max: 10.0
dt_s: 0.01
logging:
  log_every_n: 10
```

Load a namelist from YAML:

```python
from scm.config import load_namelist

cfg = load_namelist("namelist_cn.yaml")
```

## Running Validation Cases

Pre-built validation cases are located in `validation/`. Each case has its own `run.py` and two namelists (CN and AB2).
Run them from inside their directory:

```bash
cd validation/gabls1
uv run python run.py
```

This writes `out_cn.nc`, `out_ab2.nc`, and HTML diagnostic reports to the same directory.

### Available validation cases

| Case           | Directory                | Description                                              | Reference                |
|----------------|--------------------------|----------------------------------------------------------|--------------------------|
| GABLS1         | `validation/gabls1/`     | Stable boundary layer, 9-hour surface cooling, Nz=64     | Cuxart et al. (2006)     |
| Andren 1994    | `validation/andren1994/` | Neutral boundary layer with geostrophic forcing, Nz=100  | Andren et al. (1994)     |
| Wangara Day 33 | `validation/wangara/`    | Convective boundary layer from observed profiles, Nz=100 | Nakanishi & Niino (2009) |

## Numerical and Physical Features

### Time Integration

- **Explicit**: 2nd-order Adams-Bashforth (AB2) with an Euler warmup step. Optional CFL-based adaptive $\Delta t$ for
  explicit diffusion.
- **Semi-Implicit**: Crank-Nicolson (CN) for vertical diffusion, combined with AB2 for explicit source terms (Coriolis,
  large-scale advection, TKE production/dissipation). Tridiagonal matrix solver used for the implicit step.

### Spatial Discretization

Staggered vertical grid:

- **Full levels** (cell centers, `z = dz * (0.5, 1.5, ...)`): prognostic state variables (u, v, θ, q_v, qke)
- **Half levels** (cell faces, `zh = dz * (0, 1, ...)`): turbulent fluxes and diffusivities

First-order finite differences for vertical gradient operators.

### Turbulence Closure — MYNN 2.5

Implements the Mellor-Yamada-Nakanishi-Niino Level-2.5 closure:

- **Prognostic variables**: u, v, θ, q_v, qke (where qke = q² = 2×TKE)
- **Master length scale**: harmonic mean of surface-limited ($L_S$), turbulent ($L_T$), and buoyancy-limited ($L_B$)
  scales
- A 1-2-1 filter is applied to length scales and diffusivities to suppress vertical numerical oscillations

### Surface Layer

Monin-Obukhov Similarity Theory (MOST) with iterative solver for the stability parameter $\zeta = z/L$. Supports:

- Businger-Dyer stability functions (configurable parameters)
- Prescribed surface temperature (skin temperature) or prescribed sensible/latent heat fluxes
- Prescribed moisture fluxes

### Forcing

- Time-dependent geostrophic wind profiles via callable
- Coriolis forcing
- ERA5 interfaces for realistic large-scale forcing and initial conditions (untested)
