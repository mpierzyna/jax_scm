# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

JAX-SCM is a Single Column Model (SCM) for atmospheric boundary layer simulation, implemented in JAX for differentiability and JIT compilation. It implements the MYNN 2.5 turbulence closure scheme with Monin-Obukhov surface layer coupling. The code is designed for research use with ERA5/CERRA reanalysis data integration.

## Commands

This project uses `uv` for package management. Always prefix Python/tool invocations with `uv run`.

```bash
# Install (editable)
uv sync

# Run tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_e2e.py

# Run a single test
uv run pytest tests/test_e2e.py::test_name

# Launch interactive UI
uv run scm-ui

# Run a script
uv run python some_script.py
```

## Architecture

### Key Abstractions (`src/scm/interfaces.py`)

All core types are JAX-registered dataclasses (`@jax.tree_util.register_dataclass`) to support JIT compilation:

- **`Simulation[ProgVarsT, DiagVarsT]`** — full simulation setup: grid, initial conditions, forcing, time bounds
- **`Forcing[ProgVarsT, DiagVarsT]`** — time-dependent forcing: geostrophic wind, surface fluxes, Coriolis, large-scale tendencies
- **`Output[ProgVarsT, DiagVarsT]`** — results: state/diagnostic/MO trajectories over time
- **`ModelFn`** protocol — `(t_s, state) → (tendencies, diag, mo_result)`, the ODE right-hand side

### Physics Modules

- **`src/scm/mynn/`** — MYNN 2.5 closure. `ProgVarsMYNN` holds u, v, th, qv, qke (note: qke = q² = 2×TKE). `closure.py` derives length scales and eddy diffusivities; references Nakanishi & Niino 2009 (NN09). `model.py` assembles the full model function via `init_model()`.
- **`src/scm/mo.py`** — Monin-Obukhov surface layer. `init_mo_sfc()` builds an iterative solver. `MOResult` contains friction velocity, surface fluxes, Obukhov length, and stability parameter.
- **`src/scm/grid.py`** — `StaggeredGrid` with full levels (cell centers) and half levels (cell faces). Full levels: `z = dz * (0.5, 1.5, ...)`, half levels: `zh = dz * (0, 1, ...)`.
- **`src/scm/grad.py`** — `d_dz()`: 1st-order finite differences from full to half levels.

### Time Integration (`src/scm/time_stepping/`)

- **`base.py`** — `simulate()`: unified entry point using `jax.lax.scan` for JIT-compatible looping
- **`explicit.py`** — Euler (warmup) and Adams-Bashforth 2 (AB2); adaptive AB2 with CFL-based timestep `dt = CFL_max * dz² / K_max`
- **`implicit.py`** — Crank-Nicolson semi-implicit scheme; solves a tridiagonal system per variable using `jax.lax.linalg.tridiagonal_solve`
- **`utils.py`** — state clipping to physical bounds (e.g., qke ≥ qke_min, qv ≥ 0); critical for stability

### Data Flow

```
Simulation + Forcing → init_model() → ModelFn
                                          ↓
                       simulate() [lax.scan over timesteps]
                         1. ModelFn(t_s, state) → tendencies
                         2. Time stepper (AB2 or CN) → new state
                         3. Clip to physical bounds
                          ↓
                       Output → out_to_ds() → xarray Dataset → NetCDF
```

### Supporting Modules

- **`src/scm/config/`** — Pydantic namelist model; `load_namelist(yaml)` for YAML config
- **`src/scm/forcing/`** — ERA5 (`era5.py`) and CERRA (`cerra/`) reanalysis interfaces; `utils.py` for interpolation
- **`src/scm/io/`** — `out_to_ds()` converts `Output` to xarray Dataset; `cache.py` for ERA5 data caching
- **`src/scm/reporter/`** — HTML diagnostic reports
- **`src/scm_ui/`** — Bokeh web app and Click CLI
- **`validation/`** — GABLS1 and GABLS3 reference cases against literature

### Conventions

- All dataclasses holding JAX arrays must be registered with `@jax.tree_util.register_dataclass`
- Fluxes are positive upward (w_th > 0 = upward heat flux)
- Units are SI throughout
- Finite differences are 1st-order on the staggered grid; diffusion terms live on half levels
- The closure applies a 1-2-1 filter to diffusivities to suppress vertical oscillations
- Partial condensation and level-3 closure are not implemented
