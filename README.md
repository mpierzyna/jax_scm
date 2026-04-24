# JAX-SCM: Technical Description

JAX-SCM is a high-performance single-column model (SCM) implemented in JAX, designed for atmospheric boundary layer research. It leverages JIT compilation, automatic differentiation, and hardware acceleration (GPU/TPU) to provide a modern framework for boundary layer modeling.

![coverage](docs/coverage-badge.svg)
![tests](docs/test-badge.svg)

## Numerical and Physical Features

### Time Integration

- **Integration Schemes**: Supports multiple time-stepping strategies:
  - **Explicit**: 2nd-order Adams-Bashforth (AB2) with an initial Euler warmup step.
    - **Adaptive Time Stepping**: Optional CFL-based adaptive $\Delta t$ for explicit diffusion, ensuring numerical stability while maximizing computational efficiency.
  - **Semi-Implicit**: Crank-Nicolson (CN) scheme for vertical diffusion, coupled with AB2 for explicit source terms (Coriolis, large-scale advection, TKE production/dissipation). Tridiagonal matrix solver used for matrix inversion.

### Spatial Discretization

- **Vertical Grid**: staggered vertical grid 
  - prognostic state variables are located at cell centers (full levels)
  - turbulent fluxes/diffusivities are located at cell faces (half levels)
- **Finite Differences**: First-order finite difference approximations for vertical gradient operators.

### Turbulence Closure

- **MYNN Level-2.5**: Implements the improved Mellor-Yamada-Nakanishi-Niino (2009) closure scheme.
- **Prognostic Variables**:
  - Horizontal wind components ($u, v$).
  - Potential temperature ($\theta$).
  - Specific humidity ($q_v$).
  - Twice the turbulent kinetic energy ($q^2$ or $qke$).
- **Length Scale Formulation**: A master length scale derived from the harmonic mean of surface-limited ($L_S$), turbulent ($L_T$), and buoyancy-limited ($L_B$) scales.
- **Stability Functions**: Level-2.5 stability functions for momentum ($S_M$), heat ($S_H$), and TKE ($S_q$).

### Surface Layer and Coupling

- **Monin-Obukhov Similarity Theory (MOST)**: Surface boundary conditions derived via MOST.
- **Similarity Functions**: Support for standard Businger-Dyer relationships with configurable stability parameters.
- **Iterative Solution**: Robust iterative solver for the stability parameter $\zeta = z/L$ to couple the surface with the lowest model level.
- **Flexible Boundary Conditions**: Support for both prescribed surface temperature (skin temperature) and prescribed sensible/latent heat fluxes.

### Forcing and Data Integration

- **Dynamic Forcing**: Time-dependent geostrophic wind profiles and Coriolis forcing.
- **Large-Scale Tendencies**: Inclusion of horizontal advective tendencies and vertical subsidence $(-w_{ls} \frac{\partial \phi}{\partial z})$ for scalars and momentum.
- **Data Interfaces**: Automated pipelines for ERA5 reanalysis integration, including vertical interpolation and geostrophic wind calculation from geopotential fields.

### Implementation Framework

- **JAX-Native**: Fully differentiable and vectorized implementation.
- **Metadata-Rich I/O**: Xarray-based data handling with NetCDF output and integrated HTML reporting for simulation diagnostics.

## Ensemble and Optimization

- **Parameter Sweeps**: `jax.vmap` enables efficient ensemble runs over model parameters (e.g., closure constant B1) without looping — see `examples/ensemble/`.

## Validation

Multiple standard cases from literature are implemented to validate jax-scm:

- GABLS1, stable boundary layer experiment (Cuxart et al. 2006)
- Andren (1994), neutral boundary layer with geostrophic forcing
- Wangara, day 33, convective boundary layer initialized with observed profiles
