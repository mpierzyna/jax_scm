"""Physical and numerical constants used throughout JAX-SCM."""

g = 9.81  # m/s^2, gravitational acceleration
kappa = 0.4  # von-Karman constant
rho_0 = 1.225  # kg/m^3, reference air density
cp = 1005.0  # J/(kg K), Specific heat of dry air at constant pressure
gamma = 1.4  # Ratio of specific heats for dry air (dimensionless).
Rd = 287.0  # J/(kg K), specific gas constant for dry air
L_v = 2257e3  # J/kg, latent heat of vaporization of water

# Physical state floors (applied in clip_state after every step)
qke_min = 1e-10  # minimum q^2=2*TKE to avoid sqrt(0) in closure

# Numerical guards for differentiability
smooth_eps = 1e-10  # floor for x^(frac<1) expressions used in safe_root to keep gradients finite
K_min = 1e-6  # minimum eddy diffusivity for CFL denominator
L_min = 1e-3  # minimum length scale in dissipation denominator
