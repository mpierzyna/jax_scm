# GABLS3 validation

This directory contains code to validate jax-scm against the GABLS3 case.
GABLS3 is a realistic case, which covers the day/night/day transition and a nocturnal
low level jet. Additionally, it has the following interesting features to test jax-scm:

- depends on large scale forcing -> tests large scale forcing implementation
- requires realistic forcing (here ERA5 reanalysis) -> tests ERA5 forcing
- can be validated against observations
