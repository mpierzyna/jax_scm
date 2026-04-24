# Numerical Accuracy & Stability: TODO

Findings from a review of spatial discretization (`src/scm/mynn/model.py`, `src/scm/grad.py`,
`src/scm/mynn/closure.py`) and temporal integration (`src/scm/time_stepping/`).

---

## Medium Priority

### 1. Diffusivity K is first-order accurate in time in the CN scheme

**File:** `src/scm/time_stepping/implicit.py`

K is diagnosed from state `y` (start of step) and held fixed during the CN solve. True CN
requires K at the midpoint `t + dt/2`; lagging K by a full step degrades the diffusion terms
from O(dt²) to O(dt). Explicit sources use AB2 extrapolation (2nd order), creating an internal
inconsistency.

**Fix:** A single predictor-corrector iteration (predict `y2` with current K, re-diagnose K
from `y2`, re-solve) restores 2nd-order accuracy in K at the cost of one extra model evaluation
per step.

---

## Low Priority

### 2. Geostrophic advection gradient uses a widened stencil

**File:** `src/scm/mynn/model.py:96-100`

`dug_dz` is computed on half levels and then averaged back to full levels via
`(dug_dz[1:] + dug_dz[:-1]) / 2`. This full → half → full round-trip widens the stencil to
3 points and damps short-wave components of `u_geo(z)`. Computing the gradient directly on full
levels with centered differences `(u_geo[j+1] - u_geo[j-1]) / (2*dz)` and one-sided differences
at the boundaries avoids this.

---

### 3. `clip_state` corrections are not fed back into the tendency computation

**File:** `src/scm/time_stepping/utils.py`

`clip_state` is documented as a numerical floor (done), but clipping `qv ≥ 0` without a
compensating adjustment to the water vapour budget remains physically inconsistent (negligible
in dry cases). The `qke` clip to `qke_min` introduces a spurious source of TKE that accumulates
silently in long runs.

**Fix:** Apply flux-form corrections to conserve moisture when clipping `qv`.
