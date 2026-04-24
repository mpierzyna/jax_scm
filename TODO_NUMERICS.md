# Numerical Accuracy & Stability: TODO

Findings from a review of spatial discretization (`src/scm/mynn/model.py`, `src/scm/grad.py`,
`src/scm/mynn/closure.py`) and temporal integration (`src/scm/time_stepping/`).

---

## High Priority

### 1. QKE dissipation treated fully explicitly — main stability risk

**File:** `src/scm/time_stepping/implicit.py`, `src/scm/mynn/model.py:90`

The dissipation term `eps = qke^(3/2) / (B1 * L_full)` is an explicit source in both the
explicit and semi-implicit schemes. The effective destruction timescale is `tau = B1 * L / q`.
Near the surface where `L → 0` and `q → 0`, `tau` can be shorter than the timestep, causing
over-dissipation instability or large spurious TKE tendencies.

**Fix:** Linearize and treat implicitly. Factor `eps = r * qke` where `r = qke^(1/2) / (B1 * L_full)`,
then add `r` to the CN diagonal as `lhs_d += dt/2 * r` inside `get_cn_sparse_lin_system`.
This makes TKE destruction unconditionally stable with no additional cost.

---

### 2. Surface boundary gradient for theta and qv uses zeroth-order extrapolation

**File:** `src/scm/mynn/model.py:56-57`, `src/scm/mynn/closure.py:66`

`du_dz` and `dv_dz` correctly use MOST-derived surface gradients (`mo_res.dudz`, `mo_res.dvdz`),
but `dth_dz`, `dqv_dz`, and `dthv_dz` fall back to `bot="edge"`, which repeats the first
interior gradient. This zeroth-order extrapolation is inconsistent and degrades `G_H`, `Ri`,
and all stability functions that feed on the surface buoyancy gradient — most consequential in
stable boundary layers with a sharp surface inversion.

**Fix:** Derive a MOST-based surface gradient for theta and qv analogous to `mo_res.dudz`/`mo_res.dvdz`
and add them to `MOResult`. Use these as `bot=` in `d_dz` calls in `model.py` and in `closure.py`
for `dthv_dz`.

---

## Medium Priority

### 3. Diffusivity K is first-order accurate in time in the CN scheme

**File:** `src/scm/time_stepping/implicit.py:200-201`

K is diagnosed from state `y1` (start of step) and held fixed during the CN solve. True CN
requires K at the midpoint `t + dt/2`; lagging K by a full step degrades the diffusion terms
from O(dt²) to O(dt). Explicit sources use AB2 extrapolation (2nd order), creating an internal
inconsistency.

**Fix:** A single predictor-corrector iteration (predict `y2` with current K, re-diagnose K
from `y2`, re-solve) restores 2nd-order accuracy in K at the cost of one extra model evaluation
per step.

---

### 4. AB2 stability region for diffusion is narrower than forward Euler

**File:** `src/scm/time_stepping/explicit.py:30-31`, `src/scm/time_stepping/base.py:35`

The adaptive timestep uses `dt = CFL_max * dz² / K_max`, which is the parabolic CFL condition
for forward Euler. AB2's stability region on the negative real axis is smaller than Euler's, so
this CFL criterion is not conservative enough for AB2. In practice `CFL_max` must be kept below
~0.5 to avoid oscillatory instability.

**Fix (option A):** Document and enforce `CFL_max ≤ 0.5` for the explicit AB2 scheme.
**Fix (option B):** Replace AB2 with TVD-RK3, which has a better stability region for stiff
diffusive terms (stable up to the Euler CFL limit), is 3rd-order accurate, and costs only two
extra model evaluations per step.

---

## Low Priority

### 5. Surface flux terms in CN use current-step values only (no AB2)

**File:** `src/scm/time_stepping/implicit.py:160-166`

Explicit sources (Coriolis, QKE budget) are extrapolated with AB2 to 2nd order, but the surface
fluxes `mo_res.u_w`, `mo_res.w_th`, etc. injected as `sfc_flux` in `_apply_cn` come from the
current step `mo_res1` without extrapolation. This is 1st-order in time for the surface forcing,
inconsistent with the 2nd-order explicit sources.

**Fix:** Store the previous-step MO result alongside `S_prev` and apply AB2 extrapolation to
the surface fluxes as well.

---

### 6. Top half-level extrapolation for `qke_h` is zeroth-order

**File:** `src/scm/mynn/closure.py:59`

`jnp.pad(..., mode="edge")` copies `qke[-1]` as the value at `zh[-1]` (top half level). Linear
extrapolation (`1.5*qke[-1] - 0.5*qke[-2]`) is trivial to implement and more accurate.

---

### 7. Geostrophic advection gradient uses a widened stencil

**File:** `src/scm/mynn/model.py:96-100`

`dug_dz` is computed on half levels and then averaged back to full levels via
`(dug_dz[1:] + dug_dz[:-1]) / 2`. This full → half → full round-trip widens the stencil to
3 points and damps short-wave components of `u_geo(z)`. Computing the gradient directly on full
levels with centered differences `(u_geo[j+1] - u_geo[j-1]) / (2*dz)` and one-sided differences
at the boundaries avoids this.

---

### 8. `clip_state` corrections are not fed back into the tendency computation

**File:** `src/scm/time_stepping/utils.py:53-58`

`clip_state` is applied after each time step, but the clipped values are never used to recompute
tendencies. In particular, clipping `qv ≥ 0` without a compensating adjustment to the water
vapour budget is physically inconsistent (though negligible in dry cases). The `qke` clip to
`qke_min` similarly introduces a spurious source of TKE that accumulates silently in long runs.

**Fix (conservative):** Document that `clip_state` is a numerical floor, not a physical correction,
and add an assertion or diagnostic counter that tracks how often/how much clipping occurs.
**Fix (rigorous):** Apply flux-form corrections to conserve moisture when clipping `qv`.
