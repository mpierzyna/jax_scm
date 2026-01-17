# SCM Codebase TODO List

This document lists the remaining tasks and issues that should be addressed to improve the codebase's correctness, robustness, and maintainability.

## 🚨 Critical Bug Fixes & Physics Validation

These issues are likely causing incorrect simulation results and should be prioritized.

- [ ] **Investigate and Fix YSU Closure Fluxes**
  - **File**: `scm/closures/ysu.py`
  - **Issue**: The module manually flips the sign of all output fluxes at the end of the calculation, with a comment noting this is the "only way that simulation doesn't blow up." This is a strong indicator of a fundamental bug in the physics implementation.

- [ ] **Validate and Correct MYNN Closure Physics**
  - **File**: `scm/closures/mynn.py`
  - **Issue**: The implementation is likely experimental and contains numerous `TODO` comments that question the correctness of the physics:
    - Inconsistent handling of moisture (virtual vs. potential temperature).
    - Uncertainty about the sign of the TKE transport term.
    - Uncertainty about the physical origin of some equations (e.g., `G_M`).

- [ ] **Verify Equations in the Column MO Closure**
  - **File**: `scm/closures/mo.py`
  - **Issue**: There are `TODO` comments asking "where does this come from?" for the core eddy diffusivity equations. The physical basis of this closure needs to be verified and documented.

## 🧹 Code Refactoring & Maintenance

These tasks will improve the code's quality, readability, and ease of maintenance.

- [ ] **Consolidate or Remove `scm/odeint.py`**
  - **Files**: `scm/odeint.py`, `scm/time_stepping.py`
  - **Issue**: The project contains two different time-stepping modules. The `odeint.py` module also contains an RK2 implementation explicitly marked as "not tested". A decision should be made to either test and integrate this module properly or remove it to avoid confusion.

- [ ] **Clean Up `model_mynn.py`**
  - **File**: `model_mynn.py`
  - **Issue**: The main script contains a significant amount of commented-out code for alternative simulation cases and plotting. This should be removed or moved to a separate location (e.g., notebooks, example scripts).

- [ ] **Address Remaining `TODO` Comments**
  - **Files**: various
  - **Issue**: Review and address the smaller `TODO` comments throughout the codebase, such as:
    - Renaming `DiagVars` to `ClosureVars` in `scm/interfaces.py`.
    - Cross-referencing constants with established models like WRF as noted in `scm/closures/mynn.py`.

- [ ] **Generalize `ProgVars` for YSU**
  - **File**: `cases.py`
  - **Issue**: The `get_ysu` case currently initializes a `ProgVarsMYNN` object because a `ProgVars` for the YSU scheme (using `th` and `q` instead of `thv` and `q_sq`) doesn't exist. To properly use the YSU closure, a corresponding set of prognostic variables should be used.
