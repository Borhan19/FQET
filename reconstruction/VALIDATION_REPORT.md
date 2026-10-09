# Independent reconstruction — numerical validation report

*Source:* **Finite Quantum Engineering Thermodynamics**, main text and Supplemental Material, dated 9 October 2026. This report accompanies a newly written reconstruction, not the lost original software.

## 1. Source-level and algebraic tests

The reconstructed fixed-system Hilbert spaces are 35-dimensional (Bose--Hubbard with four bosons and four sites) and 16-dimensional (four-spin Ising). Separate Hamiltonian assembly methods, adaptive DOP853 propagation, and fourth-order Gauss--Magnus propagation are implemented. Projector completeness, Hermiticity, canonical-state positivity, population normalization, unitary evolution, work-energy accounting, first-derivative/current identities, exact history pullback, and 15-event complement enumeration are tested.

The `pytest` suite: **11 passed**.

Representative independent solver comparisons through t=0.4 in J=1 units:

| System | DOP853 vs independent Gauss--Magnus, operator norm | Maximum DOP853 unitarity defect | Trapezoid work-energy error |
|---|---:|---:|---:|
| Bose--Hubbard | 3.2012e-8 | 3.6237e-8 | 4.0874e-9 |
| Ising (g/J=1.5) | 5.2935e-10 | 1.0881e-8 | 1.5898e-6 |

These are short-interval verification runs with the **independent Gauss--Magnus maximum step 0.01**; they are not the original publication's full Solver-B campaign. See `results/smoke.json`.

## 2. Recovering manuscript operator geometry

These data were computed *from the reconstructed microscopic matrices and propagators*, without importing the manuscript's geometry CSV.

| System / history lag pair | First-layer overlap | Conditioned second-layer overlap | Normalized Gram condition number |
|---|---:|---:|---:|
| BH, local (0.025, 0.050) | 0.942613003 | 0.914842164 | 392026.282 |
| BH, practical (0.10, 0.20) | 0.313262032 | 0.271243446 | 2305.890 |
| Ising, local (0.06, 0.20) | 0.969165101 | 0.724417293 | 29006.579 |
| Ising, mixed (0.06, 0.70) | 0.969165101 | 0.113790187 | 1565.415 |

These agree with SM Note 12, Table S5, and the cross-platform geometry values in Note 13 to their displayed precision. The crucial convention is to condition the **second history layer on the first history layer**, while conditioning **curvature on the direct current layer**. The normalized Gram condition number includes the *entire measured operator list (E, R, H1, H2)* after removal of the identity, not only the history increments. See `results/geometry.json`.

## 3. Reconstructed constitutive response — 360 fits

An independently generated table was calibrated at **8 parameter corners x 3 forecast times (0.05, 0.10, 0.20) x 15 events = 360** cutting-plane spectral fits, with nominal-state anchoring and a fixed per-setting reference shot budget.

- Spectral LP fits not meeting the chosen convergence-gap tolerance: **0**.
- Largest certified objective upper-minus-lower gap: **1.997e-7**.
- Largest operator-norm residual across the corner fits: **0.176997**.
- Coefficient data: `results/bh_sheet_15events.npz`.

This is a **new, fixed-shot calibration**, not a reconstruction of the unavailable original jointly optimized coefficients.

## 4. Independent positive-tube holdout — 12,960 tests

The new table was frozen before constructing independent tests at 64 scrambled Sobol parameter points and 8 near-corner stress points. The holdout used an independent fourth-order Gauss--Magnus propagator with step at most 0.0025 and three independently seeded positive tube states (trace distance 0.0098 from the canonical reference), plus the canonical state at each point.

| Quantity | Reconstructed campaign |
|---|---:|
| Parameter points | 72 |
| Preparations per point | 4 |
| Forecast times | 3 |
| Binary events | 15 |
| Total predictions | **12,960** |
| Largest absolute event-probability error | **0.000550275 = 0.0550%** |

All 12,960 evaluated errors are below the 1% task tolerance in this particular sampled holdout. The reported error is neither the original **0.1278%** nor directly comparable to it, because the original had **81** future times, **349,920** predictions, different frozen coefficients, and different random tube directions.

File: `results/reconstructed_bh_holdout_15events_72points.csv`.

## 5. Independent parameter grid — 125 parameter points

The reconstructed 8-corner response was evaluated on a 5x5x5 parameter grid, over the same 3 forecast times and 15 events: **5,625 event-time parameter evaluations**.

| Finite-grid diagnostic | New reconstruction |
|---|---:|
| Largest operator residual norm | 0.176997 |
| Largest sampled preparation-tube pairwise width upper bound, kappa*d | **0.00707988 = 0.7080%** |
| Largest sampled single-state tube error upper bound | 0.00353994 = 0.3540% |
| Largest nominal response bias | 0.00034058 = 0.0341% |

These values are **on-grid diagnostics**. Numerically estimated finite differences in each parameter direction cannot be treated as rigorous global Lipschitz constants. Therefore, the 125-point audit **does not recover or independently prove** the paper's **0.7304% continuous-domain** certificate.

File: `results/reconstructed_parameter_grid_125.csv`.

## 6. Ising representation comparison — limited screened examples

A reconstructed metrology screen at g/J=1.5, horizon 0.02, all 15 binary events, but only this single future time and a finite list of possible history lags, gives:

- Best screened history cost: **595,150 repetitions** (an H3 candidate with lags 0.10 and 0.70).
- Best screened instantaneous cost: **731,560 repetitions** (ERJ).

At horizon 0.05, a longer screen reached the following *completed candidates* before hitting its execution limit:

- ERJ: 3,106,607 repetitions.
- ERJK2: 3,092,971 repetitions.
- H2 with lag 0.06: 4,542,303 repetitions.
- H3 with lags 0.06/0.70: 4,559,550 repetitions.

The final H3 candidate (0.20, 0.70) was **not completed**, so no globally preferred architecture is asserted at horizon 0.05. These example screen costs are **not comparable** with the published 81-time, 15-event, fully optimized phase map; the screening and coefficient-shot alternation are numerical heuristics with different task constraints. Neither example reproduces the original boundary J*tau_family=0.029207. The qualitative shift toward direct derivatives is consistent with that reference, but an exact boundary audit is outstanding.

The updated `--checkpoint` option allows multi-hour Ising architecture searches to resume.

## 7. Scientific limitations and next steps

The original full nonlinear metrology/lag optimizer, its exact coefficient arrays, cross-seeding seeds, positive perturbation directions, original numerical sensitivity envelopes, and original selected Solver-B constraints are unavailable. Only the explicitly written physical model, analytic identities, reported numerical tables, and methods descriptions can be reconstructed.

Before asserting full paper reproducibility in a public release, independently verify the **full 81-time x 15-event x 8-corner calibration**, the **72-point x 4-state x 81-time x 15-event holdout**, the **continuous-parameter certificate** based on genuinely upper-bounding directional sensitivity, and the **six-field Ising phase map** with its original constrained numerical optimization. That requires considerably more computation and potentially additional original methodological details. A positive reduced-grid result is no substitute.

**Provenance:** All files in this report under `results/` were generated by the newly written reconstruction, not copied from the ten original manuscript CSV tables.
