# Audit ledger — reconstructed implementation versus reported calculations

**Date:** 9 October 2026. **Source:** *Finite Quantum Engineering Thermodynamics* (31-page October 9 version), main text and SM Notes 2--13. No claim is made that reconstructed source is the unavailable original simulation source.

| Component | Source statement | What is reconstructed | Evidence / limit |
|:--|:--|:--|:--|
| Fixed-system Hilbert spaces | BH L=Q=4, dim 35; Ising L=4, dim 16 | Both Hamiltonians, canonical preparations, coarse projectors, control ramps | Unit tests check Hermiticity, trace, positivity, projectors, independent Ising tensor construction |
| Propagation | DOP853, atol 2e-11/rtol 2e-9; independent 4th-order Gauss-Magnus | Both methods implemented; independent H rebuilding for Magnus | `results/smoke.json`; short-time verification only |
| Energy/work | dE = integral Lambda-dot * <N_R> dt, closed system | Full monitored energy and work numerical comparison | `results/smoke.json`; finite quadrature error |
| Restricted operator geometry | SM Notes 2--4 | Historical pullback, currents, curvature, event operators, operator-norm distance | 15 events, analytic qubit benchmark, event expectation tests |
| BH practical history geometry | O1=0.313262; O2=0.271243; Gram~2305.89 | Independently computed | `results/geometry.json`, unit tests |
| BH local history geometry | O1=0.942613; O2=0.914842; Gram~392026.28 | Independently computed | `results/geometry.json`, unit tests |
| Ising local H3 geometry | O1=0.969165; O2=0.724417; Gram~29006.58 | Independently computed | `results/geometry.json`, unit tests |
| Ising nonlocal H3 geometry | O1=0.969165; O2=0.113790; Gram~1565.41 | Independently computed | `results/geometry.json`, unit tests |
| Spectral coefficient optimization | SM Eq. S24 | New LP eigenvector cutting-plane solver; weighted fixed-shot objective; upper/lower certificates | Tested against exact solvable qubit; **not** original coefficient set |
| Shared repeated-projective metrology | SM Eq. S25--S27 | Setting-level ranges; convex fixed-coefficient shot allocation; heuristic alternating coefficient/shot optimization | Analytic scalar allocation tested; full joint global optimum **not** certified |
| Nested supports / lag screen | SM Note 6 | ERJ<=ERJK2, H2<=H3 inheritance, searched grid may include nonlocal H3 lag | Numeric screen; not original continuation/refinement solver |
| Eight-corner BH data sheet | SM Notes 8--9 | New 8-corner fixed-shot fit, trilinear interpolation, nominal anchor | 2-time, 1-event pilot run; full 81x15 campaign not executed |
| BH independent holdout | SM Note 10 | New 64 Sobol + 8 near-corner samples, positive tube states at trace distance 0.0098, independent Magnus solver | Two-point pilot run; original perturbation directions and frozen coefficients unknown |
| BH continuous-domain certificate 0.7304% | 125 offline anchors, directional sensitivity and 20% inflation | NOT reconstructed | Original envelope values and deterministic interior checks absent; do not claim result |
| BH shared shot benchmark 3.786e7 | Full multi-event co-design | NOT reproduced | Reconstructed objective/shot architecture differs in undocumented implementation choices |
| Ising selection boundaries | SM Table S4 across g/J=0.75--2 | Model and finite-screen tool built; full original parameter-map optimizer NOT reconstructed | Existing CSV remains external reference; demo does not regenerate phase table |
| Independent Solver-B selected ten Ising constraints | SM Note 12 | Separate Magnus propagation implemented | Ten target/operator/LP comparisons not recomputed with exact original event coefficients |

## Outstanding numerical research work

1. Recover precise calibration-state and robustness handling for the coefficient optimizer, including norm conventions, global setting enumeration, and exact family-wise confidence count `M`.
2. Implement and independently cross-check the full continuous-time and continuous-parameter sensitivity envelope, including how directional maxima are bounded between grid points.
3. Reproduce the *same* frozen eight-corner table using the original loss, optimizer seeds and constraints (if available), then repeat 72-point blind testing without changing it.
4. Run the full Ising lag/support/shot search with broad screening, local refinement, continuation, and cross-seeding. Validate all claimed family/support boundary brackets.
5. Run source-independent two-node Gauss-Magnus / alternative SDP audit for the ten selected spectral-distance constraints using the *exact* corresponding coefficients.
6. Compare all new outputs with the original ten CSVs and SM Tables S1--S6; publish a discrepancies table, with numerical tolerances, before calling the new release fully reproducible.

## Data provenance and release safety

The pre-existing ten GitHub CSV files are **reference outputs from the original numerical campaign**. They are never fabricated or rewritten by this reconstruction. Newly generated results use new filenames in `results/` and must be assessed independently.

The public `v1.0-submission` tag was already published before this reconstruction, and its accompanying prose originally overstates source completeness. The appropriate publication practice is to issue a transparent correction and use a new version tag for verified reconstructed source instead of rewriting an existing published reference without explanation.

## Completed reconstructed campaigns (additional evidence)

The full 8x3x15 **new** fixed-shot corner table has now been computed: 360 fits, with no unresolved cutting-plane gap above 2e-7. The new independent 72-parameter, 4-state, 3-time, 15-event holdout contains 12,960 evaluated event predictions and has maximum sampled absolute error 0.0550%. A 125-point parameter audit of that reconstructed sheet (5,625 event-time cases) found maximum pairwise tube upper bound 0.7080% at the tested points. **These values do not recover the paper's original 81-time campaign or 0.7304% continuous-domain certificate.** See `VALIDATION_REPORT.md` for numerical tables, provenance and remaining gaps.
