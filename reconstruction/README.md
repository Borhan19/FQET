# FQET numerical reconstruction (independent implementation)
> **Updated 9 October 2026:** The complete reconstructed 8 x 81 x 15 calibration and independent 72-point/125-point local validations were completed on the user computer. The historical notes below describe the earlier pilot package. For current verified numbers, read `RESULTS_MANIFEST.json` and the repository-level `README_GITHUB_READY.md`. They are still not a reconstruction of the original joint optimization or global certificate.



This project **reconstructs** the computational methods of the 9 October 2026 manuscript *Finite Quantum Engineering Thermodynamics* (FQET) by Borhan Ahmadi. The original microscopic calibration/optimization/validation source was unavailable when this reconstruction was developed. This is **not the original numerical code**, and its reconstructed numeric outputs must not be confused with the manuscript's recorded results.

## What is implemented

The implementation uses the same closed finite systems, with \(\hbar=J=1\):

- **Bose--Hubbard**: four bosons and four open-chain sites, Hilbert dimension 35, \(H=-JT+(U/2)D+\Lambda(t)N_R\), \(U_0/J=1.3/3\), \(\Lambda_0/J=20\), \(J\tau_{BC}=20\), \(k_B T_h/J=5\). The decision is at \(Jt_c=0.2\), forecasting to \(J\tau=0.2\).
- **Ising**: four spins, Hilbert dimension 16, \(H=-J\sum Z_iZ_{i+1}-g\sum X_i+\Lambda(t)N_\uparrow\), ramp \(\Lambda/J=+2\to-2\) during \(J\tau_{drv}=4\), initial \(k_BT/J=2\), decision at \(Jt_c=1\).
- Five projectors \(P_r\) for the spatial or up-spin record, with 15 nontrivial binary events up to complements.
- Energy, record, first and second derivatives \(i[H,P_r]\), \(-[H,[H,P_r]]\), and historical back-transported record operators \(U(t_c,t_c-s)P_rU(t_c,t_c-s)^\dagger\).
- Independent **DOP853** propagation and **two-node fourth-order Gauss--Magnus** propagation. The second method rebuilds Hamiltonian matrices via a separate code path.
- Principal-angle and Hilbert--Schmidt Gram diagnostics, including the distinct conditioning applied to the second historical and derivative layers. The reported BH and Ising geometry numbers have been independently recovered.
- Spectral-norm approximation using eigenvector cutting planes and a linear-programming lower bound. The returned objective has both a computationally evaluated upper and LP lower bound.
- Coefficient-dependent measurement ranges for groupwise projective records or separate Hermitian settings; common multi-event shot allocation using a constrained numerical optimizer.
- Screening H2, H3, ERJ, and ERJK2 representations, with explicit nesting of active supports. The outer problem uses numerical alternation and finite lag screening; **it does not certify the original global optimum**.
- Eight-corner Bose--Hubbard tables with trilinear interpolation and a separate independent holdout using new deterministic Sobol points and positive tube preparations.

## Quick installation and verified checks

Run these commands in the extracted package directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_reconstruction.py smoke
python run_reconstruction.py geometry
```

On Windows you can use `py` instead of `python` for the first and third/fourth commands, and `py -m pytest -q` for the tests.

The supplied unit tests check Hamiltonian construction, independent propagator agreement, state positivity, work/energy consistency, the history/target identities, a solvable analytic qubit spectral-distance problem, exact scalar shot allocation, trilinear interpolation, tube positivity, Magnus convergence, and reproduced manuscript geometry benchmarks.

## Task-aware architecture pilot

```bash
python run_reconstruction.py compare --model ising --g 1.5 --horizons 0.03 --events 1 3 --history-grid 0.06 0.20 0.70
```

The pilot uses **two events at one horizon**, not the paper's complete many-event task, so its repetition numbers are *not* comparable to Table S4. The program reports both raw alternating costs and the effective nested-support envelope.

For an exploratory BH comparison use `--model bh --horizons 0.05 --events 1 3 --history-grid 0.05 0.10 0.20`.

## Constitutive calibration and independent holdout

A **small, executable** BH eight-corner demonstration is:

```bash
python run_reconstruction.py calibrate --horizons 0.05 0.20 --events 1 --sheet results/bh_sheet_demo.npz
python run_reconstruction.py holdout --sheet results/bh_sheet_demo.npz --max-points 2 --out-csv results/reconstructed_bh_holdout_demo.csv
```

The first program computes real operator approximations for all eight corner Hamiltonians. The second performs an independent Gauss--Magnus holdout, with **new** seeds and tube perturbations. The demonstration is not the original 72-point campaign.

To calibrate the **full declared output grid** of 81 future times and 15 events for eight corners, run:

```bash
python run_reconstruction.py calibrate --full --sheet results/bh_sheet_full.npz
python run_reconstruction.py holdout --sheet results/bh_sheet_full.npz --max-points 72 --out-csv results/reconstructed_bh_holdout_full.csv
```

**Warning:** these are computationally expensive (9,720 independent cutting-plane fits, followed by 72 holdout propagations and 349,920 event evaluations). They have **not** been completed or verified here. The full published holdout may not be recovered because the original frozen coefficients, tube-state directions, optimizer settings and seeds are unavailable.

### What the eight-corner implementation does and does not establish

The `calibrate` command uses a fixed-shot spectral optimization with optional nominal-state anchoring and interpolates those coefficients. In the source manuscript, the final response and metrology are jointly optimized with additional numerical refinements. Their exact original regularization, coefficient records, optimizer seeds and sensitivity envelopes were not distributed. Accordingly:

- The reconstructed response tables are **new fits** to the stated microscopic Hamiltonians.
- The 125-point validation anchors and 20% sensitivity inflation are described in the manuscript, but **the actual numerical directional sensitivity envelopes required for the quoted continuous-domain 0.7304% certificate are not recovered**.
- No result here should be cited as independently reproducing the manuscript's reported 0.7304% global engineering certificate, 3.786e7 shared repetitions, or the 0.1278% full blind error until a matching validation is completed.
- The Ising g-dependent transition map is retained as *reference data*, not regenerated by the small demo scan. A complete outer search with the original tolerance, weights, events, lag grid, local-basin optimization and independent spectral audits remains to be completed.

## Existing manuscript plots and reference data

The original `FQET_manuscript_all_plots.py` and `FQET_schematics.py` are included for convenience, **unchanged**. Download the ten manuscript CSVs from the previously published commit:

```bash
python fetch_reference_data.py --into data
python FQET_manuscript_all_plots.py --data-dir data --out-dir output
python FQET_schematics.py --output-dir output --no-show
```

**The figure script uses Matplotlib's external LaTeX text rendering;** a working TeX distribution is required. `fetch_reference_data.py` requires Internet access. The raw published files are data to audit, not proof that the original simulations are reproduced.

## Scientific/reproducibility status

See [`RECONSTRUCTION_LEDGER.md`](RECONSTRUCTION_LEDGER.md). This independently reconstructed code is available on the repository's default `main` branch and still requires full scientific review before it can support the manuscript's strongest reproducibility claims. In particular, do not silently replace the already published `v1.0-submission` tag with new reconstructed code; publish a new distinctly labeled revision following successful full reproducibility tests.

## Citation/permissions

The project accompanies the named research manuscript. No redistribution license is asserted here; the repository owner must choose the appropriate license before encouraging third-party reuse.

## Verified reconstructed numerical campaign (9 October 2026)

A broader newly computed campaign is bundled in `results/`:

- `results/bh_sheet_15events.npz`: 8 corner tables, 3 times and 15 events (360 fresh spectral fits).
- `results/reconstructed_bh_holdout_15events_72points.csv`: 72 parameter points, 4 preparations, 3 times, 15 events, **12,960 newly evaluated predictions**; maximum sampled error 0.0550%.
- `results/reconstructed_parameter_grid_125.csv`: 125-parameter empirical grid, 5,625 target cases, maximum sampled pairwise preparation-tube upper bound 0.7080%.
- `results/geometry.json`, `results/smoke.json`, and `results/ising_full_events_tau002.json`: independent operator geometry, propagation checks, and a limited Ising screening example.

**Neither** the sampled 0.7080% result nor the reconstructed 0.0550% holdout result is the published continuous-domain 0.7304% certificate or the full 0.1278% original holdout campaign. They are separately computed results for different coefficient tables and a smaller forecast-time grid. The full scientific distinctions appear in [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

You can reproduce the 125-point empirical scan from the 15-event fitted table:

```bash
python run_reconstruction.py grid-audit --sheet results/bh_sheet_15events.npz --grid-size 5 --out-csv results/reconstructed_parameter_grid_125.csv
```

Long calibration searches save a checkpoint **after each forecast horizon**. To continue an interrupted response table, run the *same* command with `--resume`. Use `--max-corners 1` to compute only one unfinished corner before returning. The full Ising family screen can similarly checkpoint each candidate via `--checkpoint results/ising_screen_checkpoint.json` and resume by rerunning the identical command.
