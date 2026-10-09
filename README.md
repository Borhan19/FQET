# Finite Quantum Engineering Thermodynamics (FQET)

This repository accompanies the manuscript *Finite Quantum Engineering Thermodynamics*.

## What is on this repository?

**The `main` branch now contains the independently reconstructed implementation**, assembled from the physical models and methods described in the manuscript and Supplemental Material. **It is not the unavailable original simulation source and does not establish exact end-to-end reproduction of all the manuscript's numerical claims.**

**Start here:** [Reconstruction code and instructions](reconstruction/README.md) | [Validation report](reconstruction/VALIDATION_REPORT.md) | [Reconstruction ledger](reconstruction/RECONSTRUCTION_LEDGER.md) | [Machine-readable results manifest](reconstruction/RESULTS_MANIFEST.json).

## Validated calculations

According to the completed local run and saved outputs in `reconstruction/results/`:

| Check | Reconstructed result |
| --- | ---: |
| Python unit tests | 11 passed |
| Bose--Hubbard eight-corner calibration | 8 corners × 81 forecast times × 15 events = 9,720 spectral fits |
| Spectral fits flagged nonconverged | 0 |
| Reconstructed holdout | 72 points × 4 preparations × 81 times × 15 events = 349,920 predictions |
| Largest absolute holdout error | 0.067949% |
| Finite parameter-grid audit | 125 points × 81 times × 15 events = 151,875 checks |
| Largest sampled pairwise preparation-tube upper bound | 0.729785% |
| Ising representation check | One finite lag-grid comparison at g/J = 1.5 and Jτ = 0.05 |

The sampled 0.729785% figure **is not a uniform continuous-domain certificate**. The original joint optimization of coefficients, lags, and shared measurement repetitions, the reported 0.7304% continuous-domain sensitivity envelope, the original 0.1278% holdout campaign, and the six-field Ising phase map have **not** been independently reconstructed to the original specification.

## Code, data, and reproducibility

- [`reconstruction/`](reconstruction/): model construction, independent time propagators, spectral optimization, calibration and validation programs, automated tests, saved numeric outputs and provenance notes.
- [`reconstruction/data/`](reconstruction/data/): ten **original tabulated reference CSV datasets** supplied separately from newly generated validation outputs.
- [`reconstruction/results/`](reconstruction/results/): reconstructed calibration tables and tests, with source and parameter information in the results manifest.
- [`reconstruction/FQET_manuscript_all_plots.py`](reconstruction/FQET_manuscript_all_plots.py): manuscript plots from the tabulated reference datasets.
- [`reconstruction/FQET_schematics.py`](reconstruction/FQET_schematics.py): illustrative working-medium diagrams.

The earlier plotting-focused package remains in [`FQET_GitHub_upload_starter/`](FQET_GitHub_upload_starter/) for provenance. It is not the reconstructed simulation package.

## Minimal verification

From the repository root:

```bash
cd reconstruction
python -m pip install -r requirements.txt
python -m pytest -q
python run_reconstruction.py smoke
python run_reconstruction.py geometry
```

The manuscript plot generator additionally requires a functioning LaTeX installation to render its text.

## Release status

The previously published [`v1.0-submission` release](https://github.com/Borhan19/FQET/releases/tag/v1.0-submission) is a separate, earlier snapshot and was **not modified** by this reconstruction upload. Its description should not be interpreted as proof that the missing original numerical programs were recovered. Any future release of this branch should explicitly say **independent reconstruction** and link the validation ledger.

No claim of rigorous global optimality for the outer representation-selection search, nor of a proved continuous-parameter domain envelope, follows from the tests on this branch.
