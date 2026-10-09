# FQET: independently reconstructed code and results

This folder contains an **independent reconstruction**, not the unavailable
original numerical source for *Finite Quantum Engineering Thermodynamics*.
It is prepared for review on the separate GitHub branch `reconstruction-2026-10-09`.
The earlier public `v1.0-submission` tag is **not changed**.

## Included and verified locally

- Bose--Hubbard: **8 parameter corners**, 81 forecast times and 15 independent
  events; **9,720** completed spectral fits with
  **0** unconverged fits in the stored calibration.
- Independently generated holdout: **349,920** event
  predictions on 72 parameter points and four preparations per point;
  maximum absolute error **0.067949%**.
- An empirical **125-parameter-point** grid over **151,875**
  event-time cases; maximum sampled pairwise tube bound
  **0.729785%**.
  *This does not prove a continuous-domain error bound.*
- Ising: a **single limited lag-grid scan** at `g/J=1.5`, `Jtau=0.05`,
  with up to 15 event masks. This is not the complete six-field phase map.
- The 10 **original tabulated CSVs**, as separate reference data, and the
  manuscript plotting and model-schematic source scripts.

## Scientific restrictions

The reconstructed response uses a fixed-shot spectral optimization, *not* the
unavailable original fully coupled coefficient/shot/lag co-design. The original
continuous-domain 0.7304% envelope, original 0.1278% holdout, complete Ising
transition boundaries and 3.786e7 shared-repetitions figure have **not** been
independently reproduced. Do not present this package as exact reexecution of
the original numerical campaign.

## Re-run the included code

From the `reconstruction` folder, with Python 3.10+:

```powershell
py -m pip install -r requirements.txt
py -m pytest -q
py run_reconstruction.py smoke
py run_reconstruction.py geometry
```

To display the manuscript plots from the provided *reference CSVs* (external
LaTeX installation required for text labels):

```powershell
py FQET_manuscript_all_plots.py --data-dir data --out-dir output
```

The calculation source and validation details are in `run_reconstruction.py`,
`fqet_reconstructed/`, `tests/`, `VALIDATION_REPORT.md` and
`RECONSTRUCTION_LEDGER.md`. Machine-readable checks and hashes appear in
`RESULTS_MANIFEST.json`.
