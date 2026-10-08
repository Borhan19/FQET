# Finite Quantum Engineering Thermodynamics (FQET)

This repository accompanies the manuscript **Finite Quantum Engineering Thermodynamics**.

## Current status

The figure-generation scripts and **all ten CSV input files** are uploaded. They allow the supplied plotting program to recreate the manuscript plots, subject to the software dependencies described below. The original microscopic simulation, optimization, calibration, and independent-validation programs are **not yet included**. Therefore, this repository is **not yet a complete end-to-end numerical reproduction package**, and no submission release should be tagged until those programs and their run instructions have been added and checked.

## Where the files are

The project files are currently in [`FQET_GitHub_upload_starter/`](FQET_GitHub_upload_starter/):

- [`FQET_manuscript_all_plots.py`](FQET_GitHub_upload_starter/FQET_manuscript_all_plots.py): creates four multi-panel manuscript plots plus ten standalone quantitative PDF plots from CSV inputs.
- [`FQET_schematics.py`](FQET_GitHub_upload_starter/FQET_schematics.py): creates illustrative Bose--Hubbard and Ising schematics.
- [`data/`](FQET_GitHub_upload_starter/data/): the ten uploaded CSV input files.
- [`simulations/`](FQET_GitHub_upload_starter/simulations/): currently documentation only; the microscopic numerical code is still needed.
- [`requirements.txt`](FQET_GitHub_upload_starter/requirements.txt): Python dependencies.

## Recreate the figures

Use Python 3.10+ and run these commands from the **repository root**:

```bash
cd FQET_GitHub_upload_starter
python -m pip install -r requirements.txt
python FQET_manuscript_all_plots.py --data-dir data --out-dir output
python FQET_schematics.py --output-dir output --no-show
```

On Windows, `py` can be substituted for `python`.

The main figure script uses Matplotlib's external LaTeX rendering (`text.usetex=True`). Install a working TeX distribution (such as MiKTeX or TeX Live) with the required math packages before running it. Generated PDF figures are written to `output/`.

The presence of the CSV files has been checked; full regeneration of all numerical results has **not** yet been independently verified from this repository.

## Complete numerical reproducibility

Before making a final submission release, add the original Bose--Hubbard and Ising simulation programs, validation scripts, parameter/settings files, and documented execution order. Confirm their output against the manuscript. See [UPLOAD_CHECKLIST.md](FQET_GitHub_upload_starter/UPLOAD_CHECKLIST.md).

## Versioned submission release

After verifying the complete source and data, publish a GitHub release tagged **`v1.0-submission`**. Until then, do not cite a release URL as if the complete reproducibility package has already been published.
