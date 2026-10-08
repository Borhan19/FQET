# Finite Quantum Engineering Thermodynamics (FQET)

This repository is intended to accompany the manuscript **Finite Quantum Engineering Thermodynamics**.

## Status of this starter package

This package contains the manuscript's supplied figure-plotting script and a separate script for optional illustrative schematics. **It is not yet a full numerical reproduction package:** the ten required CSV datasets and the original microscopic simulation, calibration and validation scripts have not been supplied. Add and test those files before describing the repository as sufficient to reproduce all numerical results.

## Contents

- `FQET_manuscript_all_plots.py`: generates the four main multi-panel figures and ten supplementary single-panel PDF plots from CSV data.
- `FQET_schematics.py`: generates the Bose--Hubbard and Ising model illustrations as vector PDF files and PNG previews, independently of CSV data.
- `data/`: place the ten required CSV datasets here (listed in `data/README.md`).
- `simulations/`: place the original numerical calculation and validation programs here (see `simulations/README.md`).
- `requirements.txt`: Python dependencies for both plotting scripts.

## System requirements

- Python 3.10 or newer is recommended.
- Install Python packages with `python -m pip install -r requirements.txt` (or `py -m pip install -r requirements.txt` on Windows).
- The main multi-panel plotting script uses `matplotlib` with `text.usetex=True`. It therefore needs a working external LaTeX installation (e.g., MiKTeX on Windows or TeX Live) with the `amsmath`, `amssymb`, and `amsfonts` packages, as well as the ancillary rendering tools required by Matplotlib's TeX support.
- The optional model-schematic script does not require LaTeX.

## Recreate manuscript figures from data

From the repository root, after adding the actual CSV files to `data/`:

```bash
python FQET_manuscript_all_plots.py --data-dir data --out-dir output
```

PDF figures are saved under `output/`. By default, the main script also displays figures in an interactive/notebook environment.

## Recreate model schematics (optional)

```bash
python FQET_schematics.py --output-dir output --no-show
```

This creates two vector PDF illustrations plus PNG previews. These are model schematics, not numerical many-body state snapshots.

## Recreate numerical data and checks

**Pending:** insert the actual microscopic scripts, numerical settings, CSV datasets, and execution instructions. The plotting script alone cannot reproduce the published simulation results or independent validation certificates.

## Versioning

For the manuscript submission release, create a GitHub release tagged `v1.0-submission` *only after* all required inputs and instructions have been uploaded and verified. The release tag freezes the code and data associated with the stated manuscript version.

## License

No license is included in this starter package. The repository owner should choose a license if public reuse and redistribution are intended.
