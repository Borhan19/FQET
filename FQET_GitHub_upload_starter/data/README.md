# Required numerical CSV data

The plotting script `FQET_manuscript_all_plots.py` expects the following files in this `data/` folder:

- `bh_taskaware_family_comparison.csv`
- `bh_history_geometry.csv`
- `bh_blind_parameter_points.csv`
- `bh_blind_summary.csv`
- `crossplatform_geometry.csv`
- `ising_taskaware_phase_map.csv`
- `ising_h3_branch_curve.csv`
- `ising_solverB_spectral_audit.csv`
- `ising_h3_geometry_crossing.csv`
- `ising_h3_cost_crossing.csv`

**These ten CSV files are not included in the starter package.** Insert the actual analysis CSV files produced for the manuscript before claiming the full quantitative figures are reproducible. Do not substitute synthetic data.

The plotting script reads all ten files, even though some are not directly used in the four main composite figures.
