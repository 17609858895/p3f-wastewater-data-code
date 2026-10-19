# P3F wastewater nutrient forecasting: code and data

Reproducibility release for the revised P3F analysis of Tank-1 phosphate and ammonium at the Agtrup wastewater treatment plant, Denmark. The release contains analysis/plotting code and data, including the initial Huber configuration and the revised median/quantile configuration. Manuscripts, supplementary documents, rendered figures, editorial scripts, and fitted model binaries are excluded.

P3F uses phase/context and nutrient-history features in gradient-boosted increment models. The revised configuration selects the L1 point objective and inlet-phase conformalised quantile regression (CQR-phase) on four rolling-validation folds. It forecasts 10-minute mean concentrations at 0.5, 1, 2, 4, and 6 hours. Data cover one plant; the release provides future-time evaluation rather than cross-site or prospective deployment validation.

## Data provenance

The source is Mohammadi et al., *Wastewater Treatment Plant Data for Nutrient Removal System*, Mendeley Data v2, [10.17632/34rpmsxc4z.2](https://doi.org/10.17632/34rpmsxc4z.2), with descriptor [10.48550/arXiv.2407.05346](https://doi.org/10.48550/arXiv.2407.05346). Source and derived data retain the source attribution/noncommercial terms, described in [DATA_LICENSE.md](DATA_LICENSE.md).

The source CSV contains 525,600 rows from 1 August 2021 through 31 July 2023 UTC at 2-minute spacing. Preprocessing yields 104,796 complete rows and 195 issuance-available features: 65,053 training, 12,781 calibration, 26,028 test, and 934 unused purge rows. Each boundary has a 78-hour purge. The resampling timestamp is the start of the 10-minute bin; forecasts become available after that bin is complete. Future labels use the bin index plus the selected lead time.

## Install and verify

Python 3.12.3 was used. From the repository root:

```bash
python -m venv .venv
# Activate the environment with the command appropriate for your operating system.
python -m pip install -r requirements.txt
python scripts/assemble_data.py
python scripts/validate_release.py --preprocess
```

The two large derived Parquet files are split into exact byte fragments of at most 40 MiB so every tracked file stays below GitHub's ordinary file limit. Assembly verifies each part and the final SHA-256 before replacing the output. The assembled files are ignored by Git. No Git LFS, paid downloads, credentials, or local user paths are needed.

Validation checks the SHA-256 manifest, code syntax, raw timestamps and completeness, feature/split integrity, all comparator/P3F point metrics, all selected-interval subgroup metrics, validation selection, weekly bootstrap contrasts, and optionally the complete feature reconstruction from the source CSV. It does not retrain the full model suite. The pinned package versions describe the original analysis environment; this release has not been tested across every platform.

## Recompute analyses

The released predictions allow metric verification without training. To retrain all baseline and revised models, use a new directory; this preserves the released results:

```bash
python scripts/run_pipeline.py --output runs/full --xgb-device cuda
# CPU alternative; GPU/CPU floating point differences may affect XGBoost:
python scripts/run_pipeline.py --output runs/cpu --xgb-device cpu
# Add --extended for seed, Huber-threshold and uncontended timing refits.
```

Training is CPU intensive; only the XGBoost comparator uses CUDA in the original settings. No trained model binary is supplied. The runner rebuilds data, fits the initial comparators, snapshots their tables, performs rolling-validation selection, fits the revised model, and recomputes the ablation and reliability analyses. Original script numbering is retained for traceability. The archived results under `tables/v1_huber_split_conformal/` describe the initial Huber configuration. Refit timings are hardware dependent.

The recorded selection rules are in `tables/B_selection_protocol.json`; choices and validation results are in `tables/B_selected_configuration.json` and `tables/TableS16_validation_selection*.csv`. The future-test block had already been inspected for the initial Huber configuration before the revised protocol was written. The revised comparisons use that same test block; they are not an untouched prospective confirmatory test. The public Git commit timestamp records this release and does not independently prove when the original selection protocol was written.

## Files

| Path | Content |
| --- | --- |
| `data/raw/` | Unmodified two-minute source CSV |
| `data/processed/` | Ten-minute features, labels, split definitions and metadata |
| `data/predictions/` | Calibration/test predictions for both configurations and matched ablations |
| `tables/` | Machine-readable scientific results and figure source data |
| `scripts/` | Preprocessing, training, calibration, validation, interpretation and plotting code |
| `release_manifest.json` | Byte sizes and SHA-256 hashes of tracked release files |

Point metric tables retain original identifiers, including `P3F` for the revised model and `P3F v1 (Huber)` for its initial configuration where both appear. `T1_PO4` and `T1_NH4` are Tank-1 sensor concentrations; they are not automatically final effluent measurements. Observed-future high-concentration strata are retrospective diagnostics. The current-concentration stratum is available at issuance, but is not a validated operational warning policy.

For plotting, run `python scripts/generate_initial_figures.py --only Fig02`, `python scripts/35_refine_fig02.py`, and `python scripts/59_make_p3f_v2_figures.py --dpi 600` after data assembly. These scripts create local outputs under the ignored `figures/` directory. Fig. 1 is a conceptual illustration and is not needed to reproduce the numerical analysis. The legacy Fig. 1 drawing in `generate_initial_figures.py` is retained as historical code; use `--only Fig02` for current numerical work.
