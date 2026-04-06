# NQFO Implied Volatility Surface Completion

This repository contains my end-to-end workflow and final submission pipeline for the **National Quant Finance Olympiad 2026** implied-volatility surface completion task.

The goal is to predict `iv_predicted` for every row in `test.csv` where `iv_observed` is missing, using:

- historical train data
- non-IV row features such as moneyness, maturity, and option type
- visible same-date anchors already present in the test set

The final submission pipeline is implemented in [solution.py](solution.py) and the reusable helpers live in [src/nqfo/pipeline.py](src/nqfo/pipeline.py) and [src/nqfo/io.py](src/nqfo/io.py).

## Problem framing

This is best treated as **time-ordered partial IV surface completion**, not generic row-wise regression.

For each date, the option surface is a stable lattice over:

- 15 moneyness levels
- 4 maturities: `1M`, `2M`, `3M`, `6M`
- 2 option types: `call`, `put`

That gives **120 rows per date**.

Observed repository data:

- `train.csv`: 11,640 rows across 97 dates (`2025-01-02` to `2025-05-16`)
- `test.csv`: 3,960 rows across 33 dates (`2025-05-19` to `2025-07-02`)
- missing IV rows in train: 5,148
- missing IV rows in test: 1,699

Because the test set still contains many visible IV rows, the task is closer to **future-date surface completion with anchors preserved** than full-surface forecasting.

## Final model

The locked final model is a **slice-gated hybrid IV completion model**.

In the codebase and notebooks, this model appears under the internal experiment name:

- `hybrid_slice_no_hard_case_override_pruned`

High-level design:

1. Build a structured same-date predictor from visible anchors.
2. Build a pseudo-supervised training table using time-ordered masking on train dates.
3. Fit two ML branches on top of the structured representation.
4. Route each missing test row through a fixed hybrid rule.

### Structured branch

The structured side of the model combines:

- same-date linear interpolation within maturity/option slices
- quadratic smile fitting over `log(moneyness)`
- maturity interpolation in **total variance**
- call-put shrinkage when the opposite option IV is visible at the same node

This produces the `structured_winner` prediction used both directly and as the baseline for residual learning.

### ML branches

Two locked ML branches are trained on the pseudo-training table:

- **Ridge** direct-IV model
- **pruned HistGradientBoostingRegressor** residual model on top of the structured baseline

The final inference routing is:

- use the structured branch for center-region rows and quadratic-only structured cases
- use the Ridge branch when the structured source is `tv_maturity_only`
- otherwise use the pruned HistGB residual branch

Final predictions are floored at `5.0` and validated against the submission schema and row ordering.

## Validation philosophy

Model development was done notebook-first with a validation setup designed to mirror the competition:

- split by **date**, not row
- preserve strict **time ordering**
- hide only part of later-date observations
- keep the remaining same-date IV values visible as anchors

Two locked validation protocols were carried through model selection:

- `primary_realistic`
- `stress_test`

In the Phase 6 confirmation notebook, the final hybrid candidate outperformed the main standalone alternatives on the two-protocol summary used for selection.

## Repository structure

```text
.
├── solution.py                     # final runnable submission entrypoint
├── requirements.txt               # runtime dependencies
├── src/nqfo/
│   ├── io.py                      # data loading, submission checks, file resolution
│   └── pipeline.py                # locked final feature + inference pipeline
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_validation_design.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_surface_models.ipynb
│   ├── 05_0_ml_feature_pipeline.ipynb
│   ├── 05_1_feature_family_ablations.ipynb
│   ├── 06_0_final_candidate_confirmation.ipynb
│   ├── 06_1_final_training_policy_and_tuning.ipynb
│   ├── 07_MatrixFactorisation.ipynb
│   └── 08_graphbased_models.ipynb
├── docs/                          # planning notes and project context
├── methodology.pdf                # submission methodology write-up
├── train.csv / test.csv           # root-level data inputs
├── data/                          # optional fallback location for inputs
└── submission.csv                 # generated or archived submission output
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input files

`solution.py` looks for these files in either the project root or `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv` (optional but recommended)

If `sample_submission.csv` is present, its `row_id` ordering is used exactly when building the final submission.

## Run the final pipeline

```bash
python solution.py
```

Expected output:

- `submission.csv` written to the project root
- console logs showing resolved input files, row count, and output path

## Notebook workflow

The notebooks document the full modeling progression:

- `01_eda.ipynb`: lattice structure, missingness, and finance-aware diagnostics
- `02_validation_design.ipynb`: date-based anchor-preserving validation framework
- `03_baselines.ipynb`: simple reference models
- `04_surface_models.ipynb`: structured interpolation and surface-fitting models
- `05_*`: feature engineering and ML branch development
- `06_*`: finalist confirmation, final policy locking, and submission preparation
- `07_*`, `08_*`: exploratory advanced directions

## Notes

- The implementation uses only `numpy`, `pandas`, `scikit-learn`, and `scipy`.
- The final code is designed to reproduce the competition submission CSV from local data files.
- The repository contains both the final runnable pipeline and the notebook trail used to reach it.
