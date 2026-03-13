# BrakeGuard EDA Pipeline

This project should be shared as code only. Raw telemetry data and generated data products must stay out of any public GitHub repository.

This document gives a single, clean path to go from raw CDF files to EDA-ready parquet outputs for exhaust-brake machine-learning work.

## Scope

This pipeline is intentionally narrow:

- Input: raw truck telemetry CDF files under `data/raw/`
- Output: VIN-level parquet files and exhaust-brake event datasets under `data/analysis/`
- Focus: exhaust-brake candidate events for EDA, feature engineering, and downstream ML

It does not change the existing project scripts. It adds one wrapper that runs the stable sequence in the correct order.

Reference copies of superseded scripts are stored under `src/legacy/`.

## Short Name

Project/team short name: `BrakeGuard`

## Expected Raw Data Layout

Place the proprietary raw CDF files under this structure:

```text
data/
  raw/
    Group1/
      MODEL####/
        VIN#####/
          2020/
            *.cdf
          2021/
            *.cdf
    Group2/
      ...
```

The wrapper scans recursively, so the important part is that the CDFs live under `data/raw/`.

## Environment Setup

Recommended on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-eda.txt
```

If `spacepy` fails to install with `pip`, install it through a Conda environment and then install the remaining packages there. The raw-data readers depend on `spacepy.pycdf`.

## End-to-End Run

From the project root:

```powershell
python .\src\run_brakeguard_eda_pipeline.py
```

That runs these stages:

1. Build raw-file manifest
2. Build per-file schema catalog
3. Build signal coverage catalog
4. Derive the core signal list
5. Build VIN-level `timeseries.parquet`
6. Score exhaust-brake candidate events
7. Build event-level sample windows
8. Build monthly health trends
9. Build correlation tables for EDA

## Useful Variants

Rebuild only the later stages after VIN parquet files already exist:

```powershell
python .\src\run_brakeguard_eda_pipeline.py --from-stage event_scores
```

Preview the commands without running them:

```powershell
python .\src\run_brakeguard_eda_pipeline.py --dry-run
```

Rebuild event samples with a smaller cap for testing:

```powershell
python .\src\run_brakeguard_eda_pipeline.py --from-stage event_samples --max-events-per-vin 100
```

Also write the full raw event-sample dataset:

```powershell
python .\src\run_brakeguard_eda_pipeline.py --from-stage event_samples --write-full-samples
```

## Primary Outputs

After a successful run, the project should have:

- `data/processed/cdf_manifest.csv`
- `data/processed/file_schema_catalog.parquet`
- `data/processed/signal_catalog.csv`
- `data/processed/core_signal_list.csv`
- `data/analysis/VIN*/timeseries.parquet`
- `data/analysis/vin_event_scores.csv`
- `data/analysis/vin_event_samples_slim.parquet/`
- `data/analysis/vin_health_trends.csv`
- `data/analysis/correlations_by_vin.csv`
- `data/analysis/correlations_population_pearson.csv`
- `data/analysis/correlations_population_spearman.csv`

## What Each Output Is For

- `VIN*/timeseries.parquet`: analysis-ready per-VIN telemetry tables built from raw CDF files
- `vin_event_scores.csv`: one row per candidate exhaust-brake event, with event-level metrics
- `vin_event_samples_slim.parquet`: row-level windows around scored events for EDA and ML
- `vin_health_trends.csv`: monthly rates such as `effective_rate` and `deep_rate`
- `correlations_*.csv`: filtered correlation summaries for exploratory analysis and dashboard use

## Dashboard

Once the pipeline outputs exist:

```powershell
streamlit run .\src\app_vin_exhbrake_dashboard.py
```

## GitHub Sharing Guidance

Because the raw data is restricted, do not commit:

- `data/raw/`
- large generated parquet outputs under `data/analysis/`
- local virtual environments such as `.venv/` or `myenv/`

Only share:

- `src/`
- `docs/`
- notebooks that do not embed data
- lightweight metadata or small CSV outputs if allowed internally

## Recommended Workflow

1. Clone the code repository.
2. Receive raw data through the approved private channel.
3. Place the CDF files under `data/raw/`.
4. Create the Python environment and install dependencies.
5. Run `python .\src\run_brakeguard_eda_pipeline.py`.
6. Use the resulting parquet and CSV outputs for notebooks, feature engineering, and the Streamlit dashboard.
