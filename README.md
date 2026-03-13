# BrakeGuard

BrakeGuard is an exhaust-brake telemetry analysis project for heavy-duty truck fleet data. The repository contains the code needed to transform raw CDF telemetry into VIN-level parquet files, extract exhaust-brake candidate events, and produce EDA-ready datasets for downstream machine learning.

## Repository Purpose

This repository is intended to share code, documentation, and reproducible analysis steps.

It must not be used to publish:

- proprietary raw telemetry data
- large generated parquet outputs
- private inspection or fleet-identifying data unless explicitly approved

## Quick Start

Assuming the raw CDF files are already in place under `data/raw/Group1/` and `data/raw/Group2/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-eda.txt
python .\src\run_brakeguard_eda_pipeline.py
```

After the pipeline finishes, launch the dashboard with:

```powershell
streamlit run .\src\app_vin_exhbrake_dashboard.py
```

## Primary Workflow

The recommended entrypoint is the pipeline wrapper:

```powershell
python .\src\run_brakeguard_eda_pipeline.py
```

That wrapper runs the project's stable raw-data-to-EDA sequence:

1. Build the raw CDF manifest
2. Build the per-file schema catalog
3. Build the signal coverage catalog
4. Derive the core signal list
5. Build VIN-level `timeseries.parquet`
6. Score exhaust-brake candidate events
7. Build event-level sample windows
8. Build monthly health trends
9. Build correlation tables for EDA

## Setup

Recommended on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-eda.txt
```

If `spacepy` fails to install with `pip`, use a Conda environment for `spacepy` and install the remaining packages there.

## Raw Data Layout

Place the proprietary raw CDF files under:

```text
data/
  raw/
    Group1/
    Group2/
```

The pipeline scans recursively below `data/raw/`.

## Main Outputs

After a successful run, the main outputs are:

- `data/analysis/VIN*/timeseries.parquet`
- `data/analysis/vin_event_scores.csv`
- `data/analysis/vin_event_samples_slim.parquet/`
- `data/analysis/vin_health_trends.csv`
- `data/analysis/correlations_by_vin.csv`

## Dashboard

After the analysis outputs exist:

```powershell
streamlit run .\src\app_vin_exhbrake_dashboard.py
```

## Documentation

- `docs/exhaust_brake_eda_pipeline.md`
- `docs/methodology.md`
- `docs/research_log.md`

## Notes

- The wrapper script is the supported entrypoint for rebuilding the analysis pipeline.
- Reference copies of superseded scripts are stored in `src/legacy/`.
- The supported run path remains the current scripts in `src/`, especially `src/run_brakeguard_eda_pipeline.py`.
