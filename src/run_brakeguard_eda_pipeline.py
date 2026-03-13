"""
Run the BrakeGuard exhaust-brake raw-data -> EDA pipeline.

This wrapper leaves the existing project scripts untouched and executes a
reliable event-focused sequence that can be run on another machine with the
same raw-data layout and dependencies installed.

Pipeline stages
---------------
1. manifest         -> data/processed/cdf_manifest.csv
2. schema_catalog   -> data/processed/file_schema_catalog.parquet
3. signal_catalog   -> data/processed/signal_catalog.csv
4. core_signals     -> data/processed/core_signal_list.csv
5. vin_parquet      -> data/analysis/VIN*/timeseries.parquet
6. event_scores     -> data/analysis/vin_event_scores.csv
7. event_samples    -> data/analysis/vin_event_samples_slim.parquet
8. health_trends    -> data/analysis/vin_health_trends.csv
9. correlations     -> data/analysis/correlations_*.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_TMP_ROOT = Path(tempfile.gettempdir()) / "vin_event_samples_tmp"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    script: str


STAGES = [
    Stage("manifest", "Scan raw CDF files and build a manifest", "build_manifest_simple.py"),
    Stage("schema_catalog", "Extract per-file signal schemas", "schema_catalog.py"),
    Stage("signal_catalog", "Aggregate signal coverage and emptiness statistics", "build_signal_catalog.py"),
    Stage("core_signals", "Derive the Tier-1 core signal list", "derive_core_signal_list.py"),
    Stage("vin_parquet", "Build VIN-level time-series parquet files", "build_vin_parquet.py"),
    Stage("event_scores", "Score exhaust-brake candidate events", "build_vin_event_scores.py"),
    Stage("event_samples", "Extract event-level sample windows", "build_vin_event_samples_v2.py"),
    Stage("health_trends", "Build monthly exhaust-brake health trends", "build_vin_health_trends.py"),
    Stage("correlations", "Build filtered correlation tables from slim event samples", "build_correlations_v2.py"),
]

STAGE_INDEX = {stage.name: idx for idx, stage in enumerate(STAGES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BrakeGuard exhaust-brake EDA pipeline from raw CDF data."
    )
    parser.add_argument(
        "--from-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[0].name,
        help="First stage to run.",
    )
    parser.add_argument(
        "--to-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[-1].name,
        help="Last stage to run.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the available stages and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child scripts.",
    )
    parser.add_argument(
        "--pad-seconds",
        type=int,
        default=20,
        help="Padding around each event window for event_samples.",
    )
    parser.add_argument(
        "--max-events-per-vin",
        type=int,
        default=0,
        help="Optional cap for event_samples stage. 0 means no cap.",
    )
    parser.add_argument(
        "--write-full-samples",
        action="store_true",
        help="Also write the full raw event-samples dataset.",
    )
    parser.add_argument(
        "--clean-event-samples",
        action="store_true",
        help="Delete prior event-sample outputs before rebuilding them.",
    )
    parser.add_argument(
        "--tmp-root",
        default=str(DEFAULT_TMP_ROOT),
        help="Temporary local directory used by the event_samples writer.",
    )
    return parser.parse_args()


def list_stages() -> None:
    print("Available pipeline stages:")
    for stage in STAGES:
        print(f" - {stage.name:<14} {stage.description}")


def selected_stages(start: str, stop: str) -> list[Stage]:
    start_idx = STAGE_INDEX[start]
    stop_idx = STAGE_INDEX[stop]
    if start_idx > stop_idx:
        raise ValueError("--from-stage must come before --to-stage")
    return STAGES[start_idx : stop_idx + 1]


def ensure_raw_data() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {RAW_ROOT}\n"
            "Expected raw CDF files under data/raw/Group*/MODEL*/VIN*/YEAR/*.cdf"
        )

    sample = next(RAW_ROOT.rglob("*.cdf"), None)
    if sample is None:
        raise FileNotFoundError(
            f"No .cdf files found under {RAW_ROOT}\n"
            "Expected raw CDF files under data/raw/Group*/MODEL*/VIN*/YEAR/*.cdf"
        )


def command_for(stage: Stage, args: argparse.Namespace) -> list[str]:
    cmd = [args.python, str(SRC_ROOT / stage.script)]
    if stage.name == "event_samples":
        cmd.extend(
            [
                "--pad",
                str(args.pad_seconds),
                "--max-events-per-vin",
                str(args.max_events_per_vin),
                "--tmp-root",
                str(args.tmp_root),
            ]
        )
        if args.write_full_samples:
            cmd.append("--write-full")
        if args.clean_event_samples:
            cmd.append("--clean")
    return cmd


def expected_outputs(stage_name: str) -> Iterable[Path]:
    if stage_name == "manifest":
        return [PROJECT_ROOT / "data" / "processed" / "cdf_manifest.csv"]
    if stage_name == "schema_catalog":
        return [PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"]
    if stage_name == "signal_catalog":
        return [PROJECT_ROOT / "data" / "processed" / "signal_catalog.csv"]
    if stage_name == "core_signals":
        return [PROJECT_ROOT / "data" / "processed" / "core_signal_list.csv"]
    if stage_name == "vin_parquet":
        return [PROJECT_ROOT / "data" / "analysis"]
    if stage_name == "event_scores":
        return [PROJECT_ROOT / "data" / "analysis" / "vin_event_scores.csv"]
    if stage_name == "event_samples":
        return [PROJECT_ROOT / "data" / "analysis" / "vin_event_samples_slim.parquet"]
    if stage_name == "health_trends":
        return [PROJECT_ROOT / "data" / "analysis" / "vin_health_trends.csv"]
    if stage_name == "correlations":
        return [PROJECT_ROOT / "data" / "analysis" / "correlations_by_vin.csv"]
    return []


def validate_stage(stage_name: str) -> None:
    outputs = list(expected_outputs(stage_name))
    for output in outputs:
        if not output.exists():
            raise FileNotFoundError(
                f"Stage '{stage_name}' completed without expected output: {output}"
            )

    if stage_name == "vin_parquet":
        vin_parquets = list((PROJECT_ROOT / "data" / "analysis").glob("VIN*/timeseries.parquet"))
        if not vin_parquets:
            raise FileNotFoundError(
                "Stage 'vin_parquet' did not produce any VIN*/timeseries.parquet outputs."
            )


def run_stage(stage: Stage, args: argparse.Namespace) -> None:
    cmd = command_for(stage, args)
    printable = subprocess.list2cmdline(cmd)
    print(f"\n[{stage.name}] {stage.description}")
    print(f"Command: {printable}")

    if args.dry_run:
        return

    started = time.time()
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    validate_stage(stage.name)
    print(f"[{stage.name}] done in {time.time() - started:,.1f}s")


def main() -> None:
    args = parse_args()

    if args.list_stages:
        list_stages()
        return

    ensure_raw_data()
    stages = selected_stages(args.from_stage, args.to_stage)

    print("BrakeGuard EDA pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {args.python}")
    print(f"Stages: {', '.join(stage.name for stage in stages)}")

    started = time.time()
    for stage in stages:
        run_stage(stage, args)

    print(f"\nPipeline finished in {time.time() - started:,.1f}s")
    print("Primary EDA-ready outputs:")
    print(f" - {PROJECT_ROOT / 'data' / 'analysis'}")
    print(f" - {PROJECT_ROOT / 'data' / 'analysis' / 'vin_event_samples_slim.parquet'}")
    print(f" - {PROJECT_ROOT / 'data' / 'analysis' / 'vin_event_scores.csv'}")
    print(f" - {PROJECT_ROOT / 'data' / 'analysis' / 'vin_health_trends.csv'}")


if __name__ == "__main__":
    main()
