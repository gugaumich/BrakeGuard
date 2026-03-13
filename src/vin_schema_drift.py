from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"

OUT_VIN_SUMMARY = PROJECT_ROOT / "data" / "processed" / "vin_schema_summary.csv"
OUT_VIN_VERSIONS = PROJECT_ROOT / "data" / "processed" / "vin_schema_versions.csv"
OUT_VIN_CHANGES = PROJECT_ROOT / "data" / "processed" / "vin_schema_changes.csv"


def load_signals(signals_json: str) -> set[str]:
    return set(json.loads(signals_json))


def main():
    print(">>> vin_schema_drift.py START <<<")
    print("Reading catalog:", CATALOG)

    if not CATALOG.exists():
        raise FileNotFoundError(f"Missing catalog: {CATALOG}. Run schema_catalog.py first.")

    df = pd.read_parquet(CATALOG)
    print("Catalog rows:", len(df))
    if df.empty:
        raise RuntimeError("Catalog is empty.")

    # Ensure file_ts is a sortable string
    df["file_ts"] = df["file_ts"].astype(str)
    df = df.sort_values(["vin", "file_ts"]).reset_index(drop=True)

    vin_summaries = []
    version_rows = []
    change_rows = []

    vins = df["vin"].unique().tolist()
    print("Unique VINs:", len(vins))

    for idx_vin, vin in enumerate(vins, start=1):
        g = df[df["vin"] == vin].sort_values("file_ts")
        n_files = len(g)
        if n_files == 0:
            continue

        # schema versions within VIN
        sig_counter = Counter(g["schema_sig"].tolist())
        for sig, cnt in sig_counter.items():
            version_rows.append({
                "vin": vin,
                "schema_sig": sig,
                "n_files": cnt,
                "share": cnt / n_files
            })

        # union + lifetime + presence
        union_set: set[str] = set()
        first_seen = {}
        last_seen = {}
        present_count = defaultdict(int)

        prev_set = None
        prev_ts = None
        prev_filename = None

        change_points_ts = set()

        for _, r in g.iterrows():
            s = load_signals(r["signals_json"])
            ts = r["file_ts"]
            filename = r["filename"]

            union_set |= s

            for sig in s:
                present_count[sig] += 1
                if sig not in first_seen:
                    first_seen[sig] = ts
                last_seen[sig] = ts

            if prev_set is not None:
                added = s - prev_set
                removed = prev_set - s
                if added or removed:
                    change_points_ts.add(ts)

                    for a in sorted(added):
                        change_rows.append({
                            "vin": vin,
                            "file_ts": ts,
                            "filename": filename,
                            "change_type": "added_vs_prev",
                            "signal": a,
                            "prev_file_ts": prev_ts,
                            "prev_filename": prev_filename,
                        })
                    for rem in sorted(removed):
                        change_rows.append({
                            "vin": vin,
                            "file_ts": ts,
                            "filename": filename,
                            "change_type": "removed_vs_prev",
                            "signal": rem,
                            "prev_file_ts": prev_ts,
                            "prev_filename": prev_filename,
                        })

            prev_set = s
            prev_ts = ts
            prev_filename = filename

        # stable/rare counts
        stable_count = sum(1 for sig, c in present_count.items() if (c / n_files) >= 0.95)
        rare_count = sum(1 for sig, c in present_count.items() if (c / n_files) <= 0.05)

        vin_summaries.append({
            "vin": vin,
            "n_files": n_files,
            "union_signal_count": len(union_set),
            "schema_version_count": len(sig_counter),
            "most_common_schema_share": max(sig_counter.values()) / n_files,
            "n_change_points": len(change_points_ts),
            "stable_signal_count_95pct": stable_count,
            "rare_signal_count_5pct": rare_count,
        })

        if idx_vin % 50 == 0 or idx_vin == len(vins):
            print(f"...processed VINs {idx_vin}/{len(vins)}")

    # Write outputs
    print("Writing outputs to:", OUT_VIN_SUMMARY.parent)

    pd.DataFrame(vin_summaries).to_csv(OUT_VIN_SUMMARY, index=False)
    pd.DataFrame(version_rows).to_csv(OUT_VIN_VERSIONS, index=False)
    pd.DataFrame(change_rows).to_csv(OUT_VIN_CHANGES, index=False)

    print("✅ Wrote:")
    print("-", OUT_VIN_SUMMARY)
    print("-", OUT_VIN_VERSIONS)
    print("-", OUT_VIN_CHANGES)
    print(">>> vin_schema_drift.py DONE <<<")


if __name__ == "__main__":
    main()
