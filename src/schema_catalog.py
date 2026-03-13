from __future__ import annotations

from pathlib import Path
import pandas as pd
from spacepy import pycdf
import hashlib
import json
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "processed" / "cdf_manifest.csv"

OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"
OUT_ERRORS = PROJECT_ROOT / "data" / "processed" / "file_schema_errors.csv"

# Tune this if you want smaller runs first
MAX_FILES = None  # e.g. 500 for a quick test, or None for all 21324


def schema_signature(keys: list[str]) -> str:
    """Stable signature of the schema (order-independent)."""
    norm = sorted(keys)
    blob = "\n".join(norm).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    if MAX_FILES is not None:
        df = df.head(MAX_FILES).copy()

    total = len(df)
    print("Manifest rows to process:", total)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    err_rows = []

    t0 = time.time()
    for i, r in df.iterrows():
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"...{i+1}/{total} files processed (elapsed {elapsed:,.1f}s)")

        path = r["path"]
        try:
            with pycdf.CDF(path) as cdf:
                keys = list(cdf.keys())

            sig = schema_signature(keys)

            rows.append({
                "group": r["group"],
                "model": r["model"],
                "vin": r["vin"],
                "year_folder": r["year_folder"],
                "date_yyyymmdd": str(r["date_yyyymmdd"]),
                "time_hhmm": str(r["time_hhmm"]),
                "file_ts": f"{r['date_yyyymmdd']}{r['time_hhmm']}",
                "filename": r["filename"],
                "path": path,
                "n_signals": len(keys),
                # store raw signal list as JSON string (easy to parse later)
                "signals_json": json.dumps(sorted(keys)),
                "schema_sig": sig,
            })

        except Exception as e:
            err_rows.append({
                "group": r.get("group", ""),
                "model": r.get("model", ""),
                "vin": r.get("vin", ""),
                "filename": r.get("filename", ""),
                "path": path,
                "error": str(e),
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PARQUET, index=False)

    if err_rows:
        pd.DataFrame(err_rows).to_csv(OUT_ERRORS, index=False)
        print("⚠️ Some files failed. Errors written to:", OUT_ERRORS)

    print("✅ Wrote:", OUT_PARQUET)
    print("Rows:", len(out), "Errors:", len(err_rows))


if __name__ == "__main__":
    main()
