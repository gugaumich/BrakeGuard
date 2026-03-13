from pathlib import Path
import csv
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "cdf_manifest.csv"

# This just *tries* to parse date/time if present; it will still write the row even if it can't.
DATE_PAT = re.compile(r"_(\d{8})_(\d{4})_", re.IGNORECASE)

def main():
    print("Project root:", PROJECT_ROOT)
    print("Scanning:", BASE)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(BASE.rglob("*.cdf"))
    print("Found CDF files:", len(files))
    if not files:
        print("❌ No CDF files found. Stopping.")
        return

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "model", "vin", "year_folder", "date_yyyymmdd", "time_hhmm", "filename", "path"])

        for idx, p in enumerate(files, start=1):
            # Expect data/raw/<group>/MODEL####/VIN#####/<year>/file.cdf
            parts = p.parts
            i = parts.index("data")
            group = parts[i + 2]
            model = parts[i + 3]
            vin = parts[i + 4]
            year_folder = parts[i + 5]

            m = DATE_PAT.search(p.name)
            yyyymmdd, hhmm = (m.group(1), m.group(2)) if m else ("", "")

            w.writerow([group, model, vin, year_folder, yyyymmdd, hhmm, p.name, str(p)])

            if idx % 5000 == 0:
                print(f"...written {idx}/{len(files)} rows")

    print("✅ Manifest written to:", OUT)
    print("Rows:", len(files))

if __name__ == "__main__":
    main()
