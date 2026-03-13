from pathlib import Path
import re
import csv

# Always anchor paths to the PROJECT ROOT (one level above /src)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "cdf_manifest.csv"

PAT = re.compile(
    r"^(MODEL\d+)_"
    r"(VIN\d+)_"
    r"(\d{8})_"
    r"(\d{4})_"
    r"(FAMILY\d+)_"
    r"([A-Z])_"
    r"TS_Daily\.cdf$",
    re.IGNORECASE
)

def main():
    print("Project root:", PROJECT_ROOT)
    print("Looking for CDFs under:", BASE)

    if not BASE.exists():
        print("❌ BASE folder does not exist:", BASE)
        print("Check that your folder is: data/raw/Group1 and data/raw/Group2 under Project.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Will write manifest to:", OUT)

    files = list(BASE.rglob("*.cdf"))
    print("Found CDF files:", len(files))

    if not files:
        print("❌ No .cdf files found under data/raw (recursive).")
        return

    bad = 0
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "model", "vin", "year_folder", "date_yyyymmdd", "time_hhmm", "family", "suffix", "path"])

        for idx, p in enumerate(files, start=1):
            m = PAT.match(p.name)
            if not m:
                bad += 1
                continue

            model, vin, yyyymmdd, hhmm, family, suffix = m.groups()

            # Expect: data/raw/<group>/MODEL####/VIN#####/<year>/file.cdf
            parts = p.parts
            try:
                i = parts.index("data")
                group = parts[i + 2]        # raw/<group>
                year_folder = parts[i + 5]  # <year>
            except Exception:
                group = ""
                year_folder = ""

            w.writerow([group, model.upper(), vin.upper(), year_folder, yyyymmdd, hhmm, family.upper(), suffix.upper(), str(p)])

            if idx % 2000 == 0:
                print(f"...processed {idx}/{len(files)}")

    print("✅ Manifest written:", OUT)
    print("Unmatched filenames:", bad)

if __name__ == "__main__":
    main()
