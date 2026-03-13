from pathlib import Path
from spacepy import pycdf

GROUPS = ["Group1", "Group2"]
BASE = Path("data/raw")


def find_cdf_files(group: str) -> list[Path]:
    """Find all .cdf files under data/raw/<group> recursively."""
    root = BASE / group
    return sorted(root.rglob("*.cdf"))


def check_one_file(cdf_path: Path, n_keys: int = 10) -> None:
    """Open one CDF safely and print a few variable names."""
    with pycdf.CDF(str(cdf_path)) as cdf:
        keys = list(cdf.keys())
        print(f"Variables ({len(keys)} total): {keys[:n_keys]}")


def summarize_structure(cdf_path: Path) -> None:
    """
    Expecting structure like: .../<GROUP>/MODEL####/VIN#####/<YEAR>/file.cdf
    Print the group/model/vin/year for one sample file.
    """
    parts = cdf_path.parts
    # find index of "data" then "raw" then group
    # (works even with long absolute OneDrive paths)
    try:
        i = parts.index("data")
        group = parts[i + 2]          # data/raw/<group>
        model = parts[i + 3]          # MODEL####
        vin = parts[i + 4]            # VIN#####
        year = parts[i + 5]           # 2013, 2014, ...
        print(f"Sample path breakdown → group={group}, model={model}, vin={vin}, year={year}")
    except Exception:
        print("Sample path:", cdf_path)


def main():
    for group in GROUPS:
        print(f"\n--- {group} ---")
        files = find_cdf_files(group)

        if not files:
            print("❌ No CDF files found (recursive search).")
            continue

        print(f"✅ Found {len(files)} CDF files")
        print("First file:", files[0].name)
        summarize_structure(files[0])

        # sanity-check read:
        try:
            check_one_file(files[0])
        except Exception as e:
            print("❌ Error opening sample CDF:", e)


if __name__ == "__main__":
    main()
