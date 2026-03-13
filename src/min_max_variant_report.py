from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
import re
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"

OUT_REPORT = PROJECT_ROOT / "data" / "processed" / "signal_variants_report.csv"

# We will detect suffix patterns but DO NOT merge anything.
# You can decide later.
VARIANT_SUFFIXES = ["MIN", "MAX", "AVG", "MEAN", "MEDIAN", "RMS", "STD", "STDEV", "COUNT", "SUM"]

# Tokenization heuristic: underscore + camelcase boundaries
CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def normalize_for_variant_detection(name: str) -> list[str]:
    """
    Normalize a raw signal name for variant detection:
    - keep protocol tags like _1587 as tokens (we do NOT remove them)
    - split on underscores and camelcase
    """
    parts = []
    for chunk in name.split("_"):
        parts.extend(CAMEL_SPLIT.sub(" ", chunk).split())
    return [p.upper() for p in parts if p]


def detect_variant_base(name: str) -> tuple[str, str] | None:
    """
    Try to map "Something_Min" -> base="Something", variant="MIN".
    Conservative: only if last token matches one of VARIANT_SUFFIXES.
    Does not strip protocol tags like 1587 unless it is part of the last token.
    """
    toks = normalize_for_variant_detection(name)
    if not toks:
        return None

    last = toks[-1]
    if last in VARIANT_SUFFIXES:
        # base is original name without the trailing variant part if it appears as suffix with underscore,
        # otherwise use a base label built from tokens excluding last.
        # We don't want to guess too aggressively, so we build a token-base key:
        base_key = " ".join(toks[:-1])
        return base_key, last

    return None


def main():
    if not CATALOG.exists():
        raise FileNotFoundError(
            f"Missing catalog: {CATALOG}\nRun: python src/schema_catalog.py"
        )

    df = pd.read_parquet(CATALOG)

    # Build a global set of all raw signals observed (union across files)
    all_signals = set()
    for sjson in df["signals_json"].tolist():
        all_signals.update(json.loads(sjson))

    # Map base_key -> dict(variant -> list[raw_signal])
    families = defaultdict(lambda: defaultdict(list))

    for sig in sorted(all_signals):
        d = detect_variant_base(sig)
        if d is None:
            continue
        base_key, variant = d
        families[base_key][variant].append(sig)

    # Build report rows
    rows = []
    for base_key, variants in families.items():
        # only report if we found at least 2 variant types (e.g., MIN and MAX) or multiple signals
        variant_types = sorted(variants.keys())
        raw_count = sum(len(v) for v in variants.values())
        if len(variant_types) < 2 and raw_count < 3:
            continue

        rows.append({
            "base_key": base_key,
            "variant_types": ",".join(variant_types),
            "raw_signal_count": raw_count,
            "examples": "; ".join([variants[vt][0] for vt in variant_types if variants[vt]]),
        })

    out = pd.DataFrame(rows).sort_values(["raw_signal_count", "base_key"], ascending=[False, True])
    out.to_csv(OUT_REPORT, index=False)

    print("✅ Wrote:", OUT_REPORT)
    print("Families reported:", len(out))


if __name__ == "__main__":
    main()
