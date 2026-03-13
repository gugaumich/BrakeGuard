# Research Log — Vehicle Telemetry Schema Analysis (SIADS699)

Author: Guga Gugaratshan  
Project: Vehicle Telemetry Schema Drift Analysis  
Start Date: 2026-02-10

---

## 2026-02-10 — Project Initialization

### Objective
Establish a reproducible pipeline to assess schema consistency and
signal drift in large-scale vehicle telemetry data stored in CDF format.

### Dataset Overview
- ~21,249 daily CDF files
- 96 unique VINs
- Multi-year telemetry data
- Signals embedded in CDF headers

### Initial Constraints
- Corporate laptop with restricted environment
- Data stored in OneDrive (sync considerations)
- Processing must be memory-safe and restartable

---

## 2026-02-11 — Environment Setup & Data Indexing

### Decisions
- Use Python virtual environment (`myenv`)
- Use `spacepy.pycdf` to read CDF headers only
- Build a manifest file before any heavy processing

### Rationale
Loading signal values prematurely is inefficient.
Schema-level analysis only requires header metadata.

### Outputs
- `cdf_manifest.csv`
- Verified access to all CDF files via recursive scan

---

## 2026-02-12 — Schema Strategy Definition

### Key Design Decisions
- **Schema scope:** VIN-level
- **Baseline:** Union of all signals observed per VIN
- **Protocol handling:** Preserve `_1587` signals as distinct
- **Canonicalization:** Semantic (proposal only, no automatic merging)

### Rationale
Signals evolve over time. Union preserves full telemetry history.
Protocol identifiers encode real semantic differences and must not be merged
during processing.

---

## 2026-02-28 — Schema Catalog & Drift Detection

### Work Completed
- Extracted signal lists from 21,249 CDF files
- Built schema signatures per file
- Detected schema change events per VIN

### Observations
- Some VINs show stable schemas
- Others show multiple schema versions across years
- 75 files failed to load (documented separately)

### Open Questions
- Are signal name changes due to protocol updates or semantic changes?
- Should Min/Max/Avg signals be treated as derived or independent?

---

## Open Issues / TODO
- Review `vin_schema_changes.csv`
- Inspect `signal_variants_report.csv`
- Propose canonical signal mapping (human-reviewed)
