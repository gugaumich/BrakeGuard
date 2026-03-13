# Vehicle Telemetry Schema Drift Analysis — Methodology

## 1. Data Description
The dataset consists of 21,249 daily vehicle telemetry files stored in
NASA Common Data Format (CDF), spanning multiple years and covering
96 unique vehicle identification numbers (VINs).

Each CDF file contains a set of telemetry signals defined in the file header,
along with time-series measurements.

## 2. Processing Objectives
The primary goal of this work is to:
- Validate schema consistency within individual VINs
- Detect signal additions, removals, and naming changes over time
- Preserve protocol-specific signal distinctions during processing

## 3. Metadata Indexing
All CDF files were indexed into a manifest containing:
- VIN
- Model
- File timestamp
- File path

Signal values were not loaded during indexing.

## 4. Schema Definition
A file schema is defined as the **set of raw signal names**
present in a CDF file header.

Signal order and signal values are ignored at this stage.

## 5. VIN-Scoped Schema Tracking
Schema consistency was evaluated independently for each VIN.

For each VIN:
- All file-level schemas were collected
- A union of all signals observed over time was computed
- Schema versions were identified using stable hash signatures

## 6. Schema Drift Detection
Schema drift events were defined as differences between consecutive files
for the same VIN.

Two drift types were recorded:
- Signal addition
- Signal removal

Each drift event was timestamped and logged.

## 7. Protocol Preservation
Signals containing protocol identifiers (e.g. `_1587`) were treated as
distinct and were not merged during data processing.

This ensures that protocol semantics are preserved for downstream analysis.

## 8. Variant Exploration
Signals containing statistical suffixes (e.g. Min, Max, Avg) were detected
and cataloged but not merged.

These variants were flagged for manual semantic review.

## 9. Outputs
The pipeline produces the following artifacts:
- VIN-level schema summaries
- Schema version frequency tables
- Timestamped drift event logs
- Signal variant inventory reports

These outputs form the basis for further semantic alignment and
analysis modeling.

## Schema Stability and Signal Drift Analysis
Schema Stability and Signal Drift Analysis

To quantify the stability and evolution of telemetry schemas over time, we performed a VIN-scoped schema drift analysis across all available CDF files. For each vehicle (VIN), we extracted the set of signal names present in every file and computed summary statistics characterizing schema variability, temporal drift, and signal reliability.

For each VIN, the union signal count represents the total number of distinct signal names observed across its entire history. This provides a maximal view of the available telemetry and captures signals introduced or removed over time. The schema version count measures the number of distinct signal configurations encountered, where each configuration corresponds to a unique set of signals. A high schema version count indicates frequent changes in the telemetry schema.

To assess dominance of a single schema, we computed the most common schema share, defined as the fraction of files that conform to the most frequently occurring signal set. VINs with high values (≥0.8) exhibit strong schema stability, while VINs with low values (<0.4) show significant fragmentation and ongoing schema evolution.

Temporal dynamics were captured using the number of schema change points, defined as the count of transitions between consecutive files where the signal set changed. This metric distinguishes VINs with gradual evolution from those experiencing frequent instrumentation or firmware updates.

Signal-level reliability was quantified using percentile-based metrics. Signals appearing in at least 95% of files for a VIN were classified as stable signals, representing a robust core suitable for baseline analysis. Conversely, signals appearing in fewer than 5% of files were classified as rare signals, typically reflecting experimental additions, late-stage instrumentation changes, or optional subsystems.

Across the fleet, substantial heterogeneity was observed. While some VINs maintained highly stable schemas throughout their operational history, many exhibited dozens of schema versions with no dominant configuration. These findings motivate a union-based schema representation, combined with semantic canonicalization, to support consistent longitudinal analysis while preserving raw signal provenance.