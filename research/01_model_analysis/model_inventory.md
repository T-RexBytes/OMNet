# Source Model and Artifact Inventory

| Model Family / Source | Artifact Directory | Key Files Present | Evidence Type | Usability for Quantitative Claims |
|---|---|---|---|---|
| **OMNet-V2** | `models/info_V2/`, `result_OMNet/v2_out/` | `architecture_V2.md`, `dataset_V2.md`, `knowledge_V2.md`, `output_V2.md`, `test_metrics_all_models.csv`, `cv_fold_results.csv`, `experiment_summary.json`, `test_predictions_detailed.csv`, 10 PNG plots | Primary Tier 1 (CSV, JSON) + Tier 2 (PNG) + Tier 3 (MD) | **Fully Usable** (Directly verified from raw CSVs & JSONs) |
| **OMNet-V3 (Standard / Cloud)** | `models/info_V3/`, `result_OMNet/V3_out/` | `architecture_V3.md`, `dataset_V3.md`, `knowledge_V3.md`, `output_V3.md`, `result.txt`, `summary.txt`, `fusion.txt`, 6 PNG plots | Primary Tier 1 (TXT logs) + Tier 2 (PNG) + Tier 3 (MD) | **Fully Usable** (Verified against aggregated TXT logs & figure distributions) |
| **OMNet-V3 (Local Workstation)** | `models/info_V3_local/`, `result_OMNet/V3_local_out/` | `architecture_V3.md`, `dataset_V3.md`, `knowledge_V3.md`, `output_V3.md`, `aggregated_results.json`, `metrics_summary.csv`, `dataset_metadata.csv`, `experiment_config.json`, 5 PNG plots | Primary Tier 1 (JSON, CSV) + Tier 2 (PNG) + Tier 3 (MD) | **Fully Usable** (Complementary provenance under unified V3 family) |

> **Version-Family Normalization Note:** Per the research protocol in `work.md`, `V3` and `V3 local` are normalized under a single **V3 model family**. Both represent the same mathematical architecture (Dual-branch EfficientNet-B0 + ViT-Tiny/16 + MAF module + Hierarchical consistency loss) evaluated across 5-fold patient-disjoint splits.
