# Experimental Traceability Matrix

| Claim / Metric / Finding | Model | Source File | Source Field / Location | Calculation Type | Verification Status |
|---|---|---|---|---|---|
| **ViT-B/16 Test Accuracy: 84.29%** | OMNet-V2 | `v2_out/test_metrics_all_models.csv` | Row `vit_b_16`, Col `accuracy` | Direct measurement | `Directly observed` |
| **ViT-B/16 Test Recall: 97.45%** | OMNet-V2 | `v2_out/test_metrics_all_models.csv` | Row `vit_b_16`, Col `recall` | Direct measurement | `Directly observed` |
| **ViT-B/16 Test ROC-AUC: 0.8998** | OMNet-V2 | `v2_out/test_metrics_all_models.csv` | Row `vit_b_16`, Col `roc_auc` | Direct measurement | `Directly observed` |
| **Ensemble Test Specificity: 55.95%** | OMNet-V2 | `v2_out/test_metrics_all_models.csv` | Row `ensemble`, Col `specificity` | Direct measurement | `Directly observed` |
| **Ensemble Test PR-AUC: 0.9537** | OMNet-V2 | `v2_out/test_metrics_all_models.csv` | Row `ensemble`, Col `pr_auc` | Direct measurement | `Directly observed` |
| **EfficientNet-B0 CV Acc: 84.77% +- 2.44%** | OMNet-V2 | `v2_out/cv_summary.csv` | Row `efficientnet_b0`, Cols `mean_val_acc`, `std_val_acc` | 5-Fold Mean +- Std | `Directly observed` |
| **ViT-B/16 CV Acc: 94.00% +- 1.33%** | OMNet-V2 | `v2_out/cv_summary.csv` | Row `vit_b_16`, Cols `mean_val_acc`, `std_val_acc` | 5-Fold Mean +- Std | `Directly observed` |
| **OMNet-V3 CV Binary Acc: 84.56% +- 2.48%** | OMNet-V3 | `V3_out/result.txt` | Line `binary_accuracy` | 5-Fold Mean +- Std | `Directly observed` |
| **OMNet-V3 CV Macro-F1: 80.32% +- 3.13%** | OMNet-V3 | `V3_out/result.txt` | Line `binary_macro_f1` | 5-Fold Mean +- Std | `Directly observed` |
| **OMNet-V3 Subtype Weighted-F1: 44.75% +- 5.89%** | OMNet-V3 | `V3_out/result.txt` | Line `subtype_weighted_f1` | 5-Fold Mean +- Std | `Directly observed` |
| **OMNet-V3 Subtype Macro-F1: 23.45% +- 4.30%** | OMNet-V3 | `V3_out/result.txt` | Line `subtype_macro_f1` | 5-Fold Mean +- Std | `Directly observed` |
| **Scale Gating Alpha: 40X=0.685, 400X=0.470** | OMNet-V3 | `V3_out/fusion.txt` | Lines `40X`, `400X` | Empirical mean alpha | `Directly observed` |
| **V3 Local Binary Acc: 83.82% +- 4.04%** | OMNet-V3 | `V3_local_out/aggregated_results.json` | Key `binary_accuracy` | 5-Fold Mean +- Std | `Directly observed (Provenance)` |
| **BreakHis 200X Image Count: 2,013** | Dataset | `v2_out/split.csv`, `dataset_metadata.csv` | Total row count at 200X | Exact record count | `Directly observed` |
| **BreakHis Multi-Scale Count: 7,909** | Dataset | `V3_local_out/dataset_metadata.csv` | Total row count across all scales | Exact record count | `Directly observed` |
| **Unique Patient Count: 82** | Dataset | `V3_local_out/dataset_metadata.csv` | `patient_id.nunique()` | Distinct patient groups | `Directly observed` |
