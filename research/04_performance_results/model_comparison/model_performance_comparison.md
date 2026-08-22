# Master Results Table: OMNet-V2 vs. OMNet-V3

## 1. Comprehensive Model Performance Comparison

| Model Identity | Architecture Configuration | Evaluation Scope | Binary Accuracy | Binary Macro-F1 | Sensitivity (Recall) | Specificity | ROC-AUC | PR-AUC | Subtype Weighted-F1 | Subtype Macro-F1 | Matthews Corr. Coeff. (MCC) | Comparability / Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OMNet-V2** | EfficientNet-B0 (CNN) | BreakHis 200X (Test $N=280$) | 81.79% | 87.94% (Bin F1) | 94.90% | 51.19% | 0.8712 | 0.9356 | — | — | 0.5391 | Directly observed |
| **OMNet-V2** | ViT-B/16 (Transformer) | BreakHis 200X (Test $N=280$) | **84.29%** | **89.67%** (Bin F1) | **97.45%** | 53.57% | **0.8998** | 0.9488 | — | — | **0.6105** | Directly observed |
| **OMNet-V2** | Ensemble (Soft-Voting) | BreakHis 200X (Test $N=280$) | **84.29%** | 89.57% (Bin F1) | 96.43% | **55.95%** | 0.8973 | **0.9537** | — | — | 0.6084 | Directly observed |
| **OMNet-V3** | Dual-Branch MAF Framework | BreakHis Multi-Scale (5-Fold CV) | **84.56% ± 2.48%** | **80.32% ± 3.13%** | — | — | **>0.88 (OvR)** | — | **44.75% ± 5.89%** | **23.45% ± 4.30%** | **0.6175 ± 0.0532** | Directly observed (Cloud Run) |
| **OMNet-V3** | Dual-Branch MAF Framework | BreakHis Multi-Scale (5-Fold CV) | 83.82% ± 4.04% | 81.24% ± 5.14% | — | — | >0.88 (OvR) | — | 42.02% ± 5.40% | 25.63% ± 5.03% | 0.6364 ± 0.1018 | Directly observed (Local Provenance) |

> **Comparability Note:** OMNet-V2 solves binary classification strictly on 200X magnification patches ($N=2,013$), whereas OMNet-V3 simultaneously solves multi-task binary detection and 8-class histological subtyping across all four magnifications ($40	ext{X}, 100	ext{X}, 200	ext{X}, 400	ext{X}$; $N=7,909$). Both are evaluated with zero patient data leakage.
