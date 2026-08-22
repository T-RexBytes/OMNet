# Cross-Validation Summary Across Models

## 1. OMNet-V2 5-Fold Cross-Validation (200X CV Pool, $N=1,733$)
- **EfficientNet-B0:** Val Accuracy: **84.77% ± 2.44%**, Val ROC-AUC: **0.9754 ± 0.0042**, Val F1: **87.74% ± 2.28%**
- **ViT-B/16:** Val Accuracy: **94.00% ± 1.33%**, Val ROC-AUC: **0.9915 ± 0.0027**, Val F1: **95.49% ± 1.03%**

## 2. OMNet-V3 5-Fold Stratified Group CV (Multi-Scale, $N=7,909$)
- **Binary Accuracy:** **84.56% ± 2.48%** (range: `[81.54%, 87.99%]`)
- **Binary Macro-F1:** **80.32% ± 3.13%** (range: `[76.40%, 84.60%]`)
- **Binary Balanced Accuracy:** **81.07% ± 3.26%** (range: `[75.53%, 85.16%]`)
- **Binary MCC:** **0.6175 ± 0.0532** (range: `[0.5485, 0.6949]`)
- **Subtype Accuracy:** **44.01% ± 6.38%** (range: `[35.84%, 53.75%]`)
- **Subtype Weighted-F1:** **44.75% ± 5.89%** (range: `[37.64%, 54.50%]`)
- **Subtype Macro-F1:** **23.45% ± 4.30%** (range: `[19.58%, 31.64%]`)
