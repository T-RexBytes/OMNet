# Experimental Results & Presentation Artifacts — OMNet-V3

> Comprehensive experimental results, numerical performance tables, visual figure interpretations, and publication assets for the OMNet-V3 model.

---

## 1. Master Results Table (5-Fold Stratified Group CV)

All metrics were evaluated across **5 patient-disjoint folds** strictly on unseen clinical test patients:

| Metric Category | Metric Name | Updated Cloud Run (`result_OMNet/new`) |
|---|---|---|
| **Binary Malignancy (2-Class)** | **Accuracy** | **84.56% ± 2.48%** (range: 81.54% - 87.99%) |
| | **Macro-F1** | **80.32% ± 3.13%** (range: 76.40% - 84.60%) |
| | **Balanced Accuracy** | **81.07% ± 3.26%** (range: 75.53% - 85.16%) |
| | **Matthews Corr. Coeff. (MCC)**| **0.6175 ± 0.0532** (range: 0.5485 - 0.6949)|
| **Histological Subtype (8-Class)**| **Accuracy** | **44.01% ± 6.38%** (range: 35.84% - 53.75%) |
| | **Weighted-F1** | **44.75% ± 5.89%** (range: 37.64% - 54.50%) |
| | **Balanced Accuracy** | **32.93% ± 4.70%** (range: 27.96% - 41.22%) |
| | **Macro-F1** | **23.45% ± 4.30%** (range: 19.58% - 31.64%) |
| | **Matthews Corr. Coeff. (MCC)**| **0.2738 ± 0.0504** (range: 0.2111 - 0.3614)|

---

## 2. Scale Gating Analysis: Learned Alpha Across Optical Magnifications

The learned gating parameter $\alpha(m) \in (0, 1)$ demonstrates how the network dynamically balances localized convolutional features vs. global transformer attention across zoom levels:

| Optical Magnification | Scale Index | Mean Alpha $\alpha(m)$ | Standard Deviation | Primary Representation Prioritized |
|---|---|---|---|---|
| **40X** | `0` | **0.6850** | ± 0.1301 | **ViT Global Context:** Glandular layout & tissue margins |
| **100X** | `1` | **0.5991** | ± 0.1283 | **Intermediate Balance:** Periductal arrangement & stromal boundary |
| **200X** | `2` | **0.5258** | ± 0.1449 | **Intermediate Balance:** Cellular density & tubular lumens |
| **400X** | `3` | **0.4701** | ± 0.1459 | **CNN Local Features:** High-frequency nuclear atypia & pleomorphism |

*Key finding:* The model displays a smooth, monotonic scale-conditioned gating response across the four magnification regimes.

---

## 3. Visual Artifact Catalog & Diagnostic Interpretations

The experimental runs produced high-resolution publication figures stored in `result_OMNet/new/`:

### 3.1 Binary Confusion Matrix (`main_conff.png`)
- **Diagnostic Findings:**
  - Strong diagonal dominance demonstrating consistent true-positive and true-negative classification across all 5 test folds.
  - Very low false-negative rate for malignant specimens, essential for high-sensitivity clinical cancer screening.
  - Minor false-positive occurrences where florid hyperplasia in complex benign lesions mimics low-grade carcinomas.

### 3.2 8-Class Subtype Confusion Matrix (`sub_conff.png`)
- **Diagnostic Findings:**
  - **Ductal Carcinoma (DC):** Highest overall sensitivity, aided by its larger patient sample size in the training distribution.
  - **Fibroadenoma (F):** High accuracy among benign lesions due to characteristic biphasic epithelial-stromal patterns.
  - **Inter-Subtype Ambiguity:** Expected clinical confusions occur between *Lobular Carcinoma (LC)* and *Ductal Carcinoma (DC)*, and between *Adenosis (A)* and *Tubular Adenoma (TA)* due to shared glandular architecture.

### 3.3 Receiver Operating Characteristic (ROC) Curves (`Binary ROC Curve (Aggregated Folds).png` & `Subtype OvR ROC Curves.png`)
- **Binary ROC Curve:** Exhibits high aggregated AUC ($> 0.88$), indicating robust discrimination across various operating thresholds.
- **Multiclass One-vs-Rest (OvR) Curves:** Distinct ROC curves for all 8 subtypes, demonstrating class separability in the fused embedding space.

### 3.4 Adaptive Fusion Gate Progression (`Adaptive Fusion Gate Alpha across Magnification Levels.png`)
- Visual boxplot/violin plot illustrating the monotonic transition of $\alpha$ distributions across 40X, 100X, 200X, and 400X.

### 3.5 t-SNE Latent Space Visualization (`t-SNE of OMNet-V3 Fused Embeddings by Subtype.png`)
- 2D manifold projection of the 256-dimensional fused feature space:
  - Macro-level clustering clearly separates Benign (blue) from Malignant (red) classes.
  - Subtype clusters for Fibroadenoma and Ductal Carcinoma form well-defined, dense clusters.

---

## 4. Hardware & Computational Efficiency Summary

| Environment / Hardware Specification | Cloud / GPU Environment |
|---|---|
| **Compute Device** | NVIDIA CUDA GPU / Colab |
| **Execution Precision** | AMP / FP32 |
| **Total 5-Fold Training Time** | ~2.2 hours |

---

## 5. Publication Summary Block

```
==================================================================================================
Model Framework   : OMNet-V3 (EfficientNet-B0 + ViT-Tiny/16 + MAF Module + Hierarchical Loss)
Dataset Benchmark : BreakHis (82 Clinical Patients, 7,909 Images across 40X, 100X, 200X, 400X)
Validation Scheme : 5-Fold Stratified Group Cross-Validation (Strict Zero Patient Leakage)
==================================================================================================
Diagnostic Task               Accuracy (%)        Macro-F1 (%)        Balanced Acc (%)    MCC
--------------------------------------------------------------------------------------------------
Binary Malignancy (2-Class)   84.56 ± 2.48%       80.32 ± 3.13%       81.07 ± 3.26%       0.6175 ± 0.05
--------------------------------------------------------------------------------------------------
Histological Subtype (8-Class) 44.01 ± 6.38%       23.45 ± 4.30%       32.93 ± 4.70%       0.2738 ± 0.05
  [Subtype Weighted-F1]       44.75 ± 5.89% (Peak: 54.50%)
==================================================================================================
```
