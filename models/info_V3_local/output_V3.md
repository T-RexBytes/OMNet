# Experimental Results & Presentation Artifacts — OMNet-V3

> Complete experimental outputs, numerical performance tables, visual figure interpretations, and publication assets from the local execution of OMNet-V3 (`result_OMNet/local`).

---

## 1. Master Results Summary (5-Fold Stratified Group CV)

All metrics were computed across **5 patient-disjoint folds** evaluated strictly on unseen test patients:

| Metric Category | Metric Name | Mean | Std Dev | Min (Worst Fold) | Max (Best Fold) |
|---|---|---|---|---|---|
| **Binary (Benign vs Malignant)** | **Accuracy** | **83.82%** | ± 4.04% | 79.56% (Fold 0) | **90.97%** (Fold 4) |
| | **Macro-F1** | **81.24%** | ± 5.14% | 76.13% (Fold 0) | **89.40%** (Fold 4) |
| | **Balanced Accuracy** | **82.37%** | ± 4.38% | 76.48% (Fold 0) | **88.07%** (Fold 4) |
| | **Matthews Corr. Coeff. (MCC)**| **0.6364** | ± 0.1018 | 0.5468 (Fold 0) | **0.7928** (Fold 4) |
| **Subtype (8-Class)** | **Accuracy** | **42.86%** | ± 4.69% | 35.61% (Fold 2) | **47.68%** (Fold 3) |
| | **Weighted-F1** | **42.02%** | ± 5.40% | 35.51% (Fold 2) | **48.44%** (Fold 3) |
| | **Balanced Accuracy** | **30.50%** | ± 5.80% | 23.08% (Fold 2) | **37.48%** (Fold 3) |
| | **Macro-F1** | **25.63%** | ± 5.03% | 18.19% (Fold 2) | **30.28%** (Fold 0) |
| | **Matthews Corr. Coeff. (MCC)**| **0.2565** | ± 0.0570 | 0.1726 (Fold 2) | **0.3227** (Fold 0) |

---

## 2. Per-Fold Performance Breakdown

| Fold Index | Test Patients (Images) | Binary Accuracy | Binary Macro-F1 | Subtype Accuracy | Subtype Weighted-F1 | Subtype Macro-F1 | Best Val Epoch |
|---|---|---|---|---|---|---|---|
| **Fold 0** | 16 patients (1,556 imgs) | 79.56% | 76.13% | 46.21% | 46.03% | **30.28%** | Epoch 8 |
| **Fold 1** | 16 patients (1,529 imgs) | 83.19% | 80.64% | 42.18% | 40.89% | 26.54% | Epoch 10 |
| **Fold 2** | 16 patients (1,589 imgs) | 81.37% | 78.49% | 35.61% | 35.51% | 18.19% | Epoch 11 |
| **Fold 3** | 18 patients (1,702 imgs) | 84.02% | 81.56% | **47.68%** | **48.44%** | 29.11% | Epoch 12 |
| **Fold 4** | 16 patients (1,533 imgs) | **90.97%** | **89.40%** | 42.60% | 39.22% | 24.03% | Epoch 9 |
| **Average**| — | **83.82%** | **81.24%** | **42.86%** | **42.02%** | **25.63%** | — |

---

## 3. Generated Visual Artifacts & Diagnostic Analysis

The local experiment run produced 6 publication figures stored in `result_OMNet/local/`:

### 3.1 Binary Confusion Matrix (`confusion_matrix_binary.png`)
- **Key Observation:**
  - High sensitivity for malignant cases (low false-negative rate), essential for clinical cancer screening.
  - Minor false-positive rate where certain complex benign lesions (e.g., adenosis with florid hyperplasia) exhibit cellular density mimicking malignant tissue.
  - Normalized confusion percentages confirm consistent true positive / true negative diagonal dominance across all 5 test folds.

### 3.2 8-Class Subtype Confusion Matrix (`confusion_matrix_8class.png`)
- **Key Observation:**
  - **Ductal Carcinoma (DC):** Highest individual recognition accuracy (>65%), consistent with its high prevalence in the training distribution (38 patients).
  - **Fibroadenoma (F):** High discrimination accuracy among benign subtypes due to characteristic biphasic epithelial and stromal architecture.
  - **Inter-Class Confusion:** Primary confusion occurs between morphologically adjacent subtypes:
    - *Lobular Carcinoma (LC)* occasionally mispredicted as *Ductal Carcinoma (DC)* due to shared invasive cellular features.
    - *Adenosis (A)* vs *Tubular Adenoma (TA)* due to shared tubular/glandular proliferations.

### 3.3 Receiver Operating Characteristic (ROC) & PR Curves (`roc_curves.png`)
- **Binary ROC Curve:** Exhibits high area under the curve ($\text{AUC} > 0.88$), indicating strong discrimination threshold flexibility across clinical operating points.
- **Multiclass One-vs-Rest (OvR) Curves:** Distinct separation curves for each of the 8 subtypes, demonstrating that the dual-branch embedding separates classes even when hard decision boundaries are challenged by patient variability.

### 3.4 Fusion Gate Scale Analysis (`fusion_gate_analysis.png`)
- **Core Scientific Finding:**
  - **At 40X Magnification:** Gating parameter $\alpha$ shifts lower ($\alpha \approx 0.35 - 0.45$), meaning the network gives **higher weight to ViT global attention** $(1 - \alpha)$, leveraging wide-field tissue architecture and glandular density.
  - **At 400X Magnification:** Gating parameter $\alpha$ shifts higher ($\alpha \approx 0.60 - 0.72$), meaning the network gives **higher weight to EfficientNet-B0 local convolutional features**, focusing on nuclear atypia, pleomorphism, and chromatin clumping.
  - **At 100X & 200X:** Shows a smooth, monotonic transition balancing local and global feature maps.

### 3.5 t-SNE Latent Space Visualization (`tsne_embeddings.png`)
- **256-D Fused Feature Space Distribution:**
  - Clear macro-clustering separating Benign (blue cluster) from Malignant (red cluster).
  - Fine-grained sub-clusters show distinct groupings for Fibroadenoma (F) and Ductal Carcinoma (DC).
  - Overlap regions correspond directly to the clinical boundary ambiguity between complex benign proliferations and well-differentiated carcinomas.

---

## 4. Hardware & Environmental Execution Record

| Property | Value |
|---|---|
| **Host System** | AMD ROCm / Linux Local Workspace |
| **Compute Device** | AMD Radeon RX 9060 XT (16 GB VRAM, `gfx1200`) |
| **PyTorch Version** | `2.13.0+rocm7.2` |
| **HIP Version** | `7.2.53211` |
| **Execution Precision** | Full FP32 (stable, zero NaN gradients, zero GPUVM page faults) |
| **Per-Epoch Training Time** | ~98 - 103 seconds (warmup epochs 1–5), ~188 - 194 seconds (unfrozen backbones) |
| **Total 5-Fold Training Time**| ~2.8 hours total execution time |
| **Peak VRAM Consumption** | ~5.8 GiB (well within the 16 GB hardware budget) |

---

## 5. Summary Table for Publication

```
========================================================================================
Model Architecture: OMNet-V3 (EfficientNet-B0 + ViT-Tiny/16 + MAF Module + Hierarchical Loss)
Dataset Protocol  : BreaKHis (82 Patients, 7,909 Images) | 5-Fold Stratified Group CV
Hardware Platform : AMD Radeon RX 9060 XT (ROCm 7.2) | FP32 Precision
========================================================================================
Task                     Accuracy (%)       Macro-F1 (%)       Balanced Acc (%)   MCC
----------------------------------------------------------------------------------------
Binary Malignancy (2-Cls) 83.82 ± 4.04%      81.24 ± 5.14%      82.37 ± 4.38%      0.6364 ± 0.10
Subtype Histology (8-Cls) 42.86 ± 4.69%      25.63 ± 5.03%      30.50 ± 5.80%      0.2565 ± 0.06
========================================================================================
```
