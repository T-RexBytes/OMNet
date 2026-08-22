# Dataset Specification — OMNet-V2

> Comprehensive dataset, preprocessing, splitting, and augmentation documentation for **OMNet-V2 on the BreakHis-200X Histopathology Benchmark**.

---

## 1. Dataset Overview: BreakHis 200X

| Property | Value |
|---|---|
| **Dataset Title** | Breast Cancer Histopathological Image Database (BreakHis) |
| **Kaggle Source / Identifier** | `trexbytes/breakhislink` |
| **Experimental Magnification**| **200X Optical Magnification** |
| **Tissue Modality** | Formalin-fixed paraffin-embedded (FFPE) Hematoxylin and Eosin (H&E) stained surgical breast biopsy specimens |
| **Raw Image Format** | $700 \times 460$ pixels, 24-bit RGB (PNG format) |
| **Processed Patch Resolution** | $224 \times 224$ pixels |
| **Total Images (200X Scale)** | **2,013 images** |
| **Total Unique Patients** | **82 unique clinical patients** |
| **Partitioning Allocation** | **1,733 images (CV Pool) / 280 images (Held-Out Test Set)** |

---

## 2. Class Composition & Taxonomy

In OMNet-V2, the diagnostic objective is binary malignancy detection across benign and malignant cohorts:

```
BreakHis 200X Cohort (2,013 images across 82 patients)
│
├── Benign (Class 0): 623 images (30.95%) / 24 patients
│   ├── Adenosis (A)            : 111 images (5.51%),   4 patients
│   ├── Fibroadenoma (F)        : 253 images (12.57%), 10 patients
│   ├── Phyllodes Tumor (PT)    : 121 images (6.01%),   3 patients
│   └── Tubular Adenoma (TA)    : 138 images (6.86%),   7 patients
│
└── Malignant (Class 1): 1,390 images (69.05%) / 58 patients
    ├── Ductal Carcinoma (DC)   : 883 images (43.86%), 38 patients
    ├── Lobular Carcinoma (LC)  : 156 images (7.75%),   5 patients
    ├── Mucinous Carcinoma (MC) : 211 images (10.48%),  9 patients
    └── Papillary Carcinoma (PC): 140 images (6.95%),   6 patients
```

### Class Distribution Summary

| Class Index | Binary Class Name | Subtypes Included | Total Images (200X) | Class Proportion | Patient Count |
|---|---|---|---|---|---|
| `0` | **Benign** | Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma | 623 | 30.95% | 24 |
| `1` | **Malignant** | Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma | 1,390 | 69.05% | 58 |
| — | **Total** | All 8 Histological Subtypes | **2,013** | **100.0%** | **82** |

---

## 3. Patient-Level Stratified Partitioning Protocol

To avoid the critical data leakage problem where multiple image patches from the same biopsy specimen appear across both training and testing partitions, OMNet-V2 uses a **Patient-Level Stratified Grouping Scheme**:

```
Full BreakHis 200X Cohort (2,013 images / 82 patients)
  │
  ├──► Held-Out Test Partition: 280 images (~14%) / ~12 patients (Zero Leakage)
  │
  └──► Cross-Validation Pool: 1,733 images (~86%) / ~70 patients
         │
         └──► 5-Fold Stratified Cross-Validation (Grouped by Patient)
```

### Partitioning Guarantees
1. **Zero Patient Overlap:** $\text{Patients}(\text{CV Pool}) \cap \text{Patients}(\text{Test Set}) = \emptyset$.
2. **Stratified Subtype Preservation:** Both benign and malignant subtypes are proportionately distributed across the CV pool and test partitions.
3. **Split Manifest Persistence:** Exported to `split.csv` in `result_OMNet/v2_out/` for deterministic verification and auditability.

---

## 4. Normalization Statistics

OMNet-V2 supports two normalization schemes computed over the dataset:

### 4.1 ImageNet Normalization (Default Benchmark)
- **Mean:** $\mu = [0.485, 0.456, 0.406]$
- **Standard Deviation:** $\sigma = [0.229, 0.224, 0.225]$

### 4.2 BreakHis-200X Empirical Dataset Statistics
- **Empirical Mean:** $\mu_{\text{BreakHis}} = [0.7042, 0.5361, 0.6908]$
- **Empirical Standard Deviation:** $\sigma_{\text{BreakHis}} = [0.1834, 0.2012, 0.1685]$
- *Note:* Empirical statistics reflect the dominant pink (Eosin) and purple (Hematoxylin) color distribution of H&E stained pathology tissue.

---

## 5. Data Augmentation & Preprocessing Pipeline

```
Raw PNG Image (700 × 460)
         │
         ▼
[Training Transformations]
  ├── RandomResizedCrop(224, scale=(0.8, 1.0))
  ├── RandomHorizontalFlip(p=0.5)
  ├── RandomVerticalFlip(p=0.5)
  ├── RandomRotation(degrees=20)
  ├── ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  ├── ToTensor()
  └── Normalize(mean, std)
         │
         ▼
Transformed Tensor (3 × 224 × 224)
```

### 5.1 Training Augmentations Specification
- **Geometric Invariance:**
  - `RandomResizedCrop(224, scale=(0.8, 1.0))`: Simulates varying field-of-view framing and slight tissue zoom differences.
  - `RandomHorizontalFlip(p=0.5)` & `RandomVerticalFlip(p=0.5)`: Exploits rotational and reflection symmetry in tissue section orientation.
  - `RandomRotation(degrees=20)`: Emulates arbitrary slide angle placement under microscope objectives.
- **Color Space Perturbation:**
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`: Robustly simulates inter-laboratory staining variations, incubation time differences, and scanner illumination changes.

### 5.2 Validation & Test Transformations
- Deterministic `Resize((224, 224))`
- `ToTensor()`
- ImageNet Normalization (no random perturbations applied).

---

## 6. DataLoader & Sampling Policy

```python
# Weighted Random Oversampling for Training Batches
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
```

| Parameter | Setting | Technical Rationale |
|---|---|---|
| `batch_size` | 16 | Prevents Out-Of-Memory (OOM) errors during ViT-B/16 self-attention computation |
| `sampler` | `WeightedRandomSampler` | Balances 69/31 class ratio dynamically during minibatch SGD |
| `num_workers` | 0 | Ensures deterministic execution and avoids OS multiprocessing deadlocks |
| `pin_memory` | `True` | Accelerates Host-to-Device tensor transfers via pinned memory |
| `drop_last` | `False` | Ensures all evaluation samples are evaluated without truncation |
