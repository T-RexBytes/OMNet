# Dataset Specification — OMNet-V3 (BreakHis Multi-Scale Histopathology)

> Complete data-related documentation for OMNet-V3 as implemented in `OMNet_BreakHis_V3_local.ipynb` and stored in `result_OMNet/local/dataset_metadata.csv`.

---

## 1. Dataset Overview: BreaKHis

| Property | Value |
|---|---|
| **Dataset Title** | Breast Cancer Histopathological Image Database (BreakHis) |
| **Kaggle Source / Identifier** | `saikatd1998/dataset` / `ambarish/breakhis` |
| **Tissue Modality** | Formalin-fixed paraffin-embedded (FFPE) Hematoxylin and Eosin (H&E) stained breast tissue |
| **Image Resolution** | $700 \times 460$ pixels, 24-bit RGB (PNG format) |
| **Processed Input Resolution** | $224 \times 224$ pixels (after resize & crop) |
| **Total Validated Images** | **7,909 images** |
| **Total Unique Patients** | **82 patients** |
| **Magnification Factors** | $40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$ |

---

## 2. Class Hierarchy & Taxonomy

BreakHis images are organized hierarchically into **2 binary super-classes** and **8 fine-grained histological subtypes**:

```
BreaKHis Dataset (7,909 images / 82 patients)
├── Benign (2,480 images, 31.36% / 24 patients)
│   ├── Adenosis (A)            : 444 images (5.61%),   4 patients
│   ├── Fibroadenoma (F)        : 1,014 images (12.82%), 10 patients
│   ├── Phyllodes Tumor (PT)    : 453 images (5.73%),   3 patients
│   └── Tubular Adenoma (TA)    : 569 images (7.19%),   7 patients
│
└── Malignant (5,429 images, 68.64% / 58 patients)
    ├── Ductal Carcinoma (DC)   : 3,451 images (43.63%), 38 patients
    ├── Lobular Carcinoma (LC)  : 626 images (7.92%),   5 patients
    ├── Mucinous Carcinoma (MC) : 792 images (10.01%),  9 patients
    └── Papillary Carcinoma (PC): 560 images (7.08%),   6 patients
```

### Numerical Index Mapping

| Subtype Code | Subtype Name | Binary Super-Class | Binary Label | Subtype Index | Total Images | Patient Count |
|---|---|---|---|---|---|---|
| **A** | Adenosis | Benign | `0` | `0` | 444 | 4 |
| **F** | Fibroadenoma | Benign | `0` | `1` | 1,014 | 10 |
| **PT** | Phyllodes Tumor | Benign | `0` | `2` | 453 | 3 |
| **TA** | Tubular Adenoma | Benign | `0` | `3` | 569 | 7 |
| **DC** | Ductal Carcinoma | Malignant | `1` | `4` | 3,451 | 38 |
| **LC** | Lobular Carcinoma | Malignant | `1` | `5` | 626 | 5 |
| **MC** | Mucinous Carcinoma | Malignant | `1` | `6` | 792 | 9 |
| **PC** | Papillary Carcinoma | Malignant | `1` | `7` | 560 | 6 |

---

## 3. Magnification Scale Distribution

Images were captured under 4 optical magnification levels with balanced representation across scales:

| Magnification | Index | Total Images | Percentage |
|---|---|---|---|
| **40X** | `0` | 1,995 | 25.22% |
| **100X** | `1` | 2,081 | 26.31% |
| **200X** | `2` | 2,013 | 25.45% |
| **400X** | `3` | 1,820 | 23.01% |
| **Total** | — | **7,909** | **100.0%** |

---

## 4. Filename Parsing Logic & Patient Identifier Derivation

BreakHis filenames follow the standardized clinical nomenclature:
$$\text{SOB\_}\langle\text{B}|\text{M}\rangle\_\langle\text{Subtype}\rangle\text{-}\langle\text{Year}\rangle\text{-}\langle\text{CaseID}\rangle\text{-}\langle\text{Magnification}\rangle\text{-}\langle\text{Sequence}\rangle\text{.png}$$

**Example:** `SOB_M_DC-14-2980-400-001.png`
- `SOB`: Slide Originating from Biopsy
- `M`: Malignant class indicator (`B` for benign)
- `DC`: Ductal Carcinoma subtype abbreviation
- `14-2980`: Year (2014) and unique Case/Patient ID
- `400`: 400X magnification
- `001`: Image sequence slice number

**Derived Patient Identifier:**
```python
case_id = '-'.join(parts[1:-2])  # e.g., '14-2980'
patient_id = f"{raw_subtype}_{case_id}"  # e.g., 'DC_14-2980'
```

---

## 5. Patient-Disjoint 5-Fold Stratified Group Cross-Validation Protocol

To guarantee **zero patient data leakage**, the dataset is partitioned using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`:
- **Group Key:** `patient_id` (guarantees all images from any single patient appear strictly in train OR test, never both).
- **Stratification Target:** `subtype_index` (balances the 8 subtype proportions across folds).

### 5.1 Folds Summary Breakdown

| Fold | Train Images (Patients) | Outer Val Images (Patients) | Test Images (Patients) | Total Images | Leakage Check |
|---|---|---|---|---|---|
| **Fold 0** | 5,829 (60 patients) | 524 (6 patients) | 1,556 (16 patients) | 7,909 | **0 Overlap (PASS)** |
| **Fold 1** | 5,748 (60 patients) | 632 (6 patients) | 1,529 (16 patients) | 7,909 | **0 Overlap (PASS)** |
| **Fold 2** | 5,732 (60 patients) | 588 (6 patients) | 1,589 (16 patients) | 7,909 | **0 Overlap (PASS)** |
| **Fold 3** | 5,618 (58 patients) | 589 (6 patients) | 1,702 (18 patients) | 7,909 | **0 Overlap (PASS)** |
| **Fold 4** | 5,774 (60 patients) | 602 (6 patients) | 1,533 (16 patients) | 7,909 | **0 Overlap (PASS)** |

### 5.2 Discrete Patient Grouping Constraints (Rare Subtypes)
Because certain subtypes have very few unique patients (e.g., Phyllodes Tumor has only 3 patients, Adenosis has only 4), it is mathematically impossible to have every subtype represented in every test fold under strict 5-fold patient partitioning:
- **Fold 0 Test:** PT missing (0 patients, 0 images)
- **Fold 2 Test:** A missing (0 patients, 0 images)
- **Fold 4 Test:** PT missing (0 patients, 0 images)

*Macro-F1 is computed across all classes present in each fold using `zero_division=np.nan` to prevent artificial metric penalization from structural patient scarcity.*

---

## 6. Data Augmentation & Stain Normalization Pipeline

### 6.1 Training Augmentation Pipeline
To simulate realistic clinical and laboratory slide preparation variability:
1. **Geometric Transformations:**
   - Resize to $256 \times 256$
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomVerticalFlip(p=0.5)`
   - `RandomRotation(degrees=20)`
2. **Macenko Optical Density (OD) Stain Perturbation ($p = 0.5$):**
   - RGB to Optical Density: $\text{OD} = -\log_{10}\left(\frac{I}{255} + \epsilon\right)$
   - Singular Value Decomposition (SVD) on non-background pixels to estimate Hematoxylin and Eosin stain vectors $v_{\text{HE}} \in \mathbb{R}^{2 \times 3}$.
   - Projection into stain concentration space: $S = \text{OD} \cdot v_{\text{HE}}^T$.
   - Multiplicative perturbation: $S_{\text{pert}} = S \cdot \text{Uniform}(0.85, 1.15) + \text{Uniform}(-0.05, 0.05)$.
   - Reconstruction back to RGB: $I_{\text{recon}} = 255 \cdot 10^{-S_{\text{pert}} \cdot v_{\text{HE}}}$.
3. **Color Space Normalization:**
   - ImageNet mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`

### 6.2 Validation and Test Pipeline
- Resize to $256 \times 256$
- Deterministic `CenterCrop(224)`
- ImageNet Normalization (no stochastic augmentations or stain perturbations)

---

## 7. DataLoader Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | `16` | Optimized balance between gradient stability and GPU VRAM |
| `num_workers` | `0` | Avoids multiprocessing deadlocks in ROCm/PyTorch environments |
| `pin_memory` | `True` | Accelerates host-to-device memory transfers |
| `shuffle` (Train) | `True` | Ensures randomized batch gradient descent |
| `shuffle` (Val/Test) | `False` | Ensures deterministic, reproducible metric evaluation |

---

## 8. Stored Metadata Schema (`dataset_metadata.csv`)

| Column Name | Type | Description | Example |
|---|---|---|---|
| `file_path` | string | Full local filesystem path to the PNG image | `/home/.../SOB_M_DC-14-2980-400-001.png` |
| `filename` | string | Base image filename | `SOB_M_DC-14-2980-400-001.png` |
| `class_name` | string | Binary class string (`benign` or `malignant`) | `malignant` |
| `subtype` | string | Subtype abbreviation (`A, F, PT, TA, DC, LC, MC, PC`) | `DC` |
| `subtype_index` | integer | Categorical integer target $\in [0, 7]$ | `4` |
| `binary_label` | integer | Binary integer target $\in \{0, 1\}$ | `1` |
| `magnification` | integer | Optical magnification level ($40, 100, 200, 400$) | `400` |
| `magnification_index`| integer | Scale index $\in \{0, 1, 2, 3\}$ for embedding lookup | `3` |
| `patient_id` | string | Unique patient group identifier | `DC_14-2980` |
| `sequence` | integer | Image slice sequence number | `1` |
