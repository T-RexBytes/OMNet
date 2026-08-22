# Dataset Specification — OMNet-V3

> Comprehensive data pipeline and dataset documentation for OMNet-V3 on the **BreakHis (Breast Cancer Histopathological Image Database)**.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Dataset Title** | Breast Cancer Histopathological Image Database (BreakHis) |
| **Kaggle Source / Identifier** | `trexbytes/breakhislink` |
| **Tissue Modality** | Formalin-fixed paraffin-embedded (FFPE) Hematoxylin and Eosin (H&E) stained breast surgical biopsy specimens |
| **Image Resolution (Raw)** | $700 \times 460$ pixels, 24-bit RGB (PNG format) |
| **Processed Input Resolution** | $224 \times 224$ pixels (after resize & crop) |
| **Total Validated Images** | **7,909 images** |
| **Total Unique Patients** | **82 unique clinical patients** |
| **Magnification Levels** | $40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$ |

---

## 2. Class Hierarchy & Distribution

BreakHis images are organized hierarchically into **2 binary super-classes** and **8 fine-grained histological subtypes**:

```
BreakHis Cohort (7,909 images across 82 patients)
│
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

### Numerical Target Mapping Table

| Subtype Code | Full Histological Subtype | Super-Class | Binary Label | Subtype Index | Total Images | Patient Count |
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

## 3. Optical Magnification Scale Breakdown

Images were captured under four optical objective magnifications, representing different diagnostic scopes:

| Magnification | Index | Total Images | Percentage | Typical Clinical Scope |
|---|---|---|---|---|
| **40X** | `0` | 1,995 | 25.22% | Overview of glandular architecture, stromal margin demarcation, lobular arrangement |
| **100X** | `1` | 2,081 | 26.31% | Inter-ductal spacing, periductal fibrosis, epithelial-stromal boundary |
| **200X** | `2` | 2,013 | 25.45% | Cellular density, tubular lumen formation, nuclear arrangement |
| **400X** | `3` | 1,820 | 23.01% | Nuclear atypia, hyperchromasia, mitotic figures, chromatin distribution |
| **Total** | — | **7,909** | **100.0%** | Comprehensive multi-scale representation |

---

## 4. Standardized Filename Parsing & Patient ID Extraction

BreakHis image filenames adhere to the clinical convention:
$$\text{SOB\_}\langle\text{B}|\text{M}\rangle\_\langle\text{Subtype}\rangle\text{-}\langle\text{Year}\rangle\text{-}\langle\text{CaseID}\rangle\text{-}\langle\text{Magnification}\rangle\text{-}\langle\text{Sequence}\rangle\text{.png}$$

**Example:** `SOB_M_DC-14-2980-400-001.png`
- `SOB`: Slide Originating from Biopsy
- `M`: Malignant class indicator (`B` for benign)
- `DC`: Ductal Carcinoma subtype abbreviation
- `14-2980`: Year (2014) and unique Case/Patient ID
- `400`: 400X magnification
- `001`: Image sequence slice number

**Python Extraction Logic:**
```python
fname_no_ext = filename.replace('.png', '')
parts = fname_no_ext.split('-')
prefix_parts = parts[0].split('_')

raw_class = prefix_parts[1]       # 'B' or 'M'
raw_subtype = prefix_parts[2]     # e.g., 'DC'
mag = int(parts[-2])             # 40, 100, 200, or 400
case_id = '-'.join(parts[1:-2])  # e.g., '14-2980'
patient_id = f"{raw_subtype}_{case_id}" # Unique Patient Group
```

---

## 5. Patient-Disjoint 5-Fold Stratified Group Cross-Validation

To resolve the critical **data leakage problem** prevalent in prior BreakHis literature, OMNet-V3 enforces strict **patient-level partitioning** using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`:
- **Group Key:** `patient_id` (all patches from a given patient remain exclusively in train OR test).
- **Stratification Target:** `subtype_index` (equalizes the 8 subtype proportions across folds).

### 5.1 Folds Summary Breakdown
Across the 5 folds, patient assignments are partitioned to ensure that train, validation, and test subsets are strictly patient-disjoint:
- **Zero Patient Overlap:** Evaluated and verified across all folds (zero patient ID intersections between splits).
- **Rare Subtypes Handling:** Certain subtypes have extremely low patient counts (Phyllodes Tumor has 3 patients; Adenosis has 4 patients). Due to the nature of patient-level splits, a subtype with fewer patients than the number of splits will be absent from the test subset of some folds (e.g., PT is missing in Fold 0 and Fold 4 test sets; A is missing in Fold 2 test set).
- **Metric Robustness:** Evaluation metrics (especially Macro-F1) ignore absent classes in the test fold using `zero_division=np.nan` to prevent artificial penalty.

---

## 6. Preprocessing & Data Augmentation Pipelines

### 6.1 Training Augmentation Pipeline
1. **Geometric Augmentations:**
   - Resize to $256 \times 256$
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomVerticalFlip(p=0.5)`
   - `RandomRotation(degrees=20)`
2. **Macenko Optical Density (OD) Stain Perturbation ($p = 0.5$):**
   - Converts images to Optical Density space: $\text{OD} = -\log_{10}(I/255 + \epsilon)$.
   - Estimates H&E stain vectors via SVD on the OD channels.
   - Perturbs the stain concentrations with random scaling ($\text{Uniform}(0.85, 1.15)$) and bias ($\text{Uniform}(-0.05, 0.05)$) to simulate laboratory stain variability.
   - Reconstructs perturbed OD map back to RGB space.
3. **Color Space Normalization:**
   - ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

### 6.2 Validation & Test Pipeline
- Resize to $256 \times 256$
- Deterministic `CenterCrop(224)`
- ImageNet Normalization (no random cropping, flipping, or stain perturbation)

---

## 7. DataLoader Specifications

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | `16` | Optimized balance between gradient stability and GPU memory |
| `num_workers` | `0` | Avoids multiprocessing deadlocks in ROCm/PyTorch environments |
| `pin_memory` | `True` | Accelerates host-to-GPU memory copies |
| `shuffle` (Train) | `True` | Randomizes minibatch gradient updates |
| `shuffle` (Val/Test) | `False` | Guarantees deterministic, reproducible metric evaluation |
