# Dataset V4 — OMNet BreakHis Data Documentation

> All dataset-related information for the OMNet V4 BreakHis-Only pipeline.

---

## 1. Primary Dataset: BreakHis

| Property | Value |
|---|---|
| Dataset name | Breast Cancer Histopathological Image Database (BreakHis) |
| Kaggle slug | `ambarish/breakhis` |
| Acquisition | `kagglehub.dataset_download()` or `kaggle datasets download` CLI |
| Format | PNG images, folder-structured |
| Total images | ~7,909 (varies by version; all magnifications) |
| Total patients | 82 unique patients |

---

## 2. Image Characteristics

| Property | Value |
|---|---|
| Original resolution | 700 x 460 pixels (RGB) |
| Training input size | 224 x 224 (after resize + crop) |
| File format | PNG |
| Color space | RGB |
| Magnification factors | 40X, 100X, 200X, 400X |

---

## 3. Class Structure

### Binary Labels

| Class | Label | Description |
|---|---|---|
| Benign | 0 | Non-cancerous tissue |
| Malignant | 1 | Cancerous tissue |

### 8-Class Subtype Labels

| Subtype | Label | Category |
|---|---|---|
| Adenosis | 0 | Benign |
| Fibroadenoma | 1 | Benign |
| Phyllodes Tumor | 2 | Benign |
| Tubular Adenoma | 3 | Benign |
| Ductal Carcinoma | 4 | Malignant |
| Lobular Carcinoma | 5 | Malignant |
| Mucinous Carcinoma | 6 | Malignant |
| Papillary Carcinoma | 7 | Malignant |

---

## 4. Filename Convention & Parsing

BreakHis filenames follow the standard:
```
SOB_<B|M>_<subtype>-<patient_id>-<magnification>-<sequence>.png
```

**Example:** `SOB_M_DC-14-2980-400-001.png`
- `M` = Malignant
- `DC` = Ductal Carcinoma
- `14-2980` = Patient identifier
- `400` = Magnification
- `001` = Sequence number

Patient ID is constructed as: `SOB_<B|M>_<subtype>-<patient_num>`

**Abbreviation map used in parsing:**

| Abbreviation | Full Subtype |
|---|---|
| `a` | adenosis |
| `f` | fibroadenoma |
| `pt` | phyllodes_tumor |
| `ta` | tubular_adenoma |
| `dc` | ductal_carcinoma |
| `lc` | lobular_carcinoma |
| `mc` | mucinous_carcinoma |
| `pc` | papillary_carcinoma |

---

## 5. Quality Control Pipeline

All images pass through the following QC gates before being added to the manifest:

1. **Integrity verification:** `Image.open(fp).verify()` — corrupted files are excluded and logged to `corrupt_files.csv`
2. **MD5 hash deduplication:** Exact byte-level duplicates are detected; first occurrence is kept, duplicates logged to `dedup_report.json`
3. **Label validation:** Images with unresolvable subtype labels (`subtype_label == -1`) are excluded
4. **Manifest construction:** All valid images are saved to `dataset_manifest.csv` with full metadata

---

## 6. Patient-Grouped Splitting Strategy

> **Zero patient overlap is enforced across ALL splits.**

### Split Protocol

```
All 82 patients
    |
    ├── 15% → Frozen Test Set  (patient-level stratified)
    │         StratifiedGroupKFold(n_splits=7), first fold used
    │
    └── 85% → Development Pool
              |
              └── 5-Fold Cross-Validation
                  StratifiedGroupKFold(n_splits=5)
                  Each fold: ~68% train / ~17% val (of total)
```

**Stratification key:** `subtype_label` (8-class)
**Grouping key:** `patient_id` (ensures same patient never splits across folds)

### Split Ratios (approximate image-level)

| Split | Ratio | Usage |
|---|---|---|
| Development Pool | ~85% | 5-fold CV training/validation |
| Frozen Test Set | ~15% | Final evaluation (evaluated exactly once) |

---

## 7. Leakage Verification Gate

After splitting, the following automated assertions are run (must all PASS before training proceeds):

| Check | Description | Required Result |
|---|---|---|
| Patient overlap: dev vs test | Patients appearing in both dev pool and test | == 0 |
| CV fold patient overlap | Patients appearing in both train and val of any CV fold | == 0 |
| Hash crossing: dev vs test | Exact image duplicates crossing the dev/test boundary | == 0 |
| **Overall verdict** | Combined pass/fail | **PASS** |

Results saved to: `leakage_report.json`
If any check fails → `AssertionError` is raised and training is halted.

---

## 8. Data Augmentation

### Training Split Only

| Augmentation | Parameters |
|---|---|
| Resize | 256 x 256 |
| RandomCrop | 224 x 224 |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.5 |
| RandomRotation | degrees=15 |
| ColorJitter | brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0 |
| ToTensor + Normalize | ImageNet mean/std |

### Validation / Test Split

| Step | Parameters |
|---|---|
| Resize | 256 x 256 |
| CenterCrop | 224 x 224 |
| ToTensor + Normalize | ImageNet mean/std |

---

## 9. Class Imbalance Handling

Inverse frequency class weighting is applied per-fold:
```
w_c = total_samples / (n_classes * count_c)
weights normalized by mean(weights)
```

Applied independently to:
- Subtype criterion (8 classes)
- Detection criterion (2 classes)

Per-fold weights are recomputed from the training fold labels each iteration.

---

## 10. DataLoader Configuration

| Parameter | Value |
|---|---|
| Batch size | 16 |
| Shuffle (train) | True |
| Shuffle (val/test) | False |
| num_workers | 0 (conservative; avoids multiprocessing crashes) |
| pin_memory | True (if GPU available) |
| persistent_workers | False |
| drop_last | False |

---

## 11. Dataset Manifest Fields

The `dataset_manifest.csv` and `split_manifest.csv` contain the following columns:

| Column | Type | Description |
|---|---|---|
| `file_path` | str | Absolute path to image file |
| `filename` | str | Bare filename |
| `patient_id` | str | Derived patient identifier |
| `binary_class` | str | "benign" or "malignant" |
| `binary_label` | int | 0 or 1 |
| `subtype_name` | str | Full subtype name string |
| `subtype_label` | int | Integer class index (0-7) |
| `magnification` | int | 40, 100, 200, or 400 |
| `sequence` | int | Image sequence number |
| `md5_hash` | str | MD5 hash for deduplication |
| `split` | str | "dev_pool" or "test" |
| `cv_fold` | int | CV fold index (0-4), -1 for test |

---

## 12. Output Artifacts (Data-Related)

| File | Location | Description |
|---|---|---|
| `dataset_manifest.csv` | `runs/<run_id>/` | All valid images with metadata |
| `split_manifest.csv` | `runs/<run_id>/` | Manifest with split + cv_fold columns |
| `class_distribution.csv` | `runs/<run_id>/` | Per-split subtype class counts |
| `corrupt_files.csv` | `runs/<run_id>/` | Excluded corrupt image records |
| `dedup_report.json` | `runs/<run_id>/` | MD5 duplicate cluster report |
| `leakage_report.json` | `runs/<run_id>/` | Leakage verification results |
| `proposed_model_test_predictions.csv` | `runs/<run_id>/predictions/` | Per-image test predictions + probabilities |
| `low_confidence_review.csv` | `runs/<run_id>/predictions/` | Images with max confidence < 50% |
