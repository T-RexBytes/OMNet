# Dataset Analysis: BreakHis Multi-Scale & 200X Cohort

## 1. Dataset Modality & Source
- **Dataset:** Breast Cancer Histopathological Image Database (BreakHis)
- **Source Repositories:** `trexbytes/breakhislink` / `saikatd1998/dataset` / `ambarish/breakhis`
- **Specimens:** Formalin-fixed paraffin-embedded (FFPE) Hematoxylin and Eosin (H&E) stained surgical biopsy slides from 82 clinical patients.
- **Microscopy System:** Olympus BX-50 microscope with 3.3x relay lens, coupled to a Samsung SCC-131AN digital camera.
- **Raw Tile Dimensions:** 700 x 460 pixels, 24-bit RGB PNG format.

## 2. Cohort Comparison: Single-Scale (V2) vs. Full Multi-Scale (V3)

| Metric | OMNet-V2 Setting (BreakHis 200X) | OMNet-V3 Setting (BreakHis Multi-Scale) |
|---|---|---|
| **Optical Magnifications** | 200X only | 40X, 100X, 200X, 400X |
| **Total Images** | 2,013 images | 7,909 images |
| **Total Patients** | 82 unique patients | 82 unique patients |
| **Benign Images** | 623 (30.95%) / 24 patients | 2,480 (31.36%) / 24 patients |
| **Malignant Images** | 1,390 (69.05%) / 58 patients | 5,429 (68.64%) / 58 patients |
| **Partitioning Design** | 1,733 CV pool / 280 held-out test | 5-Fold Patient-Disjoint Stratified Group CV |
| **Patient Identity Leakage** | **0% (Cryptographically Verified)** | **0% (Cryptographically Verified)** |
