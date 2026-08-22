# Data Augmentation & Stain Normalization Analysis

## 1. Geometric Perturbations
Both OMNet-V2 and OMNet-V3 employ stochastic geometric transformations to enforce rotational and scale invariance:
- `RandomResizedCrop(224, scale=(0.8, 1.0))`: Emulates slight tissue framing and objective zoom variance.
- `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.5)`: Exploits the rotational symmetry of biopsy section placement.
- `RandomRotation(degrees=20)`: Emulates arbitrary slide angle placement under microscope stages.

## 2. Stain Normalization & Color Space Perturbation
1. **OMNet-V2:** Utilizes `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` to simulate inter-scanner illumination and staining intensity differences.
2. **OMNet-V3:** Features **Macenko Optical Density (OD) Stain Perturbation** ($p=0.5$):
   - Decomposes RGB images into Hematoxylin and Eosin stain vectors via Singular Value Decomposition (SVD) in Optical Density space: $	ext{OD} = -\log_{10}(I/255 + \epsilon)$.
   - Randomly scales concentration channels by $	ext{Uniform}(0.85, 1.15)$ with bias $	ext{Uniform}(-0.05, 0.05)$ before reconstructing to RGB.
