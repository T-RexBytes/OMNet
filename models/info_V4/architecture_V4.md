# Architecture V4 — OMNet Multi-Task Breast Histopathology Model

> Full end-to-end architectural specification, start to finish.

---

## 1. Overview

**OMNet V4** is a multi-task deep learning framework for **breast histopathology detection and 8-class subtype classification** on the BreakHis dataset.

It is structured as a **shared-backbone, dual-head** network combining:
1. An **ImageNet-pretrained CNN encoder** (DenseNet-201)
2. A **Lightweight Convolutional Block Attention Module (LCBAM)** on feature maps
3. A **Projection head** reducing to a shared latent space
4. A **Fourier-KAN + Weight-Tied Iterative Refinement block** for deep implicit representation refinement
5. Two output heads: **8-class subtype head** (primary) + **2-class binary detection head** (auxiliary)

---

## 2. Input Pipeline

| Property | Value |
|---|---|
| Input resolution | `224 x 224 x 3` (RGB) |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Train augmentations | RandomCrop(256->224), H-Flip, V-Flip, Rotation(+/-15 deg), ColorJitter |
| Eval transforms | CenterCrop(256->224), Normalize |

---

## 3. Backbone — DenseNet-201

```
Input: (B, 3, 224, 224)
   └─► DenseNet-201 Feature Extractor (backbone_features)
       ├── Dense Block 1  → Transition 1
       ├── Dense Block 2  → Transition 2
       ├── Dense Block 3  → Transition 3
       └── Dense Block 4  (final)
Output: (B, 1920, H', W')   [H'≈7 for 224 input]
```

- Loaded with **ImageNet1K_V1** pretrained weights (`pretrained=True`)
- Fallback: `EfficientNet-B0` (output `(B, 1280, H', W')`)
- The classifier head is **discarded**; only `model.features` is retained

---

## 4. LCBAM — Lightweight Convolutional Block Attention Module

Applied **before** global pooling, directly on the 4D feature map `(B, C, H, W)`.

### 4a. Channel Attention

```
AvgPool(H,W -> 1,1) → flatten → Linear(C, C//r) → ReLU → Linear(C//r, C)
MaxPool(H,W -> 1,1) → flatten → (same shared MLP weights)
channel_att = sigmoid(avg_out + max_out)          # (B, C, 1, 1)
x_ca = x * channel_att
```

- Reduction ratio `r = 16`
- Min reduced channels: `max(C//16, 16)`

### 4b. Spatial Attention

```
avg over channels → (B, 1, H, W)
max over channels → (B, 1, H, W)
concat            → (B, 2, H, W)
DepthwiseConv2d(2→1, k=7, pad=3) → BatchNorm2d → sigmoid
spatial_att = (B, 1, H, W)
x_out = x_ca * spatial_att
```

**Returns:** `(x_out, spatial_att)` — spatial attention map is cached as `model.last_spatial_attention` for explainability (Figure J).

---

## 5. Global Pooling & Projection Head

```
AdaptiveAvgPool2d(1,1) → flatten → (B, 1920)
   └─► Projector:
       Linear(1920, 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
Output: z_proj  (B, 256)
```

---

## 6. Fourier-KAN Layer

A novel activation layer replacing standard MLPs, inspired by Kolmogorov-Arnold Networks with Fourier basis functions.

```
FourierKANLayer(in_features=256, out_features=256, grid_size=8)
```

**Forward pass:**
```
base_out = Linear(x)                                  # (B, 256)

k = [1, 2, ..., grid_size]                            # frequency indices
x_exp = x.unsqueeze(-1) * k                           # (B, 256, 8)

sin_basis = sin(x_exp)                                # (B, 256, 8)
cos_basis = cos(x_exp)                                # (B, 256, 8)

fourier_out = einsum(cos_basis, cos_coeffs)
            + einsum(sin_basis, sin_coeffs)            # (B, 256)

output = base_out + fourier_out
```

**Parameters:**
- `fourier_coeffs`: shape `(2, 256, 256, 8)` — learnable sin + cos coefficients
- `base_linear`: standard linear residual connection
- Init: `N(0, 1/sqrt(in * grid_size))`

---

## 7. Weight-Tied Iterative Refinement Block

A **fixed-iteration unrolled loop** sharing weights across all `N=6` iterations — inspired by DEQ models, but deterministic.

```
WeightTiedRefinementBlock(dim=256, grid_size=8, n_iters=6, dropout=0.2)
```

**Single shared sub-network (weight-tied across all iterations):**
```
LayerNorm(256) → FourierKANLayer(256→256) → GELU → Dropout(0.2)
```

**Unrolled loop:**
```
z = z_proj
for _ in range(6):
    residual = GELU(FKAN(LayerNorm(z)))
    z = z + Dropout(residual)          # residual connection each iteration
z_shared = z                           # (B, 256)
```

---

## 8. Output Heads

### 8a. Subtype Head (Primary — 8 classes)
```
Linear(256, 8)  →  logits_subtype (B, 8)
```
Classes: `adenosis(0), fibroadenoma(1), phyllodes_tumor(2), tubular_adenoma(3), ductal_carcinoma(4), lobular_carcinoma(5), mucinous_carcinoma(6), papillary_carcinoma(7)`

### 8b. Detection Head (Auxiliary — 2 classes)
```
Linear(256, 2)  →  logits_detection (B, 2)
```
Classes: `benign (0), malignant (1)`

Both heads share the **same** `z_shared` vector — fully shared representation.

---

## 9. Full Forward Pass (Data Flow)

```
Input (B, 3, 224, 224)
    |
    v
[DenseNet-201 backbone]
    |  (B, 1920, 7, 7)
    v
[LCBAM: Channel Attn + Spatial Attn]  →  cache spatial_att for explainability
    |  (B, 1920, 7, 7)
    v
[AdaptiveAvgPool2d(1,1)] → flatten
    |  (B, 1920)
    v
[Projector: Linear → BN → ReLU → Dropout]
    |  z_proj (B, 256)
    v
[WeightTiedRefinementBlock x 6 iters]
    |  z_shared (B, 256)
    |___________________________
    v                           v
[Linear(256, 8)]        [Linear(256, 2)]
subtype_logits (B,8)    detection_logits (B,2)
```

---

## 10. Loss Functions

### Multi-Task Combined Loss

```
L_total = 1.0 * L_subtype + 0.3 * L_detection
```

| Component | Weight | Function |
|---|---|---|
| Subtype loss | `1.0` | Weighted CrossEntropy (inverse frequency) |
| Detection loss | `0.3` | Weighted CrossEntropy (inverse frequency) |

Class weights: `w_c = total / (n_classes * count_c)`, normalized by mean.
Optional: Focal Loss with `gamma=2.0` (configurable via `SUBTYPE_LOSS="focal"`).

---

## 11. Optimizer & Scheduler

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) |
| Min LR | `1e-6` |
| Grad clip norm | `1.0` |
| Max epochs | `50` |
| Early stopping patience | `10` |
| Monitor metric | `val_macro_f1` |

---

## 12. Precision Policy

| Setting | Value |
|---|---|
| Mixed precision (AMP) | `False` — FP32 full training |
| Reason | Prevents GPUVM faults on RDNA 4 (AMD RX 9060 XT) under ROCm 7.2 |
| Deterministic seeds | `True` |
| Global seed | `42` |

---

## 13. Baseline & Ablation Architecture Variants

| Model ID | Backbone | FKAN | LCBAM | Refinement Iters | Detection Head |
|---|---|---|---|---|---|
| B1_FromScratch | DenseNet-201, random init | No | No | 0 | No |
| B2_Pretrained | DenseNet-201, ImageNet | No | No | 0 | No |
| B3_LiteratureRepro | DenseNet-201, random init | Yes | Yes | 6 | Yes |
| A0_Pretrained_Floor | = B2 | No | No | 0 | No |
| A1_FKAN_Only | DenseNet-201, ImageNet | Yes | No | 1 | No |
| A2_FKAN_LCBAM | DenseNet-201, ImageNet | Yes | Yes | 1 | No |
| A3_Refinement_SubtypeOnly | DenseNet-201, ImageNet | Yes | Yes | 6 | No |
| **A4_ProposedModel** | **DenseNet-201, ImageNet** | **Yes** | **Yes** | **6** | **Yes** |

---

## 14. Parameter Count (Approximate)

| Module | Parameters (approx.) |
|---|---|
| DenseNet-201 backbone | ~18M |
| LCBAM | ~0.5M |
| Projector | ~0.5M |
| FourierKAN (weight-tied, reused x6) | ~0.5M |
| Output heads (x2) | ~4K |
| **Total** | **~19.5M** |

---

## 15. Hardware Target

- **Primary:** AMD Radeon RX 9060 XT 16GB (RDNA 4, gfx1200) under ROCm 7.2 / Linux
- **Batch size:** 16 (fits 16GB VRAM with FP32 dual-head training)
- **DataLoader workers:** 0 (conservative; avoids multiprocessing crashes in CV loops)
- **Pin memory:** True (when CUDA/ROCm available)
