# Architecture Specification — OMNet-V3 (Local Magnification-Aware Multi-Task Model)

> Comprehensive, end-to-end architectural specification for OMNet-V3 as implemented in `OMNet_BreakHis_V3_local.ipynb`.

---

## 1. Architectural Overview

**OMNet-V3** is a dual-branch, multi-task deep neural network designed for breast cancer histopathology classification on the BreakHis dataset across all 4 magnification scales (40X, 100X, 200X, 400X).

The model combines:
1. **CNN Local Morphology Branch:** Pretrained EfficientNet-B0 capturing fine cellular and nuclear textural patterns.
2. **Vision Transformer (ViT) Global Context Branch:** Pretrained ViT-Tiny/16 capturing long-range spatial relationships and tissue architectural organization.
3. **Magnification-Aware Adaptive Fusion (MAF) Module:** A scale-conditioned gating mechanism that dynamically modulates the fusion weight $\alpha \in (0, 1)$ between CNN and ViT representations based on the image's magnification scale embedding ($e_m \in \mathbb{R}^{64}$).
4. **Multi-Task Classification Heads:** Simultaneous prediction of primary binary malignancy (2 classes) and detailed histological subtyping (8 classes).
5. **Hierarchical Consistency Loss:** A multi-task objective with an explicit Kullback-Leibler (KL) divergence constraint enforcing probabilistic alignment between subtype predictions and binary predictions.

---

## 2. Input Pipeline & Tensor Specifications

| Stage | Input Dimensions | Operations / Output |
|---|---|---|
| **Input Image** | `(B, 3, 224, 224)` | RGB Histopathology tile (normalized ImageNet mean/std) |
| **Magnification Index** | `(B,)` integer $\in \{0, 1, 2, 3\}$ | $0 \rightarrow 40\text{X}$, $1 \rightarrow 100\text{X}$, $2 \rightarrow 200\text{X}$, $3 \rightarrow 400\text{X}$ |
| **CNN Preprocessing** | `(B, 3, 224, 224)` | Fed into EfficientNet-B0 feature extractor |
| **ViT Preprocessing** | `(B, 3, 224, 224)` | Fed into ViT-Tiny patch embedding ($16 \times 16$ patches $\rightarrow 196$ tokens) |

---

## 3. Dual-Branch Feature Extractors

### 3.1 Branch 1: CNN Local Feature Extractor (EfficientNet-B0)
- **Model Source:** `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)`
- **Forward Path:**
  ```python
  x_feat = self.cnn_backbone.forward_features(x)  # (B, 1280, 7, 7)
  x_pool = self.cnn_backbone.global_pool(x_feat)   # (B, 1280)
  f_cnn  = self.cnn_proj(x_pool)                   # (B, 256)
  ```
- **Projection Head (`cnn_proj`):**
  - `nn.Linear(1280, 256, bias=True)`
  - `nn.LayerNorm(256)`
  - Output representation: $f_{\text{cnn}} \in \mathbb{R}^{B \times 256}$

### 3.2 Branch 2: ViT Global Context Extractor (ViT-Tiny/16)
- **Model Source:** `timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)`
- **Forward Path:**
  ```python
  x_tokens = self.vit_backbone.forward_features(x) # (B, 197, 192) including [CLS]
  cls_token = x_tokens[:, 0, :]                    # (B, 192)
  f_vit     = self.vit_proj(cls_token)             # (B, 256)
  ```
- **Projection Head (`vit_proj`):**
  - `nn.Linear(192, 256, bias=True)`
  - `nn.LayerNorm(256)`
  - Output representation: $f_{\text{vit}} \in \mathbb{R}^{B \times 256}$

---

## 4. Magnification-Aware Adaptive Fusion (MAF) Module

The core innovation of OMNet-V3 is the **Magnification-Aware Adaptive Fusion (MAF)** layer, which dynamically conditions the feature fusion on the scale of observation.

```
f_cnn (B, 256) ───┐
f_vit (B, 256) ───┼──► Concat (B, 576) ──► Gate MLP ──► alpha (B, 1) in (0, 1)
e_m   (B, 64)  ───┘                                        │
                                                            ▼
fused = alpha * f_cnn + (1 - alpha) * f_vit  ◄──────────────┘
  │
  ├──► Concat([fused, e_m]) (B, 320) ──► Residual MLP ──► residual (B, 256)
  │                                                              │
  └──────────────────────── (+) ◄────────────────────────────────┘
                             │
                             ▼
                        f_out (B, 256)
```

### Mathematical Formulation
1. **Magnification Embedding:**
   $$e_m = \text{Embedding}(m) \in \mathbb{R}^{64}, \quad m \in \{0, 1, 2, 3\}$$

2. **Adaptive Gating Parameter $\alpha$:**
   $$\alpha = \sigma\left(\mathbf{W}_g [f_{\text{cnn}} \,\|\, f_{\text{vit}} \,\|\, e_m] + b_g\right) \in (0, 1)$$
   where $[ \cdot \,\|\, \cdot ]$ denotes concatenation ($256 + 256 + 64 = 576\text{-d}$), $\mathbf{W}_g \in \mathbb{R}^{1 \times 576}$, and $\sigma$ is the sigmoid activation.

3. **Convex Combination:**
   $$f_{\text{fused}} = \alpha \odot f_{\text{cnn}} + (1 - \alpha) \odot f_{\text{vit}} \in \mathbb{R}^{256}$$

4. **Scale-Conditioned Residual Refinement:**
   $$f_{\text{res}} = \mathbf{W}_2 \cdot \left(\text{Dropout}_{0.3}\left(\text{GELU}\left(\mathbf{W}_1 [f_{\text{fused}} \,\|\, e_m] + b_1\right)\right)\right) + b_2$$
   where $\mathbf{W}_1 \in \mathbb{R}^{256 \times 320}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 256}$.

5. **Final Fused Latent Representation:**
   $$f_{\text{out}} = f_{\text{fused}} + f_{\text{res}} \in \mathbb{R}^{256}$$

---

## 5. Multi-Task Output Classification Heads

Both classification heads branch directly from the unified representation $f_{\text{out}} \in \mathbb{R}^{256}$:

### 5.1 Binary Malignancy Head
- **Architecture:** `nn.Linear(256, 2)`
- **Logits:** $z_{\text{binary}} \in \mathbb{R}^{B \times 2}$
- **Classes:** Index 0 = Benign, Index 1 = Malignant

### 5.2 Histological Subtype Head
- **Architecture:** `nn.Linear(256, 8)`
- **Logits:** $z_{\text{subtype}} \in \mathbb{R}^{B \times 8}$
- **Subtype Mapping:**
  - Index 0: `A` (Adenosis - Benign)
  - Index 1: `F` (Fibroadenoma - Benign)
  - Index 2: `PT` (Phyllodes Tumor - Benign)
  - Index 3: `TA` (Tubular Adenoma - Benign)
  - Index 4: `DC` (Ductal Carcinoma - Malignant)
  - Index 5: `LC` (Lobular Carcinoma - Malignant)
  - Index 6: `MC` (Mucinous Carcinoma - Malignant)
  - Index 7: `PC` (Papillary Carcinoma - Malignant)

---

## 6. Hierarchical Multi-Task Loss Formulation

The network is trained with a composite hierarchical loss $L_{\text{total}}$:

$$L_{\text{total}} = w_{\text{bin}} L_{\text{binary}} + w_{\text{sub}} L_{\text{subtype}} + w_{\text{cons}} L_{\text{consistency}}$$

where the weights are:
- $w_{\text{bin}} = 0.3$
- $w_{\text{sub}} = 0.6$
- $w_{\text{cons}} = 0.1$

### 6.1 Weighted Binary Cross-Entropy
$$L_{\text{binary}} = -\frac{1}{B} \sum_{i=1}^B w_{y_i^{\text{bin}}} \log \left(\frac{\exp(z_{\text{bin}, i}[y_i^{\text{bin}}])}{\sum_{c=0}^1 \exp(z_{\text{bin}, i}[c])}\right)$$

### 6.2 Weighted Subtype Cross-Entropy
$$L_{\text{subtype}} = -\frac{1}{B} \sum_{i=1}^B w_{y_i^{\text{sub}}} \log \left(\frac{\exp(z_{\text{sub}, i}[y_i^{\text{sub}}])}{\sum_{k=0}^7 \exp(z_{\text{sub}, i}[k])}\right)$$

*Class weights are calculated per-fold inversely proportional to class frequency:*
$$w_c = \frac{N_{\text{total}}}{C \cdot \max(N_c, 1)}$$

### 6.3 Hierarchical Probability Consistency Constraint
To ensure that fine-grained subtype probabilities agree with coarse binary predictions, the 8 subtype softmax probabilities are aggregated into two super-classes:
$$\hat{p}_{\text{benign}}^{\text{sub}} = \sum_{k=0}^3 \text{Softmax}(z_{\text{sub}})_k, \quad \hat{p}_{\text{malignant}}^{\text{sub}} = \sum_{k=4}^7 \text{Softmax}(z_{\text{sub}})_k$$

The binary prediction acts as the detached reference distribution $q = \text{Softmax}(z_{\text{bin}})_{\text{detached}}$:
$$L_{\text{consistency}} = D_{\text{KL}}\left(\hat{p}^{\text{sub}} \,\|\, q\right) = \sum_{c \in \{0, 1\}} q_c \log\left(\frac{q_c}{\hat{p}_c^{\text{sub}} + \epsilon}\right)$$

---

## 7. Complete Forward Pass Data Flow Summary

```
Input Image x (B, 3, 224, 224) ──┬─► EfficientNet-B0 ─► (B, 1280) ─► cnn_proj ─► f_cnn (B, 256)
                                 │
                                 └─► ViT-Tiny/16      ─► (B, 192)  ─► vit_proj ─► f_vit (B, 256)
                                                                                       │
Magnification Index m (B,) ────────► nn.Embedding(4, 64) ────────────────────────► e_m (B, 64)
                                                                                       │
                                                                                       ▼
                                                  MAF Module: [f_cnn, f_vit, e_m]
                                                  ├─► Adaptive Gate alpha in (0, 1)
                                                  ├─► Convex Fusion + Residual MLP
                                                  ▼
                                                  Fused Embedding f_out (B, 256)
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
                 Binary Head (Linear)                            Subtype Head (Linear)
                         ▼                                                 ▼
               binary_logits (B, 2)                              subtype_logits (B, 8)
                         │                                                 │
                         └─────────────────► KL Consistency ◄──────────────┘
```

---

## 8. Hyperparameter & Optimization Summary

| Hyperparameter | Value | Description |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay regularization |
| **Base Learning Rate** | `1e-5` | Learning rate for pretrained backbone feature extractors |
| **Head Learning Rate** | `1e-4` | Learning rate for fusion module and classification heads |
| **Weight Decay** | `1e-4` | L2 weight regularization |
| **Batch Size** | `16` | Optimized for GPU memory |
| **Epochs** | `30` | Max epochs per fold |
| **Warmup Epochs** | `5` | Linear warmup period |
| **Early Stopping Patience** | `8` | Monitored on validation subtype Macro-F1 |
| **Gradient Clipping Norm** | `1.0` | Max norm for gradient stabilization |
| **Precision Mode** | `FP32` | Full 32-bit floating point precision (ROCm 7.2 / AMD RX 9060 XT stable) |
| **Stain Augmentation Prob** | `0.5` | Macenko optical density stain perturbation probability |
| **Random Seed** | `42` | Global deterministic seed |
