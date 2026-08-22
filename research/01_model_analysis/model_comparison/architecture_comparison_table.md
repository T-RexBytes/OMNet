# Architecture Specification — OMNet-V3

> Comprehensive, end-to-end architectural specification for **OMNet-V3: Magnification-Aware Multi-Task Fusion Framework for Breast Cancer Histopathology**.

---

## 1. Architectural Overview

**OMNet-V3** is a dual-branch, scale-aware deep learning framework designed for multi-scale histopathological biopsy analysis on the BreakHis dataset. The model simultaneously performs coarse-grained binary malignancy detection (benign vs. malignant) and fine-grained histological subtyping (8 distinct subtypes) across all four optical magnification levels ($40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$).

### Core Design Principles
1. **Dual-Branch Complementary Feature Encoders:**
   - **CNN Branch (EfficientNet-B0):** Extracts localized morphological primitives, cellular membrane boundaries, and nuclear chromatic textures with strong inductive bias for spatial locality.
   - **Vision Transformer Branch (ViT-Tiny/16):** Captures long-range spatial context, glandular architectural patterning, and stromal organization via global self-attention across non-overlapping patches.
2. **Magnification-Aware Adaptive Fusion (MAF) Module:**
   - Injects a learnable magnification embedding ($e_m \in \mathbb{R}^{64}$) into an adaptive gating mechanism to dynamically predict a scale-dependent mixing coefficient $\alpha \in (0, 1)$.
   - Performs convex feature blending followed by scale-conditioned non-linear residual refinement.
3. **Multi-Task Dual Heads:**
   - Binary Head (2-class) and Subtype Head (8-class) operating on the shared 256-dimensional fused embedding.
4. **Hierarchical Consistency Regularization:**
   - Explicit Kullback-Leibler (KL) divergence loss enforcing structural probability alignment between the sum of subtype probabilities and the binary prediction.

---

## 2. Input Pipeline & Tensor Dimensions

```
Input Image Tile: x ∈ R^(B × 3 × 224 × 224)
Magnification Scale: m ∈ {0, 1, 2, 3}  [0: 40X, 1: 100X, 2: 200X, 3: 400X]
```

| Component | Input Shape | Processing Stage | Output Shape |
|---|---|---|---|
| **Image Input** | `(B, 3, 224, 224)` | RGB Normalized (ImageNet Mean/Std) | `(B, 3, 224, 224)` |
| **Magnification Input** | `(B,)` integer $\in [0, 3]$ | Embedding Lookup (`nn.Embedding(4, 64)`) | `(B, 64)` |
| **CNN Feature Map** | `(B, 3, 224, 224)` | EfficientNet-B0 Backbone + Global Pooling | `(B, 1280)` |
| **ViT Token Map** | `(B, 3, 224, 224)` | ViT-Tiny/16 Patch Embedding + Transformer Encoder | `(B, 197, 192)` |
| **ViT [CLS] Token** | `(B, 197, 192)` | Slice Index `[:, 0, :]` | `(B, 192)` |
| **CNN Projected Latent** | `(B, 1280)` | `Linear(1280, 256) -> LayerNorm(256)` | `(B, 256)` |
| **ViT Projected Latent** | `(B, 192)` | `Linear(192, 256) -> LayerNorm(256)` | `(B, 256)` |
| **MAF Fused Representation**| `(B, 256), (B, 256), (B, 64)` | Adaptive Gating + Residual Refinement MLP | `(B, 256)` |
| **Binary Logits** | `(B, 256)` | `Linear(256, 2)` | `(B, 2)` |
| **Subtype Logits** | `(B, 256)` | `Linear(256, 8)` | `(B, 8)` |

---

## 3. Detailed Component Architecture

### 3.1 Branch 1: CNN Local Morphology Encoder (EfficientNet-B0)
- **Model Source:** `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)`
- **Structure:** Mobile Inverted Bottleneck Convolutions (MBConv) with Squeeze-and-Excitation (SE) optimization.
- **Forward Path:**
  ```python
  x_feat = self.cnn_backbone.forward_features(x)  # (B, 1280, 7, 7)
  x_pool = self.cnn_backbone.global_pool(x_feat)   # (B, 1280)
  f_cnn  = self.cnn_proj(x_pool)                   # (B, 256)
  ```
- **CNN Projection Layer (`cnn_proj`):**
  - `nn.Linear(in_features=1280, out_features=256, bias=True)`
  - `nn.LayerNorm(normalized_shape=256)`

### 3.2 Branch 2: ViT Global Context Encoder (ViT-Tiny/16)
- **Model Source:** `timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)`
- **Structure:** 12 Transformer encoder blocks, 3 attention heads per block, hidden dimension $D=192$, MLP expansion ratio $4.0$.
- **Patch Partitioning:** $16 \times 16$ non-overlapping patches yielding $14 \times 14 = 196$ spatial visual tokens $+ 1$ prependable `[CLS]` token $\rightarrow 197$ total tokens.
- **Forward Path:**
  ```python
  x_tokens  = self.vit_backbone.forward_features(x) # (B, 197, 192)
  cls_token = x_tokens[:, 0, :]                    # (B, 192)
  f_vit     = self.vit_proj(cls_token)             # (B, 256)
  ```
- **ViT Projection Layer (`vit_proj`):**
  - `nn.Linear(in_features=192, out_features=256, bias=True)`
  - `nn.LayerNorm(normalized_shape=256)`

---

## 4. Magnification-Aware Adaptive Fusion (MAF) Module

The MAF module bridges multi-scale visual features by parameterizing the fusion balance directly as a function of optical magnification.

```
       f_cnn ∈ R^(B × 256)          f_vit ∈ R^(B × 256)          e_m ∈ R^(B × 64)
              │                            │                            │
              └────────────────────────────┼────────────────────────────┘
                                           │
                                  Concatenation [576-d]
                                           │
                                    Linear(576, 1)
                                           │
                                        Sigmoid
                                           │
                                           ▼
                                    α ∈ (0, 1)
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
             f_cnn scaled by α                   f_vit scaled by (1 - α)
                         │                                   │
                         └─────────────────┬─────────────────┘
                                           │
                                    f_fused ∈ R^(B × 256)
                                           │
                               ┌───────────┴───────────┐
                               │                       │
                               │              Concat([f_fused, e_m]) [320-d]
                               │                       │
                               │                Linear(320, 256)
                               │                       │
                               │                     GELU()
                               │                       │
                               │                  Dropout(0.3)
                               │                       │
                               │                Linear(256, 256)
                               │                       │
                               │              f_res ∈ R^(B × 256)
                               │                       │
                               └──────────► (+) ◄──────┘
                                             │
                                             ▼
                                    f_out ∈ R^(B × 256)
```

### Mathematical Equations
1. **Magnification Embedding:**
   $$e_m = \mathbf{E}[m], \quad \mathbf{E} \in \mathbb{R}^{4 \times 64}, \quad m \in \{0, 1, 2, 3\}$$

2. **Gating Parameter $\alpha$:**
   $$\alpha = \sigma\left(\mathbf{W}_g [f_{\text{cnn}} \,\|\, f_{\text{vit}} \,\|\, e_m] + b_g\right) \in (0, 1)$$
   where $[ \cdot \,\|\, \cdot ]$ is feature concatenation ($256 + 256 + 64 = 576\text{-d}$), $\mathbf{W}_g \in \mathbb{R}^{1 \times 576}$, and $\sigma$ is the Sigmoid activation.

3. **Convex Combination:**
   $$f_{\text{fused}} = \alpha \odot f_{\text{cnn}} + (1 - \alpha) \odot f_{\text{vit}} \in \mathbb{R}^{B \times 256}$$

4. **Scale-Conditioned Residual Refinement:**
   $$f_{\text{res}} = \mathbf{W}_2 \cdot \left(\text{Dropout}_{0.3}\left(\text{GELU}\left(\mathbf{W}_1 [f_{\text{fused}} \,\|\, e_m] + b_1\right)\right)\right) + b_2$$
   where $\mathbf{W}_1 \in \mathbb{R}^{256 \times 320}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 256}$.

5. **Final Fused Latent Representation:**
   $$f_{\text{out}} = f_{\text{fused}} + f_{\text{res}} \in \mathbb{R}^{B \times 256}$$

---

## 5. Multi-Task Classification Heads

Both downstream diagnostic tasks branch directly from the unified representation $f_{\text{out}} \in \mathbb{R}^{B \times 256}$:

### 5.1 Primary Binary Malignancy Head
- **Architecture:** `nn.Linear(256, 2)`
- **Logits:** $z_{\text{bin}} \in \mathbb{R}^{B \times 2}$
- **Classes:** Index 0 = Benign, Index 1 = Malignant

### 5.2 Histological Subtype Head
- **Architecture:** `nn.Linear(256, 8)`
- **Logits:** $z_{\text{sub}} \in \mathbb{R}^{B \times 8}$
- **Subtype Index Map:**
  - `0`: Adenosis (`A`) - Benign
  - `1`: Fibroadenoma (`F`) - Benign
  - `2`: Phyllodes Tumor (`PT`) - Benign
  - `3`: Tubular Adenoma (`TA`) - Benign
  - `4`: Ductal Carcinoma (`DC`) - Malignant
  - `5`: Lobular Carcinoma (`LC`) - Malignant
  - `6`: Mucinous Carcinoma (`MC`) - Malignant
  - `7`: Papillary Carcinoma (`PC`) - Malignant

---

## 6. Hierarchical Multi-Task Loss Formulation

The total objective function is a convex combination of three complementary losses:

$$L_{\text{total}} = 0.3 \cdot L_{\text{binary}} + 0.6 \cdot L_{\text{subtype}} + 0.1 \cdot L_{\text{consistency}}$$

### 6.1 Weighted Binary Cross-Entropy ($L_{\text{binary}}$)
$$L_{\text{binary}} = -\frac{1}{B} \sum_{i=1}^B w_{y_i^{\text{bin}}} \log \left(\frac{\exp(z_{\text{bin}, i}[y_i^{\text{bin}}])}{\sum_{c=0}^1 \exp(z_{\text{bin}, i}[c])}\right)$$

### 6.2 Weighted Subtype Cross-Entropy ($L_{\text{subtype}}$)
$$L_{\text{subtype}} = -\frac{1}{B} \sum_{i=1}^B w_{y_i^{\text{sub}}} \log \left(\frac{\exp(z_{\text{sub}, i}[y_i^{\text{sub}}])}{\sum_{k=0}^7 \exp(z_{\text{sub}, i}[k])}\right)$$

*Per-fold class weights $w_c$ compensate for significant dataset class imbalance:*
$$w_c = \frac{N_{\text{total}}}{C \cdot \max(N_c, 1)}$$

### 6.3 Hierarchical Probability Consistency Loss ($L_{\text{consistency}}$)
To prevent semantic contradiction between the two heads, the 8-class subtype softmax distribution $\hat{p}^{\text{sub}} = \text{Softmax}(z_{\text{sub}})$ is marginalized into two super-class probabilities:
$$\hat{p}_{\text{benign}}^{\text{sub}} = \sum_{k=0}^3 \hat{p}_k^{\text{sub}}, \quad \hat{p}_{\text{malignant}}^{\text{sub}} = \sum_{k=4}^7 \hat{p}_k^{\text{sub}}$$

The binary head prediction acts as a detached reference distribution $q = \text{Softmax}(z_{\text{bin}})_{\text{detached}} \in \mathbb{R}^{B \times 2}$. The Kullback-Leibler divergence is minimized:
$$L_{\text{consistency}} = D_{\text{KL}}\left(\hat{p}^{\text{sub}} \,\|\, q\right) = \sum_{c \in \{0, 1\}} q_c \log\left(\frac{q_c}{\hat{p}_c^{\text{sub}} + \epsilon}\right)$$

---

## 7. Optimization & Training Specification

| Hyperparameter | Value | Description / Rationale |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay regularization |
| **Base Learning Rate** | `1e-5` | Fine-tuning rate for pretrained EfficientNet-B0 & ViT-Tiny backbones |
| **Head Learning Rate** | `1e-4` | Faster convergence rate for randomly initialized MAF & classification heads |
| **Weight Decay** | `1e-4` | L2 weight penalty preventing overfitting |
| **Batch Size** | `16` | Optimized for GPU memory and gradient variance |
| **Max Epochs** | `30` | Full cross-validation convergence budget |
| **Warmup Epochs** | `5` | Linear learning rate warmup |
| **Early Stopping Patience** | `8` | Monitored on validation subtype Macro-F1 |
| **Gradient Clipping** | `1.0` | $L_2$ norm gradient threshold |
| **Stain Augmentation Prob** | `0.5` | Macenko optical density stain perturbation rate |
| **Execution Precision** | `FP32` | 32-bit floating point precision (eliminating ROCm BF16 page faults) |
| **Random Seed** | `42` | Global deterministic seed |
