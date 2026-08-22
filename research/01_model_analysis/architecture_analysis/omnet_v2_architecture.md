# Architecture Specification — OMNet-V2

> Comprehensive, end-to-end architectural specification for **OMNet-V2: Transfer-Learning Framework for Breast Cancer Histopathology Classification**.

---

## 1. Architectural Overview

**OMNet-V2** is a modular transfer-learning and ensemble framework specifically optimized for binary histopathology malignancy classification (Benign vs. Malignant) on the BreakHis-200X dataset.

The architecture systematically benchmarks and unifies two complementary visual paradigms:
1. **Convolutional Neural Network (CNN) Backbone:** ImageNet-pretrained **EfficientNet-B0** capturing fine-grained localized textural primitives, cellular boundaries, and nuclear chromatin distribution via inverted residual blocks (MBConv) and Squeeze-and-Excitation attention.
2. **Vision Transformer (ViT) Backbone:** ImageNet-pretrained **ViT-B/16** capturing long-range spatial context, stromal organization, and inter-glandular architectural patterns via 12 Transformer encoder blocks with multi-head self-attention.
3. **Customized 2-Layer Projection & Classification Head:** A non-linear bottleneck projector transforming high-dimensional backbone features into a standardized 256-dimensional latent embedding space, stabilized by Layer Normalization, GELU non-linearity, and heavy Dropout (0.4) before binary logits projection.
4. **2-Phase Progressive Fine-Tuning Engine:** A staged optimization strategy with frozen backbone warm-up (Phase 1) followed by selective unfreezing of top $N=2$ representation layers with differential learning rates (Phase 2).
5. **Soft-Voting Ensemble Engine:** An uncertainty-mitigating ensemble combining output probabilities:
   $$p_{\text{ensemble}}(y = 1 \mid x) = \frac{1}{2} \left( p_{\text{effnet}}(y = 1 \mid x) + p_{\text{vit}}(y = 1 \mid x) \right)$$

---

## 2. Input Pipeline & Tensor Dimensions

```
Input Patch: x ∈ R^(B × 3 × 224 × 224) [ImageNet RGB Normalized]
```

| Pipeline Stage | Input Shape | Component / Operation | Output Shape |
|---|---|---|---|
| **Input Image Tile** | `(B, 3, 224, 224)` | Preprocessing / RGB Normalization | `(B, 3, 224, 224)` |
| **EfficientNet Trunk**| `(B, 3, 224, 224)` | `tvm.efficientnet_b0(weights=IMAGENET1K_V1)` | `(B, 1280, 7, 7)` |
| **EfficientNet Pool** | `(B, 1280, 7, 7)` | `AdaptiveAvgPool2d((1, 1))` + `Flatten(1)` | `(B, 1280)` |
| **ViT-B/16 Trunk** | `(B, 3, 224, 224)` | `tvm.vit_b_16(weights=IMAGENET1K_V1)` | `(B, 197, 768)` |
| **ViT-B/16 [CLS]** | `(B, 197, 768)` | Class Token Slicing `[:, 0]` | `(B, 768)` |
| **Latent Embedding** | `(B, D_in)` | `Linear(D_in, 256) -> LayerNorm(256) -> GELU()` | `(B, 256)` |
| **Regularized Latent**| `(B, 256)` | `Dropout(p=0.4)` | `(B, 256)` |
| **Binary Logits** | `(B, 256)` | `Linear(256, 2)` | `(B, 2)` |

---

## 3. Detailed Component Architecture

### 3.1 Backbone Configuration & Feature Dimensions

```python
BACKBONE_CONFIGS = {
    'efficientnet_b0': {
        'builder': tvm.efficientnet_b0,
        'weights': tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'feature_dim': 1280,
    },
    'vit_b_16': {
        'builder': tvm.vit_b_16,
        'weights': tvm.ViT_B_16_Weights.IMAGENET1K_V1,
        'feature_dim': 768,
    },
}
```

### 3.2 EfficientNet-B0 Encoder Module
- **Trunk:** Mobile Inverted Bottleneck Convolutional blocks (MBConv1, MBConv6) with Squeeze-and-Excitation (SE) modules.
- **Input Resolution:** $224 \times 224 \times 3$.
- **Feature Extraction:**
  $$\mathbf{F}_{\text{cnn}} = \text{AdaptiveAvgPool2d}(\text{Features}(x)) \in \mathbb{R}^{B \times 1280}$$
- **Trainable Parameters:** $\approx 5.3 \text{M}$ total parameters.

### 3.3 ViT-B/16 Transformer Encoder Module
- **Trunk:** Standard Vision Transformer Base architecture:
  - Patch size: $16 \times 16 \rightarrow N = (224/16)^2 = 196$ patch tokens.
  - Hidden Dimension: $D = 768$.
  - Encoder Depth: 12 Transformer layers.
  - Multi-Head Self-Attention: 12 attention heads ($d_k = 64$).
  - MLP Dimension: $3072$ ($4 \times D$).
- **Feature Extraction:**
  $$\mathbf{Z} = \text{TransformerEncoder}(\text{PatchEmbed}(x) + \mathbf{E}_{\text{pos}}) \in \mathbb{R}^{B \times 197 \times 768}$$
  $$\mathbf{F}_{\text{vit}} = \mathbf{Z}[:, 0] \in \mathbb{R}^{B \times 768} \quad (\text{[CLS] token})$$
- **Trainable Parameters:** $\approx 86.6 \text{M}$ total parameters.

---

## 4. Custom 2-Layer Classification Head

In OMNet-V2, the default linear classifier is replaced by a non-linear bottleneck head designed to prevent catastrophic forgetting and regularize high-dimensional embeddings:

```
Backbone Feature Vector F ∈ R^(B × D_in)  [D_in = 1280 for EfficientNet, 768 for ViT]
                     │
                     ▼
             Linear(D_in, 256)
                     │
                     ▼
               LayerNorm(256)
                     │
                     ▼
                  GELU()
                     │
                     ▼
               Dropout(p=0.4)
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
Feature Embedding Output    Linear(256, 2)
  e ∈ R^(B × 256)                │
                                 ▼
                          Binary Logits z ∈ R^(B × 2)
```

### Mathematical Formulation
1. **Projection to Shared Latent Space:**
   $$h_1 = \text{LayerNorm}\left(\mathbf{W}_{\text{proj}} \mathbf{F} + b_{\text{proj}}\right) \in \mathbb{R}^{B \times 256}$$
2. **Non-linear Activation & Regularization:**
   $$e = \text{Dropout}_{0.4}\left(\text{GELU}(h_1)\right) \in \mathbb{R}^{B \times 256}$$
3. **Logits Projection:**
   $$z = \mathbf{W}_{\text{out}} e + b_{\text{out}} \in \mathbb{R}^{B \times 2}$$
4. **Class Probability Output:**
   $$p(y = c \mid x) = \frac{\exp(z_c)}{\exp(z_0) + \exp(z_1)}, \quad c \in \{0 (\text{Benign}), 1 (\text{Malignant})\}$$

---

## 5. Selective Layer Unfreezing Mechanism

OMNet-V2 implements fine-grained structural parameter control to enable stable transfer learning:

```python
def set_trainable_layers(self, unfreeze_all=False, unfreeze_last_n=2):
    # Head is ALWAYS trainable
    for param in self.head.parameters():
        param.requires_grad = True

    if unfreeze_all:
        for param in self.backbone.parameters():
            param.requires_grad = True
        return

    # Freeze entire backbone by default
    for param in self.backbone.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        if self.backbone_name == 'efficientnet_b0':
            # Unfreeze top N stages in backbone.features
            n_features = len(self.backbone.features)
            for i in range(n_features - unfreeze_last_n, n_features):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = True
        elif self.backbone_name == 'vit_b_16':
            # Unfreeze top N Transformer encoder blocks
            n_blocks = len(self.backbone.encoder.layers)
            for i in range(n_blocks - unfreeze_last_n, n_blocks):
                for param in self.backbone.encoder.layers[i].parameters():
                    param.requires_grad = True
```

---

## 6. Two-Phase Progressive Training Engine

To prevent destroying pretrained representations during early iterations, OMNet-V2 follows a two-phase schedule:

### Phase 1: Classification Head Warm-up (3 Epochs)
- **Trainable Parameters:** Classification head only (backbone frozen).
- **Learning Rate:** $\eta_{\text{head}} = 10^{-3}$, $\eta_{\text{backbone}} = 0$.
- **Goal:** Adapt random projection weights to backbone feature geometry without backpropagating noisy gradients into pretrained layers.

### Phase 2: Differential Fine-Tuning (12 Epochs with Early Stopping)
- **Trainable Parameters:** Top $N=2$ backbone blocks $+$ Classification head.
- **Differential Learning Rates:**
  - $\eta_{\text{head}} = 10^{-3}$
  - $\eta_{\text{backbone}} = 10^{-5}$ ($100\times$ smaller learning rate)
- **LR Scheduler:** Cosine Annealing scheduler ($T_{\text{max}} = 12$, $\eta_{\text{min}} = 10^{-7}$).
- **Early Stopping:** Patience $= 4$ epochs monitored on validation ROC-AUC.

---

## 7. Loss Functions & Class Imbalance Handling

### 7.1 Cross-Entropy with Label Smoothing
$$L_{\text{CE}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=0}^1 q_{i,c} \log p_{i,c}$$
where smoothed targets $q_{i,c} = (1 - \epsilon) y_{i,c} + \frac{\epsilon}{2}$ with $\epsilon = 0.05$.

### 7.2 Focal Loss Option ($\gamma = 2.0$)
$$L_{\text{Focal}} = -\frac{1}{B} \sum_{i=1}^B w_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$

### 7.3 Imbalance Mitigation Policy
- **Inverse Frequency Class Weights:**
  $$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}$$
- **Weighted Random Oversampling:** `WeightedRandomSampler` applied to training batches to guarantee balanced class draws per minibatch.

---

## 8. Ensemble Fusion Mechanism

OMNet-V2 computes an unweighted average of predicted posterior class probabilities:

$$p_{\text{ensemble}}(y = 1 \mid x) = \frac{1}{2} \left[ \sigma(z_{\text{effnet}}[1] - z_{\text{effnet}}[0]) + \sigma(z_{\text{vit}}[1] - z_{\text{vit}}[0]) \right]$$

This ensemble combines the high sensitivity of ViT-B/16 ($97.45\%$) with the spatial discriminability of EfficientNet-B0, resulting in the highest specificity ($55.95\%$) and Precision-Recall AUC ($0.9537$).

---

## 9. Hyperparameter Reference Table

| Hyperparameter | Value | Description |
|---|---|---|
| **Backbone Models** | EfficientNet-B0, ViT-B/16 | Dual transfer-learning backbones |
| **Input Image Size** | $224 \times 224 \times 3$ | RGB image tile dimensions |
| **Batch Size** | 16 | Optimized for GPU memory |
| **Embedding Dimension** | 256 | Latent feature dimension |
| **Head Dropout** | 0.4 | Dropout rate in classification head |
| **Head Learning Rate** | $1 \times 10^{-3}$ | Initial learning rate for head |
| **Backbone Learning Rate** | $1 \times 10^{-5}$ | Fine-tuning learning rate for unfreezed blocks |
| **Weight Decay** | $1 \times 10^{-4}$ | AdamW L2 regularization |
| **Mixed Precision (AMP)** | Enabled (`torch.amp.autocast`) | Float16 GPU acceleration |
| **Phase 1 Epochs** | 3 | Head warmup epoch budget |
| **Phase 2 Epochs** | 12 | Differential fine-tuning budget |
| **Early Stopping Patience** | 4 | Monitored metric: Validation ROC-AUC |
| **CV Scheme** | 5-Fold Stratified CV | 6 epochs per fold |
| **Random Seed** | 42 | Global deterministic seed |
