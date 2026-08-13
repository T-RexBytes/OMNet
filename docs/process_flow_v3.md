# OMNet-V3 — End-to-End Process Flow & Architecture Specification

This document provides a comprehensive technical overview of the **OMNet-V3** research architecture, data processing pipeline, magnification-aware adaptive fusion mechanism, hierarchical classification heads, loss function, and patient-disjoint cross-validation protocol.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Env["1. Environment & Data Discovery"]
        A[Colab / Local Setup] --> B[GPU Detection T4/AMP]
        B --> C[Mount Google Drive /content/drive/MyDrive/BreaKHis]
        C --> D[Recursive Regex Filename Parser]
        D --> E[Extract Patient ID, Subtype, Magnification, Binary Label]
    end

    subgraph Splits["2. Patient-Disjoint Splitting"]
        E --> F[Group by Patient ID: YEAR-SLIDE_ID]
        F --> G[StratifiedGroupKFold 5-Splits, Seed=42]
        G --> H[Hold out 10% Patient-Disjoint Val per fold]
        H --> I[Zero Identity Leakage Verified]
    end

    subgraph Pipe["3. Robust Data Pipeline"]
        I --> J[Dataset-Specific / ImageNet Normalize]
        J --> K[Train Augmentations: Crop, Flips, Rot90]
        K --> L[Online Macenko Stain Augmentation p=0.5]
        L --> M[PyTorch Weighted DataLoader]
    end

    subgraph Architecture["4. OMNet-V3 Model Architecture"]
        M --> Img[Input Image: 3 x 224 x 224]
        M --> Mag[Magnification Index: m ∈ {0:40X, 1:100X, 2:200X, 3:400X}]
        
        Img --> CNN[EfficientNet-B0 Backbone\nImageNet Pretrained]
        Img --> ViT[ViT-Tiny/16 Backbone\nImageNet Pretrained]
        
        CNN --> f_cnn[f_cnn ∈ R^1280\nLocal Morphological Features]
        ViT --> f_vit[f_vit ∈ R^192\nGlobal Context Features]
        
        f_cnn --> ProjCNN[Linear 1280 -> 256 + LayerNorm]
        f_vit --> ProjViT[Linear 192 -> 256 + LayerNorm]
        Mag --> MagEmbed[Embedding Table: 4 x 64]
        
        ProjCNN --> Cat[Concat: 256 + 256 + 64 = 576-d]
        ProjViT --> Cat
        MagEmbed --> Cat
        
        Cat --> Gate[Linear 576 -> 1 + Sigmoid]
        Gate --> Alpha[Scalar Gate: Alpha ∈ 0, 1]

        Alpha --> Blend[Alpha * ProjCNN + 1 - Alpha * ProjViT]
        Blend --> ResidualMLP[Residual MLP: Linear 320->256 -> GELU -> Dropout 0.3 -> Linear 256->256]
        ResidualMLP --> f_out[Fused Embedding: f_out ∈ R^256]
        
        f_out --> BinaryHead[Binary Classifier Head\nLinear 256 -> 2]
        f_out --> SubtypeHead[Subtype Classifier Head\nLinear 256 -> 8]
    end

    subgraph Training["5. Multi-Task Training & Loss"]
        BinaryHead --> BinaryLogits[Binary Logits: Benign / Malignant]
        SubtypeHead --> SubtypeLogits[Subtype Logits: 8 Tumor Subtypes]
        
        BinaryLogits --> Loss[Hierarchical Multi-Task Loss]
        SubtypeLogits --> Loss
        
        Loss --> LossEq[L_total = 0.3*L_binary + 0.6*L_subtype + 0.1*L_consistency]
        LossEq --> Opt[AdamW Optimizer + Cosine Annealing + Warmup]
        Opt --> Amp[FP16 Mixed Precision AMP]
        Amp --> Chk[Save checkpoint_latest.pth & best_model.pth to Drive]
    end

    subgraph Evaluation["6. Comprehensive Evaluation & Interpretability"]
        Chk --> EvalTest[Test Set Inference per Fold]
        EvalTest --> Metrics[13 Test Metrics + 95% Confidence Intervals]
        EvalTest --> PatientScore[Spanhol Patient Recognition Rate]
        EvalTest --> Visuals[Confusion Matrices, ROC Curves, t-SNE]
        EvalTest --> GateAnalysis[Fusion Gate Alpha Distribution by Mag]
        EvalTest --> GradCAM[EfficientNet-B0 Grad-CAM Heatmaps]
        EvalTest --> Export[Export experiment_config.json & CSVs]
    end
```

---

## 2. Model Specifications & Design Rationale

### 2.1 Feature Extractors
- **EfficientNet-B0 Branch**: Extracts high-resolution, local morphological details (cellular boundaries, nuclear atypia, gland architecture). Output shape: `(B, 1280)`.
- **ViT-Tiny/16 Branch**: Extracts global contextual and structural relationships across spatial patch tokens via self-attention. Output shape: `(B, 192)`.

### 2.2 Magnification-Aware Adaptive Fusion (MAF)
Higher magnifications ($400\times$) reveal localized nuclear details where CNN features dominate, whereas lower magnifications ($40\times$) emphasize gross architecture where ViT features dominate. The MAF module learns scale-conditioned weighting:
$$\text{Gate Input} = [f_{\text{cnn\_proj}}, f_{\text{vit\_proj}}, e_m] \in \mathbb{R}^{576}$$
$$\alpha = \sigma(W \cdot \text{Gate Input} + b) \in (0, 1)$$
$$f_{\text{fused}} = \alpha \cdot f_{\text{cnn\_proj}} + (1 - \alpha) \cdot f_{\text{vit\_proj}} \in \mathbb{R}^{256}$$
$$f_{\text{out}} = f_{\text{fused}} + \text{MLP}([f_{\text{fused}}, e_m]) \in \mathbb{R}^{256}$$

### 2.3 Hierarchical Multi-Task Loss
$$L_{\text{total}} = 0.3 L_{\text{binary}} + 0.6 L_{\text{subtype}} + 0.1 L_{\text{consistency}}$$
- $L_{\text{binary}}$: Weighted Cross-Entropy over Benign vs. Malignant ($2$ classes).
- $L_{\text{subtype}}$: Weighted Cross-Entropy over $8$ tumor subtypes (Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma, Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma).
- $L_{\text{consistency}}$: KL Divergence between subtype marginal probabilities and binary head probabilities:
$$D_{KL}\left(\left[\sum_{i \in \text{benign}} P(s_i), \sum_{j \in \text{malignant}} P(s_j)\right] \parallel P_{\text{binary}}\right)$$

---

## 3. Patient-Disjoint Validation Protocol

- **Dataset**: Full BreaKHis ($7,909$ images across $82$ patients).
- **Splitter**: `StratifiedGroupKFold(n_splits=5, shuffle=True, seed=42)`.
- **Group Key**: `patient_id` (`YEAR-SLIDE_ID`). All images ($40\times, 100\times, 200\times, 400\times$) of a patient remain strictly within the same fold.
- **Hold-out Validation**: $10\%$ patient-disjoint validation set sampled per fold for early stopping.
- **Zero Identity Leakage**: Verified with explicit set intersection assertions.

---

## 4. Drive Output Directory Layout

All results automatically save to `/content/drive/MyDrive/OMNet/v3_outputs/`:

```
v3_outputs/
├── fold_0/
│   ├── best_model.pth            # Peak validation macro-F1 checkpoint
│   ├── checkpoint_latest.pth     # Auto-resume checkpoint
│   ├── training_history.csv      # Per-epoch metrics & LR
│   └── test_metrics.json         # Fold test metrics
├── fold_1/ ... fold_4/
├── aggregated_results.json       # Cross-fold mean, std, 95% CI
├── confusion_matrix_8class.png   # Count & normalized confusion matrices
├── confusion_matrix_binary.png   # Binary confusion matrix
├── roc_curves.png                # Per-subtype OvR ROC curves with AUC
├── fusion_gate_analysis.png      # Alpha distribution boxplot & bar chart across 40X-400X
├── gradcam_samples.png           # EfficientNet-B0 Grad-CAM overlays across magnifications
├── tsne_embeddings.png           # 2D t-SNE feature visualization by Subtype & Magnification
├── patient_score_summary.csv     # Per-patient recognition rate summary
└── experiment_config.json        # Full experiment config and system dump
```
