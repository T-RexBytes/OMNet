# Publication Figure Captions

### Figure 1: Proposed OMNet-V3 Dual-Branch Architecture
**Figure 1.** Schematic diagram of the proposed OMNet-V3 multi-task histopathology framework. Input patches ($224 \times 224 \times 3$) and magnification metadata are processed through parallel feature extraction branches: an EfficientNet-B0 backbone for localized morphological features ($1280$-d) and a ViT-Tiny/16 backbone for global self-attention context ($192$-d). The Magnification-Aware Adaptive Fusion (MAF) module computes a scale-dependent dynamic gate $\alpha(m) \in (0, 1)$ conditioned on a $64$-d magnification embedding $e_m$, performing convex feature blending and residual refinement into a unified $256$-d latent vector. Multi-task heads simultaneously predict binary malignancy (2-class) and histological subtype (8-class), regularized by a hierarchical KL-divergence consistency loss.

### Figure 2: Adaptive Fusion Gate Progression across Optical Magnifications
**Figure 2.** Learned adaptive fusion gating parameter $\alpha(m)$ distribution across BreakHis optical magnification scales ($40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$). The boxplot demonstrates a monotonic scale transition: lower magnifications ($40\text{X}$) exhibit higher reliance on wide-field contextual representations (mean $\alpha = 0.6850$), while high magnifications ($400\text{X}$) prioritize localized high-frequency convolutional features (mean $\alpha = 0.4701$).

### Figure 3: Receiver Operating Characteristic (ROC) and Precision-Recall Curves
**Figure 3.** Cross-validation diagnostic curves for OMNet-V3. (Left) Binary malignancy ROC curve aggregated across 5 patient-disjoint folds showing strong discrimination power (AUC $> 0.88$). (Right) One-vs-Rest (OvR) multiclass ROC curves for each of the 8 histological subtypes.

### Figure 4: Diagnostic Confusion Matrices
**Figure 4.** Test-set confusion matrices for (Left) binary malignancy classification and (Right) fine-grained 8-class histological subtyping. Values indicate raw counts with row-normalized percentages in parentheses, demonstrating diagonal dominance and identifying clinical confusion pairs (e.g., Ductal Carcinoma vs. Lobular Carcinoma).

### Figure 5: t-SNE Feature Embedding Visualizations
**Figure 5.** Two-dimensional t-SNE manifold projections of the 256-dimensional bottleneck latent embeddings extracted from unseen test samples, showing distinct topological clustering between benign and malignant tissue specimens.
