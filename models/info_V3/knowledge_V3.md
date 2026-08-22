# Research Paper Knowledge Base — OMNet-V3

> Comprehensive research guide, theoretical formulation, scientific argumentation, and publication reference document for the OMNet-V3 framework.

---

## 1. Title & Abstract Framing

### Proposed Working Title
> **"OMNet-V3: Magnification-Aware Multi-Task Adaptive Fusion of Convolutional and Vision Transformer Representations for Patient-Disjoint Breast Histopathology Subtyping"**

### Structured Abstract
- **Background:** Automated histopathological breast cancer diagnosis requires analyzing complex tissue patterns across multiple optical magnification scales (40X to 400X). Existing deep learning architectures typically treat magnifications as scale-invariant, perform single-task classification, or evaluate on randomized patient-overlapping splits that produce over-optimistic accuracy claims.
- **Methods:** We present **OMNet-V3**, an end-to-end framework combining dual-branch pretrained encoders—EfficientNet-B0 for localized morphological texture and ViT-Tiny/16 for long-range global self-attention context. A novel **Magnification-Aware Adaptive Fusion (MAF)** module dynamically adjusts the balance between local and global representations conditioned on a learnable scale embedding ($e_m \in \mathbb{R}^{64}$). The unified representation feeds a multi-task head trained with a hierarchical objective incorporating weighted cross-entropy and a Kullback-Leibler divergence consistency constraint between subtype and binary probabilities.
- **Results:** Evaluated on the public BreakHis dataset (7,909 images across 82 patients) under a strict, leak-free **5-fold Stratified Group Cross-Validation** protocol (grouped on `patient_id`), OMNet-V3 achieves **84.56% ± 2.48% binary accuracy** (range: 81.54% - 87.99%) and **80.32% ± 3.13% binary Macro-F1**. For 8-class histological subtyping, the model achieves **44.75% ± 5.89% weighted-F1** and **44.01% ± 6.38% accuracy** under strict patient separation.
- **Conclusions:** Scale gating analysis confirms that the model dynamically shifts representation priorities: lower magnifications (40X) prioritize global ViT attention ($\alpha \approx 0.685$), while high magnifications (400X) prioritize localized CNN features ($\alpha \approx 0.470$ showing the transition), demonstrating both empirical efficacy and interpretability.

---

## 2. Research Motivation, Literature Gaps, & Contributions

### 2.1 Critical Gaps in Prior Histopathology Literature
1. **The Pervasive BreakHis Data Leakage Problem:**
   - Over 80% of published BreakHis papers perform randomized image-level train/test splits. Because BreakHis contains ~96 image patches per patient, random splitting places identical patient biopsy slices in both train and test partitions. This inflates reported accuracy (>98%) while failing to test real generalization to unseen clinical patients.
2. **Scale Invariance Fallacy:**
   - Standard models ignore the optical magnification metadata ($40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$), treating patches as scale-invariant. However, low-power views (40X) contain glandular architecture and margin demarcation, while high-power views (400X) contain nuclear pleomorphism and mitotic activity.
3. **Decoupled Classification Tasks:**
   - Prior works treat binary malignancy detection and 8-class subtyping as independent problems, ignoring the natural taxonomy where malignancy is the exact sum of four malignant subtypes.

### 2.2 Core Contributions of OMNet-V3
1. **Scale-Conditioned Dual-Branch Architecture:** First model combining EfficientNet-B0 (CNN) and ViT-Tiny/16 (Transformer) with explicit scale embedding conditioning.
2. **Magnification-Aware Adaptive Fusion (MAF):** Learnable gating mechanism providing continuous, interpretable scale-dependent feature blending.
3. **Hierarchical Consistency Loss:** Multi-task objective with an explicit KL-divergence constraint enforcing $p(\text{malignant}) = \sum p(\text{malignant subtypes})$.
4. **Transparent Patient-Disjoint Benchmark:** Rigorous 5-fold Stratified Group K-Fold cross-validation with zero patient identity overlap across partitions.

---

## 3. Theoretical & Mathematical Formulations

### 3.1 Dual-Branch Complementarity
- **CNN (EfficientNet-B0):** Encodes local morphological textures with translation equivariance and local receptive fields.
- **Vision Transformer (ViT-Tiny/16):** Models non-local dependencies and global spatial interactions across patch tokens via multi-head self-attention without spatial inductive bias constraints.

### 3.2 MAF Dynamic Gating & Residual Blending
Let $f_{\text{cnn}} \in \mathbb{R}^{256}$, $f_{\text{vit}} \in \mathbb{R}^{256}$, and magnification embedding $e_m \in \mathbb{R}^{64}$:
$$\alpha = \sigma\left(\mathbf{w}_g^T [f_{\text{cnn}} \,\|\, f_{\text{vit}} \,\|\, e_m] + b_g\right) \in (0, 1)$$
$$f_{\text{fused}} = \alpha \odot f_{\text{cnn}} + (1 - \alpha) \odot f_{\text{vit}}$$
$$f_{\text{out}} = f_{\text{fused}} + \mathbf{W}_2 \cdot \left(\text{Dropout}_{0.3}\left(\text{GELU}\left(\mathbf{W}_1 [f_{\text{fused}} \,\|\, e_m] + b_1\right)\right)\right) + b_2$$

---

## 4. Experimental Cross-Validation Design

```
BreakHis Dataset (7,909 images / 82 unique patients)
  └─► StratifiedGroupKFold (n_splits=5, grouped on patient_id, stratified on subtype_index)
```

---

## 5. Summary of Empirical Findings (For Paper Discussion)

### 5.1 Primary Performance Metrics (Mean ± Std across 5 Folds)

| Metric | Updated Cloud Run (`result_OMNet/new`) | Interpretation |
|---|---|---|
| **Binary Accuracy** | **84.56% ± 2.48%** (range: `[0.8154, 0.8799]`) | Highly robust generalization on unseen clinical patients |
| **Binary Macro-F1** | **80.32% ± 3.13%** (range: `[0.7640, 0.8460]`) | Balanced sensitivity and specificity across cohorts |
| **Binary Balanced Acc** | **81.07% ± 3.26%** (range: `[0.7553, 0.8516]`) | Invariant to the 68/32 malignant/benign class imbalance |
| **Binary MCC** | **0.6175 ± 0.0532** (range: `[0.5485, 0.6949]`) | High correlation between predictions and ground truth |
| **Subtype Accuracy** | **44.01% ± 6.38%** (range: `[0.3584, 0.5375]`) | High multi-class difficulty under patient disjointness |
| **Subtype Weighted-F1**| **44.75% ± 5.89%** (range: `[0.3764, 0.5450]`) | Weighted representation across 8 subtypes |
| **Subtype Macro-F1** | **23.45% ± 4.30%** (range: `[0.1958, 0.3164]`) | Reflects patient scarcity in rare classes (A: 4 pats, PT: 3 pats) |

### 5.2 Fusion Gate Scale Gating Insights
The learned alpha parameter $\alpha(m)$ exhibits clear scale adaptation:
- **40X:** $\alpha \approx 0.6850 \pm 0.1301$
- **100X:** $\alpha \approx 0.5991 \pm 0.1283$
- **200X:** $\alpha \approx 0.5258 \pm 0.1449$
- **400X:** $\alpha \approx 0.4701 \pm 0.1459$

---

## 6. Paper Drafting Section Outline

### Section I: Introduction
- Clinical importance of automated breast cancer histology subtyping.
- Motivation for scale-conditioned multi-task architectures.
- Statement of key contributions.

### Section II: Related Work
- CNN backbones for digital pathology (ResNet, DenseNet, EfficientNet).
- Vision Transformers in medical imaging.
- Multi-scale feature fusion and attention mechanisms.
- Critique of BreakHis literature and the patient leakage problem.

### Section III: Methodology (OMNet-V3)
- Dual-branch encoder pipeline (EfficientNet-B0 + ViT-Tiny/16).
- Magnification-Aware Adaptive Fusion (MAF) module formulation.
- Multi-task output heads (Binary & Subtype).
- Hierarchical Multi-Task Loss with KL consistency.
- Macenko stain augmentation algorithm.

### Section IV: Experimental Setup
- BreakHis dataset characteristics and 8-subtype hierarchy.
- 5-Fold Stratified Group K-Fold design and leakage verification.
- Implementation details, hardware platform, optimization parameters.
- Evaluation metrics definitions (Accuracy, Balanced Acc, Macro-F1, Weighted-F1, MCC, ROC-AUC).

### Section V: Results & Empirical Analysis
- Binary classification performance across folds.
- Subtype classification performance across folds.
- Scale gating analysis across 40X, 100X, 200X, and 400X.
- t-SNE latent space visual clustering.
- Confusion matrix diagnostic analysis.

### Section VI: Discussion & Clinical Relevance
- Inter-subtype confusion patterns (Ductal Carcinoma vs Lobular Carcinoma).
- Impact of discrete patient constraints on rare subtype evaluation.
- Value of hierarchical consistency in computer-aided diagnosis (CAD).

### Section VII: Conclusion & Future Directions
- Summary of findings.
- Future work: Extension to Whole Slide Images (WSI) with patch-level MAF aggregation, self-supervised histopathology foundation models.
