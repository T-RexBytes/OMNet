# Research Paper Knowledge Base — OMNet-V2

> Comprehensive research framing, theoretical formulations, literature benchmarking, and publication drafting reference for OMNet-V2.

---

## 1. Title & Abstract Framing

### Proposed Working Title
> **"OMNet-V2: Benchmarking Transfer Learning and Soft-Voting Ensembles of Convolutional and Vision Transformer Architectures for Patient-Disjoint Breast Cancer Histopathology Classification"**

### Structured Abstract
- **Background:** Accurate automated binary malignancy classification of breast histopathology biopsies requires robust feature representations capable of distinguishing complex benign proliferations from malignant carcinomas. Standard diagnostic pipelines often suffer from data leakage across patient slices or over-rely on a single architecture family without evaluating the complementary inductive biases of CNNs and Vision Transformers.
- **Methods:** We present **OMNet-V2**, a modular transfer-learning and ensemble framework benchmarking ImageNet-pretrained **EfficientNet-B0** and **ViT-B/16** encoders on the BreakHis-200X benchmark. Both models integrate a custom 2-layer projection head ($D=256$) with Layer Normalization, GELU activations, and Dropout ($p=0.4$), trained via a staged **2-Phase progressive fine-tuning strategy** (head warmup followed by differential backbone fine-tuning with cosine annealing). A soft-voting ensemble combines posterior probabilities. All evaluations follow a strict **patient-level stratified partition** (1,733 CV pool / 280 held-out test patches) guaranteeing zero patient identity overlap.
- **Results:** In 5-fold cross-validation, ViT-B/16 achieves **94.00% ± 1.33% accuracy** and **0.9915 ± 0.0027 ROC-AUC**, while EfficientNet-B0 achieves **84.77% ± 2.44% accuracy** and **0.9754 ± 0.0042 ROC-AUC**. On the independent held-out test set ($N=280$), ViT-B/16 achieves **84.29% accuracy**, **97.45% sensitivity (recall)**, and **0.8998 ROC-AUC**. The soft-voting ensemble achieves **84.29% accuracy**, **96.43% sensitivity**, **55.95% specificity** (highest among all configurations), and **0.9537 Precision-Recall AUC**.
- **Conclusions:** Vision Transformers demonstrate superior sensitivity to global tissue architecture, while CNNs maintain sharp spatial localization. The progressive fine-tuning strategy and soft-voting ensemble provide a strong, leak-free benchmark for computer-aided breast histopathology triage.

---

## 2. Research Problem, Hypotheses, & Scientific Contributions

### 2.1 Problem Formulation & Literature Gaps
1. **Patient Leakage in BreakHis Literature:** Many published models report $>98\%$ accuracy by randomly shuffling image patches, contaminating train and test sets with images from the same patient. This invalidates real-world generalization.
2. **CNN vs. Vision Transformer Inductive Biases:** Prior studies rarely perform head-to-head benchmarking using identical projection heads and staged progressive fine-tuning schedules on the exact same patient-disjoint partitions.
3. **High-Sensitivity Clinical Triage Requirement:** In histopathology screening, minimizing false negatives (high sensitivity/recall) is paramount to avoid missing malignant cases, while preserving sufficient specificity to prevent unnecessary surgical interventions.

### 2.2 Core Contributions of OMNet-V2
1. **Systematic Head-to-Head Architectural Benchmark:** Rigorous comparison between inverted residual CNNs (EfficientNet-B0) and global self-attention Vision Transformers (ViT-B/16) on BreakHis-200X.
2. **2-Phase Progressive Fine-Tuning Protocol:** Decoupled training schedule preventing catastrophic forgetting of pretrained ImageNet representations.
3. **Uncertainty-Reducing Soft-Voting Ensemble:** Combining CNN and ViT posteriors to maximize Precision-Recall AUC ($0.9537$) and specificity ($55.95\%$).
4. **Multi-Faceted Model Interpretability:** Visual validation using Grad-CAM, Score-CAM, and 2D t-SNE latent embedding clustering.

---

## 3. Theoretical & Methodological Formulations

### 3.1 Inductive Bias Contrast: EfficientNet-B0 vs. ViT-B/16
- **EfficientNet-B0 (Localized Texture Bias):**
  - Receptive field expands progressively through stacked MBConv layers.
  - Squeeze-and-Excitation dynamically re-weights channel correlations.
  - Excellent for identifying nuclear pleomorphism, chromatin distribution, and mitotic spindle figures.
- **ViT-B/16 (Global Long-Range Dependency Bias):**
  - Receptive field is global from Layer 1 via full self-attention across 196 patch tokens.
  - Captures stromal tissue margins, glandular architectural spacing, and overall cellular distribution without rigid translational constraints.

### 3.2 Progressive 2-Phase Fine-Tuning Dynamics
$$\text{Phase 1 (Warm-up):} \quad \theta_{\text{backbone}} \leftarrow \text{frozen}, \quad \theta_{\text{head}} \leftarrow \theta_{\text{head}} - \eta_{\text{head}} \nabla_{\theta_{\text{head}}} L$$
$$\text{Phase 2 (Differential):} \quad \begin{cases} \theta_{\text{head}} \leftarrow \theta_{\text{head}} - \eta_{\text{head}} \nabla_{\theta_{\text{head}}} L \\ \theta_{\text{backbone}[-2:]} \leftarrow \theta_{\text{backbone}[-2:]} - \eta_{\text{backbone}} \nabla_{\theta_{\text{backbone}[-2:]}} L \end{cases}$$
where $\eta_{\text{head}} = 10^{-3}$ and $\eta_{\text{backbone}} = 10^{-5}$ ($100\times$ differential).

### 3.3 Soft-Voting Probability Averaging
$$P(y = 1 \mid x) = \frac{1}{M} \sum_{m=1}^M \sigma\left(z_m[1] - z_m[0]\right)$$
By averaging probabilities rather than hard voting, the ensemble preserves confidence calibrations and reduces variance on ambiguous borderline samples.

---

## 4. Published Literature Benchmark Comparison

In the literature (e.g., Jahan et al. 2025, *Neural Computing and Applications*, 37:9311–9330, DOI: 10.1007/s00521-025-10984-2), several architectures have been reported on breast histology tasks:

| Model / Benchmark Reference | Method / Setting | Reported Accuracy | Context & Protocol Note |
|---|---|---|---|
| **Jahan et al. 2025 (ViT WSI-level)** | ViT + Majority Vote | **98.19%** | Private Whole Slide Image (WSI) cohort |
| **Jahan et al. 2025 (ViT Patch-level)** | ViT-B/16 on patches | **96.74%** | Private WSI patch-level dataset |
| **Jahan et al. 2025 (CNN Ensemble)** | Multi-CNN Ensemble | **96.59%** | Private WSI patch-level dataset |
| **Jahan et al. 2025 (MobileNetV2)** | MobileNetV2 | **96.10%** | Private WSI patch-level dataset |
| **Jahan et al. 2025 (DenseNet-201)** | DenseNet-201 | **94.50%** | Private WSI patch-level dataset |
| **Literature BreakHis-200X Baseline** | Cited CNN benchmark | **96.50%** | Standard literature (frequently patient-leaking) |
| **OMNet-V2 (ViT-B/16)** | ViT-B/16 + 2-Phase Tuning | **84.29%** (Test) / **94.00%** (CV) | **Strict Patient-Disjoint Held-Out Test Set** |
| **OMNet-V2 (Ensemble)** | ViT-B/16 + EfficientNet-B0 | **84.29%** (Test) | **Highest PR-AUC ($0.9537$) & Specificity ($55.95\%$)** |

*Scientific Takeaway for Paper Discussion:* The difference between literature figures (~96%) and OMNet-V2 test accuracy (84.29%) highlights the critical effect of patient-level partitioning. Models evaluated without patient grouping achieve artificially elevated scores, whereas OMNet-V2 provides a true measure of generalization to previously unseen patient cases.

---

## 5. Model Interpretability Framework

1. **Grad-CAM (Gradient-Weighted Class Activation Mapping):**
   - Applied to the final convolutional layer of EfficientNet-B0 (`backbone.features[-1]`).
   - Highlights high-frequency morphological boundaries, confirming the model focuses on cellular atypia rather than background slide artifacts.
2. **Score-CAM:**
   - Gradient-free post-hoc visualization confirming activation maps without gradient saturation artifacts.
3. **t-SNE Embedding Analysis (256-D Latent Space):**
   - 2D projections of the 256-d bottleneck representation show distinct separation between Benign and Malignant clusters with minimal overlap.

---

## 6. Paper Drafting Section Outline

### Section I: Introduction
- Clinical context of breast cancer histopathology biopsy grading.
- Need for reliable, leak-free computer-aided diagnosis (CAD) benchmarks.
- Key contributions of OMNet-V2.

### Section II: Related Work
- CNN architectures in histopathology (ResNet, DenseNet, EfficientNet).
- Vision Transformers in digital pathology.
- Transfer learning and fine-tuning strategies for medical images.

### Section III: Methodology (OMNet-V2)
- Backbone feature extraction (EfficientNet-B0 & ViT-B/16).
- Custom 2-layer projection head design.
- 2-Phase progressive fine-tuning engine.
- Loss functions (Label-smoothed Cross-Entropy & Focal Loss).
- Soft-voting ensemble formulation.

### Section IV: Experimental Setup
- BreakHis 200X dataset composition and patient-level splitting.
- Augmentation pipeline and empirical color normalization.
- 5-Fold Stratified Cross-Validation and held-out test protocol.
- Evaluation metrics definitions.

### Section V: Results & Comparative Analysis
- 5-Fold CV results for EfficientNet-B0 vs. ViT-B/16.
- Held-out test set performance (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, MCC).
- Ensemble performance gains in specificity and PR-AUC.
- Comparison against published benchmarks and discussion of the leakage gap.

### Section VI: Interpretability & Error Analysis
- Grad-CAM and Score-CAM visual attention overlays.
- 2D t-SNE feature embedding manifolds.
- Diagnostic analysis of false positives and false negatives.

### Section VII: Conclusion & Future Directions
- Summary of findings.
- Future work: Extension to multi-magnification fusion (OMNet-V3), multi-task subtyping, and Whole Slide Image (WSI) aggregation.
