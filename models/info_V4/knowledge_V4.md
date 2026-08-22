# Knowledge V4 — OMNet Research Paper Reference Document

> All information relevant to writing a research paper based on OMNet V4.

---

## 1. Research Question

> **Core Research Question:**
> On the public BreakHis dataset, does a shared-backbone architecture combining an ImageNet-pretrained CNN encoder with an attention-enhanced Fourier-KAN residual block and a weight-tied, fixed-iteration refinement stage improve 8-class subtype macro-F1 beyond from-scratch and pretrained single-task baselines, and does an auxiliary binary detection head improve that result further — all under a patient-level, leakage-safe evaluation protocol, with a frozen final test set evaluated exactly once?

---

## 2. Theoretical Motivation & Novelty

### 2.1 Motivation
- Standard CNN classifiers trained on BreakHis treat binary (benign/malignant) and multi-class subtype classification as separate problems
- Fourier-KAN (FKAN) layers offer richer function approximation than standard MLPs via learnable Fourier basis expansions — validated in Ali et al. 2026 for binary classification only
- Deep Equilibrium Models (DEQ) enable implicit-depth networks but introduce solver convergence risks; weight-tied unrolled refinement offers the depth benefit without instability
- Attention mechanisms (CBAM family) improve CNN feature selectivity; LCBAM extends this with depthwise spatial convolutions
- Multi-task learning with an auxiliary binary head regularizes the shared representation and may improve subtype performance

### 2.2 Claimed Contributions
1. **First multi-class (8-class) validation** of the FKAN+attention+unrolled-refinement combination on BreakHis
2. **Weight-tied fixed-iteration refinement** (N=6 iterations, shared FKAN) as a computationally stable alternative to full DEQ solving
3. **Strict patient-disjoint leakage-safe protocol** with automated leakage verification gate
4. **Dual-head multi-task architecture** with provably non-leaking evaluation

---

## 3. Prior Work & Literature Context

### 3.1 BreakHis Classification Literature
- Most published results use patient-overlapping random splits — inflating reported accuracy
- Published binary accuracy benchmark: ~96.60% (non-disjoint protocol, DenseNet-201 + FKAN + attention)
- Few papers address 8-class subtype macro-F1 under strict patient-grouped splits
- Ali et al. 2026: validated FKAN + attention + DEQ-style refinement for binary classification on BreakHis

### 3.2 Key Architectural References
| Technique | Original Paper / Source |
|---|---|
| DenseNet | Huang et al., CVPR 2017 — "Densely Connected Convolutional Networks" |
| CBAM | Woo et al., ECCV 2018 — "CBAM: Convolutional Block Attention Module" |
| KAN / Fourier-KAN | Liu et al. 2024 — "KAN: Kolmogorov-Arnold Networks"; Xu et al. for Fourier extension |
| DEQ / Implicit Networks | Bai et al., NeurIPS 2019 — "Deep Equilibrium Models" |
| Multi-task Learning | Caruana 1997; Ruder 2017 survey |
| BreakHis Dataset | Spanhol et al., IEEE TBME 2016 |

### 3.3 Evaluation Protocol Critique
Published BreakHis papers frequently report:
- Image-level accuracy (not patient-level)
- Random splits without patient grouping
- No cross-patient leakage verification

This work uses:
- Patient-grouped StratifiedGroupKFold
- MD5-verified deduplication
- Frozen test set evaluated exactly once

---

## 4. Experimental Design

### 4.1 Ablation Hierarchy
The ablation is designed to attribute performance gain to each architectural component:

| Stage | Change from previous | Scientific Claim |
|---|---|---|
| B1 | From-scratch DenseNet | Floor baseline |
| B2 / A0 | ImageNet pretraining | Effect of transfer learning |
| A1 | + Fourier-KAN (1 iter, no attention) | Effect of FKAN alone |
| A2 | + LCBAM attention | Effect of attention module |
| A3 | + Weight-tied refinement (6 iters, subtype-only) | Effect of iterative refinement |
| A4 | + Auxiliary detection head | Effect of multi-task learning |

### 4.2 Statistical Testing
- **Test:** Wilcoxon signed-rank test (paired, fold-level F1 scores)
- **Alternative:** Two-sided
- **Multiple comparisons correction:** Holm-Bonferroni (via `stats.false_discovery_control`)
- **Significance threshold:** alpha = 0.05
- **Comparisons:** All adjacent ablation pairs + B1/B2 vs A4

### 4.3 Multi-Seed Robustness
Proposed model A4 trained with 3 independent seeds: [42, 7, 123]
Each seed: full 5-fold CV
Purpose: Confirm result is not seed-sensitive

### 4.4 Final Evaluation Protocol
- Model selection: best CV configuration
- Retrain on full 85% dev pool (90% train, 10% val for early stopping)
- Evaluate once on frozen 15% test set
- All metrics recorded at single evaluation moment

---

## 5. Evaluation Metrics

### Subtype Head (8-class)
| Metric | Formula / Notes |
|---|---|
| Accuracy | Fraction correctly classified |
| Balanced Accuracy | Mean per-class recall |
| Macro-F1 | Unweighted mean F1 across 8 classes — **primary metric** |
| Weighted-F1 | Class-frequency-weighted F1 |
| Per-class F1, Precision, Recall | Reported per subtype |
| OvR ROC-AUC | One-vs-Rest, per class + macro |
| Confusion Matrix | Normalized (true), 8x8 |

### Detection Head (binary)
| Metric | Notes |
|---|---|
| Accuracy | Binary |
| Balanced Accuracy | |
| Precision / Recall / F1 | Binary |
| Specificity | TN / (TN + FP) |
| ROC-AUC | Binary |
| Confusion Matrix | 2x2 |

### Why Macro-F1 as Primary Metric
- Treats all 8 subtypes equally regardless of class frequency
- Robust to class imbalance (especially sparse subtypes like papillary_carcinoma)
- More meaningful than accuracy for skewed distributions

---

## 6. Explainability

### LCBAM Spatial Attention Maps (Figure J)
- The `last_spatial_attention` tensor is cached during the forward pass
- Overlaid as a jet colormap heatmap on original images
- Visualized for 8 random test samples
- Demonstrates which tissue regions the model focuses on per subtype

### Failure Analysis (Figure K)
- High-confidence misclassifications on test set are visualized
- Low-confidence predictions (max prob < 50%) flagged in `low_confidence_review.csv`
- Supports qualitative error analysis section of the paper

---

## 7. Publication Figures

| Figure | Description |
|---|---|
| Fig. A | Morphological diversity across 8 subtypes x 4 magnifications (sample grid) |
| Fig. B | Class imbalance: image count + unique patient count per subtype |
| Fig. C | Data pipeline schematic: Raw → QC → Split → Augment → Normalize |
| Fig. D | Patient-grouped split verification: dev/test separation audit |
| Fig. E | Model performance comparison: B1 vs B2 vs A4 (CV Macro-F1 bar chart) |
| Fig. F | Test-set confusion matrices: 8-class subtype + 2-class detection |
| Fig. G | OvR ROC curves + Precision-Recall curves per subtype |
| Fig. H | Learning curves: train/val loss + val macro-F1 vs epoch (final model) |
| Fig. I | Stepwise ablation progression: A0 → A1 → A2 → A3 → A4 |
| Fig. J | LCBAM spatial attention overlays on 8 test samples |
| Fig. K | High-confidence misclassification diagnostic review |

All figures: 300 DPI, saved as PDF + PNG.

---

## 8. Reported Results Structure (for paper tables)

### Table 1: Baseline Comparison
| Model | CV Mean Macro-F1 | CV Std | Test Macro-F1 |
|---|---|---|---|
| B1: From-Scratch DenseNet | TBD | TBD | — |
| B2: ImageNet Pretrained DenseNet | TBD | TBD | — |
| **A4: Proposed (Full Model)** | **TBD** | **TBD** | **TBD** |

### Table 2: Ablation Study
| Stage | Added Component | CV Macro-F1 | Delta vs prev |
|---|---|---|---|
| A0 (=B2) | Pretrained Floor | TBD | — |
| A1 | + Fourier-KAN | TBD | +TBD% |
| A2 | + LCBAM Attention | TBD | +TBD% |
| A3 | + Weight-Tied Refinement | TBD | +TBD% |
| A4 | + Aux Detection Head | TBD | +TBD% |

### Table 3: Statistical Significance
| Comparison | Delta Macro-F1 | p-value (raw) | p-value (corrected) | Significant? |
|---|---|---|---|---|
| B2 vs A4 | TBD | TBD | TBD | TBD |

---

## 9. Reproducibility Information

| Element | Value |
|---|---|
| Framework | PyTorch (ROCm/HIP or CUDA compatible) |
| Python version | 3.10.x |
| Key libraries | torch, timm, torchvision, sklearn, scipy, imagehash |
| Dataset | BreakHis (Kaggle slug: `ambarish/breakhis`) |
| Random seed | 42 (primary), 7 and 123 for robustness |
| Deterministic mode | True (non-ROCm only) |
| Config file | Saved to `config.json` in each run directory |
| Environment record | Saved to `environment.json` in each run directory |

---

## 10. Limitations & Future Work

### Limitations
- Single-dataset evaluation (BreakHis only in this pipeline version)
- Binary detection head evaluated on training domain only (no external generalization in this variant)
- Weight-tied refinement is a fixed-iteration approximation — may not reach true DEQ fixed point
- FP32 training required on RDNA 4 hardware; BF16/FP16 instability may not affect other hardware
- Magnification-pooled evaluation — magnification as a covariate is not controlled in main experiments

### Future Work
- Cross-magnification analysis (train/test on specific magnifications)
- External dataset validation (IDC, TCGA)
- Replacing global average pooling with region-of-interest pooling guided by LCBAM attention
- Exploration of full DEQ convergence with gradient-free fixed-point solvers on stable hardware
- Hyperparameter search for N_REFINEMENT_ITERS and grid_size

---

## 11. Key Terminology

| Term | Definition |
|---|---|
| FKAN | Fourier Kolmogorov-Arnold Network layer |
| LCBAM | Lightweight Convolutional Block Attention Module |
| DEQ | Deep Equilibrium Model |
| Weight-tied refinement | Shared weights reused across N unrolled iterations |
| Macro-F1 | Unweighted mean of per-class F1 scores |
| Patient-disjoint split | No patient appears in both train and test/val |
| OvR ROC | One-vs-Rest ROC curve for multi-class classification |
| Leakage | Data contamination where test information influences training |
| RDNA 4 / gfx1200 | AMD GPU architecture (RX 9060 XT) |
| ROCm | AMD's GPU compute platform (Radeon Open Compute) |
