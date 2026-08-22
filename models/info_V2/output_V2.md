# Experimental Results & Presentation Artifacts — OMNet-V2

> Comprehensive numerical performance tables, per-fold cross-validation logs, held-out test evaluations, visual figure interpretations, and publication assets from the execution of **OMNet-V2** (`result_OMNet/v2_out`).

---

## 1. Master Held-Out Test Set Performance Table ($N=280$)

Evaluated on an independent, **patient-disjoint held-out test set** ($280$ images from unseen clinical biopsy patients):

| Evaluated Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Balanced Acc | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 81.79% | 81.94% | 94.90% | 51.19% | 87.94% | 73.04% | 0.8712 | 0.9356 | 0.5391 |
| **ViT-B/16 (Best Single)** | **84.29%** | 83.04% | **97.45%** | 53.57% | **89.67%** | 75.51% | **0.8998** | 0.9488 | **0.6105** |
| **Ensemble (Soft-Voting)** | **84.29%** | **83.63%** | 96.43% | **55.95%** | 89.57% | **76.19%** | 0.8973 | **0.9537** | 0.6084 |

### Key Diagnostic Takeaways
- **Ultra-High Sensitivity on Malignancies:** ViT-B/16 achieves **97.45% Recall** (Sensitivity), detecting almost every malignant case in the test set.
- **Ensemble Synergies:** The soft-voting ensemble achieves the **highest Specificity (55.95%)**, the **highest Precision (83.63%)**, the **highest Balanced Accuracy (76.19%)**, and the **highest Precision-Recall AUC (0.9537)**.
- **Balanced Accuracy Boost:** The ensemble outperforms individual models on Balanced Accuracy ($76.19\%$ vs $75.51\%$ for ViT and $73.04\%$ for EfficientNet), compensating for the $69/31$ class imbalance.

---

## 2. 5-Fold Stratified Cross-Validation Summary ($N=1,733$)

Cross-validation performance across the 5 validation folds (mean ± standard deviation):

| Evaluated Backbone | Mean Val Accuracy | Std Val Acc | Mean Val ROC-AUC | Std Val AUC | Mean Val F1 | Std Val F1 |
|---|---|---|---|---|---|---|
| **EfficientNet-B0** | **84.77%** | ± 2.44% | **0.9754** | ± 0.0042 | **87.74%** | ± 2.28% |
| **ViT-B/16** | **94.00%** | ± 1.33% | **0.9915** | ± 0.0027 | **95.49%** | ± 1.03% |

### Per-Fold Detailed Breakdown (`cv_fold_results.csv`)

| Backbone | Fold | Best Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val ROC-AUC | Val F1 | Epoch Time |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | Fold 1 | Epoch 5 | 0.3338 | 0.4239 | 86.56% | 85.30% | 0.9687 | 88.28% | 38.6s |
| | Fold 2 | Epoch 4 | 0.3411 | 0.3441 | 84.88% | 87.90% | 0.9758 | 90.71% | 38.0s |
| | Fold 3 | Epoch 5 | 0.3055 | 0.4246 | 87.06% | 81.27% | 0.9797 | 84.49% | 38.9s |
| | Fold 4 | Epoch 5 | 0.3310 | 0.4383 | 85.25% | 83.82% | 0.9778 | 86.85% | 37.3s |
| | Fold 5 | Epoch 5 | 0.3000 | 0.3910 | 87.79% | 85.55% | 0.9751 | 88.37% | 38.0s |
| **ViT-B/16** | Fold 1 | Epoch 5 | 0.1911 | 0.3071 | 95.13% | 92.22% | 0.9905 | 94.09% | 44.9s |
| | Fold 2 | Epoch 6 | 0.1929 | 0.2566 | 95.57% | 95.68% | 0.9954 | 96.77% | 44.9s |
| | Fold 3 | Epoch 6 | 0.1903 | 0.2888 | 95.20% | 93.95% | 0.9926 | 95.46% | 43.8s |
| | Fold 4 | Epoch 6 | 0.1898 | 0.2913 | 95.49% | 93.35% | 0.9910 | 95.01% | 44.1s |
| | Fold 5 | Epoch 6 | 0.1921 | 0.2743 | 95.57% | 94.80% | 0.9881 | 96.12% | 44.1s |

---

## 3. Visual Artifact Catalog & Diagnostic Analysis

The local execution generated 10 high-resolution publication assets saved in `result_OMNet/v2_out/`:

### 3.1 Confusion Matrices (`confusion_matrix.png`)
- Compares EfficientNet-B0, ViT-B/16, and the Ensemble side-by-side on the held-out test set ($N=280$).
- **Malignant Detection:** Out of 196 malignant test patches, ViT-B/16 correctly identifies 191 ($97.45\%$), while the Ensemble identifies 189 ($96.43\%$).
- **Benign Detection:** Out of 84 benign patches, the Ensemble correctly classifies 47 ($55.95\%$), outperforming single models ($51.19\%$ for EfficientNet, $53.57\%$ for ViT).

### 3.2 ROC Curves (`roc_curve.png`)
- Compares Receiver Operating Characteristic curves across all three models on the test partition.
- ViT-B/16 achieves the highest individual AUC ($\text{AUC} = 0.8998$), followed closely by the Ensemble ($\text{AUC} = 0.8973$) and EfficientNet-B0 ($\text{AUC} = 0.8712$).

### 3.3 Precision-Recall Curves (`precision_recall.png`)
- The Ensemble achieves the highest Area Under the Precision-Recall Curve ($\text{PR-AUC} = 0.9537$), outperforming both ViT-B/16 ($0.9488$) and EfficientNet-B0 ($0.9356$).
- Demonstrates exceptional precision preservation across clinically relevant high-recall thresholds.

### 3.4 Cross-Validation Boxplot (`cv_results_boxplot.png`)
- Visual distribution of validation accuracy, ROC-AUC, and F1 across the 5 CV folds.
- Highlights the tight variance and high stability of ViT-B/16 (all 5 folds achieve $\text{ROC-AUC} \ge 0.9881$).

### 3.5 Training History Curves (`training_curve.png`)
- Plots training and validation loss and ROC-AUC across Phase 1 (warm-up, epochs 1–3) and Phase 2 (fine-tuning, epochs 4–15).
- Confirms rapid convergence during Phase 1 with further steady generalization gain in Phase 2 without overfitting.

### 3.6 t-SNE 2D Latent Space Projections (`tsne_embeddings.png`)
- 2D t-SNE visualization of the 256-dimensional feature representations.
- Shows clear macro-separation between Benign and Malignant clusters, validating that the 2-layer bottleneck projection successfully organizes the latent feature geometry.

### 3.7 Error Analysis (`error_analysis.png`)
- Visual display of high-confidence false positives (complex benign lesions with dense tubular proliferation mimicking malignancy) and false negatives (well-differentiated low-grade carcinomas with minimal nuclear pleomorphism).

### 3.8 Dataset Composition & Sample Imagery
- `dataset_composition.png`: Visual distribution bar charts illustrating benign vs. malignant ratios and subtype breakdowns.
- `sample_images.png`: Representative histological image tiles across benign and malignant categories.
- `augmentation_examples.png`: Multi-panel demonstration of random crops, flips, rotations, and color jitter transformations.

---

## 4. Benchmark Comparison Card

```
==================================================================================================
Model Framework   : OMNet-V2 (EfficientNet-B0 / ViT-B/16 Transfer Learning & Soft-Voting Ensemble)
Dataset Benchmark : BreakHis 200X (2,013 Images across 82 Patients)
Validation Scheme : 5-Fold Stratified CV (1,733 imgs) + Patient-Disjoint Test Set (280 imgs)
==================================================================================================
Model Configuration           Test Acc (%)    Sensitivity (%) Specificity (%) ROC-AUC    PR-AUC
--------------------------------------------------------------------------------------------------
EfficientNet-B0 (CNN)         81.79%          94.90%          51.19%          0.8712     0.9356
ViT-B/16 (Transformer - Best) 84.29%          97.45%          53.57%          0.8998     0.9488
Ensemble (Soft-Voting)        84.29%          96.43%          55.95%          0.8973     0.9537
--------------------------------------------------------------------------------------------------
Published Literature (Cited)  96.50%          —               —               —          —
  * Note: Literature benchmarks typically lack patient-level disjointness (data leakage gap).
==================================================================================================
```

---

## 5. Computational Execution Profile

| Execution Property | Value / Setting |
|---|---|
| **Compute Hardware** | NVIDIA CUDA GPU (Accelerated) |
| **Precision Mode** | Mixed Precision (`torch.amp.autocast`) |
| **CV Training Time per Fold** | ~38s (EfficientNet) / ~44s (ViT) |
| **Final Refit Training Time** | ~6.5 minutes total (Phase 1 + Phase 2) |
| **Peak GPU VRAM** | ~4.6 GiB (well within standard GPU memory constraints) |
