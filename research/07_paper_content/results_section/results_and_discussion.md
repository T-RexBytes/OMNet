# Results and Discussion

## 1. Experimental Setup
All experiments were performed using the public Breast Cancer Histopathological Image Database (BreakHis). To prevent data leakage, patient-disjoint partitions were cryptographically verified using `StratifiedGroupKFold` grouped on unique `patient_id`. OMNet-V2 was benchmarked on the 200X optical magnification cohort ($N=2,013$), partitioned into a 1,733-image cross-validation pool and a 280-image held-out test set. OMNet-V3 was evaluated on the entire multi-scale dataset ($N=7,909$ across 40X, 100X, 200X, and 400X) under a leak-free 5-fold Stratified Group Cross-Validation protocol.

## 2. Quantitative Comparison & Inductive Bias Analysis
On the held-out test partition of BreakHis-200X, **ViT-B/16** established the highest single-model performance, achieving **84.29% accuracy**, **97.45% sensitivity**, and **0.8998 ROC-AUC**. The high sensitivity reflects the Vision Transformer's capacity to detect malignant glandular disruption across wide receptive fields. The **Soft-Voting Ensemble** combined the global contextual awareness of ViT-B/16 with the localized morphological precision of EfficientNet-B0, achieving the **highest Specificity (55.95%)**, **highest Precision (83.63%)**, and **highest Precision-Recall AUC (0.9537)**.

In the multi-scale multi-task regime, **OMNet-V3** demonstrated robust patient-level generalization, achieving **84.56% ± 2.48% binary accuracy** and **80.32% ± 3.13% binary Macro-F1** across all four optical scales simultaneously. For fine-grained 8-class histological subtyping, OMNet-V3 achieved **44.75% ± 5.89% weighted-F1** under strict patient separation.

## 3. Scale-Conditioned Gating Behavior
Analysis of the learned gating parameter $lpha(m)$ in the Magnification-Aware Adaptive Fusion (MAF) module revealed an interpretable, monotonic transition:
- At **40X magnification**, $lpha = 0.6850 \pm 0.1301$, demonstrating higher weighting on global contextual features necessary for glandular architecture evaluation.
- At **400X magnification**, $lpha = 0.4701 \pm 0.1459$, shifting prioritization toward localized high-frequency nuclear features, pleomorphism, and chromatin texture.

## 4. Latent Space Feature Representation & Explainability
Two-dimensional t-SNE projections of the 256-dimensional bottleneck embeddings confirmed clear macro-clustering separating Benign and Malignant cohorts. Grad-CAM and Score-CAM visual activations confirmed that the models focus specifically on cellular atypia and periductal boundaries rather than slide background artifacts.

## 5. Limitations
1. **Rare Subtype Patient Scarcity:** Certain histological subtypes (e.g., Phyllodes Tumor with $N=3$ patients; Adenosis with $N=4$ patients) suffer from severe discrete patient scarcity, causing structural absence in individual test folds and depressing unweighted Macro-F1.
2. **Whole Slide Context:** Evaluations were conducted on cropped regions-of-interest (ROIs) rather than gigapixel Whole Slide Images (WSIs).
