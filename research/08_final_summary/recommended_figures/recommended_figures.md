# Recommended Figure Selection for Conference Paper

| Order | Figure Name | Location | Include in Main Paper? | Primary Scientific Rationale | Key Takeaway Message |
|---|---|---|---|---|---|
| **1** | Architecture Schematic | `05_figures/architecture/v3_architecture.png` (or markdown diagram) | **Yes (Tier A)** | Explains the dual-branch hybrid design and MAF module | End-to-end multi-scale fusion with hierarchical loss |
| **2** | Adaptive Fusion Gate Alpha | `05_figures/performance_comparison/v3_adaptive_fusion_gate_alpha.png` | **Yes (Tier A)** | Provides empirical proof of scale-conditioned gating | Monotonic shift from global ViT ($40\text{X}$) to local CNN ($400\text{X}$) |
| **3** | Binary & Subtype ROC Curves | `05_figures/roc_pr/v3_binary_roc_curve.png`, `v3_subtype_ovr_roc_curves.png` | **Yes (Tier A)** | Demonstrates discrimination across thresholds | High binary AUC ($>0.88$) and distinct subtype separation |
| **4** | Confusion Matrices | `05_figures/confusion_matrix/v3_binary_confusion_matrix.png`, `v3_subtype_confusion_matrix.png` | **Yes (Tier A)** | Details sensitivity, specificity, and misclassification pairs | High sensitivity for DC/F; explains LC vs DC ambiguity |
| **5** | t-SNE Latent Space Projections | `05_figures/embeddings/v3_tsne_embeddings.png` | **Tier B (Optional / Poster)** | Demonstrates feature manifold organization | Clear topological separation between benign and malignant |
| **6** | V2 Training & PR Curves | `05_figures/training_curves/v2_training_curves.png`, `v2_precision_recall_curves.png` | **Tier C (Supplementary)** | Documents 2-phase fine-tuning convergence | Rapid convergence without overfitting divergence |
