# Architectural Evolution: OMNet-V2 to OMNet-V3

| Design Dimension | OMNet-V2 (Predecessor) | OMNet-V3 (Proposed Flagship) | Scientific Rationale for Evolution |
|---|---|---|---|
| **Scope of Scale** | Single Scale (200X only, 2,013 images) | Full Multi-Scale (40X, 100X, 200X, 400X, 7,909 images) | Clinical pathologists review biopsies across multiple zoom levels; V3 eliminates single-scale isolation. |
| **Branch Integration** | Late Post-Hoc Soft-Voting Ensemble (Separate models) | End-to-End Early/Mid-Level Feature Fusion (Single model) | Joint end-to-end backpropagation trains CNN and ViT encoders to be mutually complementary. |
| **Scale Adaptation** | None (Scale-agnostic) | Learnable Magnification Embedding ($e_m \in \mathbb{R}^{64}$) + Gating $lpha(m)$ | Dynamically adjusts reliance on global architecture (40X) vs nuclear atypia (400X). |
| **Task Formulation** | Single-Task Binary Classification | Multi-Task: Binary Malignancy + 8-Class Subtyping | Solves fine-grained subtyping while regularizing the shared feature space. |
| **Loss Function** | Standard Cross-Entropy / Focal Loss | Hierarchical Loss with KL-Divergence Consistency | Enforces structural probability agreement ($p(	ext{malignant}) = \sum p(	ext{malignant subtypes})$). |
| **ViT Backbone Size** | ViT-B/16 (86.6M parameters) | ViT-Tiny/16 (5.7M parameters) | Drastically reduces compute/VRAM footprint while achieving joint multi-scale fusion. |
