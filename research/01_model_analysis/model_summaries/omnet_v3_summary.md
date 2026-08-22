# OMNet-V3 Summary Card

- **Model Identity:** OMNet-V3 (Magnification-Aware Multi-Task Adaptive Fusion Framework)
- **Task Modality:** Multi-Task Joint Binary Detection (2 classes) + Histological Subtyping (8 classes)
- **Dataset Domain:** Full Multi-Scale BreakHis (40X, 100X, 200X, 400X; 7,909 images, 82 patients)
- **Dual-Branch Backbones:**
  1. EfficientNet-B0 (1280-d local morphology extractor) -> Projector -> f_cnn (256-d)
  2. ViT-Tiny/16 (192-d global context self-attention) -> Projector -> f_vit (256-d)
  3. Scale Embedding: nn.Embedding(4, 64) -> e_m (64-d)
- **Fusion Layer:** Magnification-Aware Adaptive Fusion (MAF):
  - Dynamic Gate: alpha = Sigmoid(Linear(576, 1)) in (0, 1)
  - Blending: f_fused = alpha * f_cnn + (1 - alpha) * f_vit
  - Scale-conditioned Residual MLP: f_out = f_fused + MLP([f_fused, e_m]) (256-d)
- **Loss Formulation:** Hierarchical composite loss: L_total = 0.3 * L_bin + 0.6 * L_sub + 0.1 * L_consistency
- **Primary Achievements (5-Fold Stratified Group CV):**
  - Binary Accuracy: 84.56% +- 2.48% (range: 81.54% - 87.99%)
  - Binary Macro-F1: 80.32% +- 3.13% (range: 76.40% - 84.60%)
  - Subtype Weighted-F1: 44.75% +- 5.89% (range: 37.64% - 54.50%)
  - Scale Gating: alpha transitions monotonically: 40X (0.685) -> 100X (0.599) -> 200X (0.526) -> 400X (0.470)
