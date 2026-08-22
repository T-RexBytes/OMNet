# OMNet-V2 Summary Card

- **Model Identity:** OMNet-V2 (Transfer-Learning & Soft-Voting Ensemble Benchmark)
- **Task Modality:** Binary Histopathology Malignancy Classification (Benign vs. Malignant)
- **Dataset Domain:** BreakHis 200X Optical Magnification subset (2,013 images, 82 patients)
- **Backbone Architectures:**
  1. `efficientnet_b0` (ImageNet1K_V1 pretrained, 1280-d feature map)
  2. `vit_b_16` (ImageNet1K_V1 pretrained, 768-d [CLS] token)
- **Classification Head:** Linear(D_in, 256) -> LayerNorm(256) -> GELU() -> Dropout(0.4) -> Linear(256, 2)
- **Optimization Strategy:** 2-Phase Progressive Fine-Tuning:
  - Phase 1: Frozen backbone warmup (3 epochs, LR = 1e-3)
  - Phase 2: Top N=2 layers unfreezing + Cosine Annealing (12 epochs, LR_head = 1e-3, LR_backbone = 1e-5)
- **Ensemble Policy:** Soft-voting probability average: p_ensemble = 0.5 * (p_effnet + p_vit)
- **Primary Achievements (Held-out Test N=280):**
  - ViT-B/16: 84.29% Accuracy, 97.45% Sensitivity, 0.8998 ROC-AUC
  - Ensemble: 84.29% Accuracy, 96.43% Sensitivity, 55.95% Specificity, 0.9537 PR-AUC
