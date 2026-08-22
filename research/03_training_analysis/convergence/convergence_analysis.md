# Training, Convergence, and Stability Analysis

## 1. OMNet-V2 Training Dynamics (2-Phase Progressive Fine-Tuning)
OMNet-V2 was trained using a staged protocol on both EfficientNet-B0 and ViT-B/16 backbones:
- **Phase 1 (Epochs 1–3):** Classification head warmup ($\eta_{	ext{head}} = 10^{-3}$, backbone frozen).
  - EfficientNet-B0 rapidly converges from initial loss $0.62$ down to $0.34$.
  - ViT-B/16 achieves near-instantaneous alignment, reaching train accuracy $>94\%$ within 3 warmup epochs.
- **Phase 2 (Epochs 4–15):** Top $N=2$ blocks fine-tuning ($\eta_{	ext{head}} = 10^{-3}$, $\eta_{	ext{backbone}} = 10^{-5}$, Cosine Annealing).
  - Validation ROC-AUC steadily improves without overfitting divergence.
  - Early stopping triggers around epoch 10–12 when validation AUC reaches optimal plateau ($0.9758$ for EfficientNet, $0.9954$ for ViT).

## 2. OMNet-V3 Training Dynamics (End-to-End Joint Multi-Task Optimization)
OMNet-V3 trains all dual-branch components end-to-end with the hierarchical loss:
- **Warmup Epochs (1–5):** Dual backbones fine-tuned at base LR $10^{-5}$, MAF and heads at head LR $10^{-4}$.
- **Full Unfreezing (Epochs 6–30):** Loss smoothly descends from $1.15$ to $pprox 0.22$.
- **Numerical Safety:** Executed in full FP32 precision, guaranteeing finite gradient norms and zero non-finite tensors throughout all 5 folds.
