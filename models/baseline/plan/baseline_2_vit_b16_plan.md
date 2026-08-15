# Experiment 2 Plan: ViT-B/16 Vision Transformer Baseline (BreakHis)

## 0. Role of This Document

This is a build spec for a single Google Colab notebook (`.ipynb`). It is meant to be handed to an
implementation agent with no other context. Follow it in order. This is Experiment 2 of 3 in a
larger CNN-ViT feature-fusion research project. It must use the **exact same dataset, split, and
evaluation protocol** as Experiment 1 (DenseNet-201) — see `baseline_1_densenet201_plan.md` for the
canonical `split.csv`. Do not regenerate the split independently; load the one Experiment 1 produced.

## 1. Objective

Fine-tune a pretrained ViT-B/16 on BreakHis to produce:
1. A primary classification baseline (benign vs malignant).
2. A histopathological subtype classification baseline.
3. A saved, reusable feature extractor ([CLS] token embedding) for later use in Experiment 3
   (CNN+ViT fusion).

This is a baseline result for comparison against DenseNet-201, not the final contribution. Results
must be directly comparable to Experiment 1 — same split, same metrics, same protocol.

## 2. Dataset Scope (Locked Decisions — Must Match Experiment 1 Exactly)

- **Dataset:** BreakHis, **200X magnification only** — identical subset to Experiment 1.
- **Split:** Load the `split.csv` produced in Experiment 1 (patient-level split, verified no
  patient overlap across train/val/test). **Do not re-split the data.** If `split.csv` is not
  available, stop and request it rather than generating a new split — a mismatched split silently
  invalidates the ablation comparison in Experiment 3.
- **Storage/access:** Same private Kaggle dataset, fetched via Kaggle API into Colab. Do not
  hardcode credentials in the notebook.
- **Tasks (both required):**
  - Task A: benign vs malignant.
  - Task B: histopathological subtype (8-class, same class list as Experiment 1).
- **Model structure for two tasks:** Single shared ViT-B/16 backbone, **two classification heads**
  (multi-task), matching Experiment 1's structure for architectural symmetry going into the fusion
  model.

## 3. Data Audit — Reuse, Don't Repeat

The full data audit (patient IDs, class distribution, leakage check) was already done in
Experiment 1. Import `split.csv` and the class-distribution numbers directly rather than
recomputing from scratch — but do run a quick sanity check (row count, class counts) to confirm
the loaded file matches expectations before training.

## 4. Data Pipeline

- Kaggle API → download the same private dataset (identical download step to Experiment 1).
- Load `split.csv` to assign each image to train/val/test.
- Preprocessing (ViT-B/16 specific):
  - Resize to **224×224** (standard ViT-B/16 input; confirm against the specific pretrained
    checkpoint used — some ViT-B/16 checkpoints expect 224, verify before assuming).
  - Normalize using the mean/std the chosen pretrained checkpoint was trained with (ImageNet
    stats for standard torchvision/timm ViT-B/16 ImageNet1K weights — do not reuse DenseNet's
    normalization blindly without checking they match).
  - Patch size 16×16 (fixed by the "B/16" architecture — no decision needed here).
- Augmentation (train split only) — same policy as Experiment 1 for fair comparison:
  - Random horizontal/vertical flip.
  - Small-angle random rotation (±20°).
  - Mild color jitter (conservative — preserve stain-color diagnostic signal).
  - Keep augmentation policy identical to Experiment 1 unless there's a specific reason ViT needs
    different treatment (there usually isn't at this dataset scale) — note any deviation explicitly
    if introduced.

## 5. Model Architecture

- Backbone: pretrained ViT-B/16, e.g. `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`
  or `timm.create_model('vit_base_patch16_224', pretrained=True)` — pick one library and document
  the exact checkpoint/version used (affects reproducibility and normalization stats above).
- Remove/replace the original classification head.
- Use the [CLS] token's final hidden representation (768-dim for ViT-B/16) as the pooled feature.
- Add two heads on top of the [CLS] embedding:
  - Head A: Linear → 2 classes (benign/malignant).
  - Head B: Linear → 8 classes (subtype). Optional single hidden layer if underfitting, same as
    Experiment 1 policy — start simple.
- Keep a documented, clean way to extract the 768-dim [CLS] embedding for reuse in Experiment 3.

## 6. Training Strategy

- **Phase 1 (head-only warmup):** Freeze the entire ViT backbone. Train only the two heads for a
  few epochs (e.g. 3–5).
- **Phase 2 (gradual unfreeze):** ViT fine-tuning is more prone to instability/overfitting on
  smaller medical datasets than CNNs — unfreeze conservatively. Recommended: unfreeze only the
  last 2–4 transformer blocks initially, not the whole backbone, and monitor validation loss
  closely for divergence. Full unfreezing is optional and should only be attempted if val
  performance clearly plateaus with partial unfreezing.
- **Loss:** Same structure as Experiment 1 — `total_loss = loss_A + λ * loss_B`. Use the same λ
  value used in Experiment 1 for comparability unless there's a documented reason to change it.
- **Class imbalance handling:** Use the same strategy decided in Experiment 1's audit (weighted
  CE / focal loss / etc.) — do not pick a different imbalance strategy for ViT than for DenseNet;
  that would confound the baseline comparison.
- **Optimizer:** AdamW is standard for ViT fine-tuning; weight decay ~0.01–0.05 is typical for
  transformers (higher than the CNN's typical 1e-4 — this is a legitimate architecture-specific
  difference, not an unfair-comparison violation, but document it).
- **LR schedule:** Cosine annealing with warmup (ViTs are more sensitive to LR warmup than CNNs —
  include a short linear warmup, e.g. first 5% of steps, before cosine decay). Lower peak LR than
  DenseNet's — ViT fine-tuning typically wants smaller LRs (e.g. 1e-5 to 5e-5 for backbone, higher
  for the new heads).
- **Batch size:** ViT-B/16 at 224×224 is more memory-hungry per-image than DenseNet-201 in
  practice — expect to use a smaller batch size than Experiment 1 on the same Colab GPU, or use
  gradient accumulation to reach an equivalent effective batch size. Use mixed precision
  (`torch.cuda.amp`).
- **Early stopping:** Same criterion as Experiment 1 — validation macro-F1 on Task B, patience
  ~5–10 epochs.
- **Checkpointing:** Save best checkpoint by validation macro-F1, plus periodic checkpoints against
  Colab disconnection.
- **Reproducibility:** Fix and log all random seeds.
- **Test set:** Touch only once, at the end.

## 7. Colab-Specific Practicalities

- Mount Drive for checkpoint persistence.
- ViT fine-tuning can be more memory-intensive than DenseNet at the same batch size — check GPU
  memory usage early (small dry-run batch) before committing to a full training run; reduce batch
  size or add gradient accumulation if needed.
- Use mixed-precision training by default.
- If Colab assigns a smaller GPU (e.g. T4) and OOM occurs even at batch size 8, gradient
  accumulation over 2–4 steps is preferred over reducing image resolution (resolution changes are
  architecturally significant for a patch-based model — avoid changing patch/image size to work
  around memory issues).

## 8. Evaluation Protocol (Must Match Experiment 1 Exactly)

Run once on the held-out test set (same test split as Experiment 1) after model selection.

**Task A (primary) metrics:**
- Accuracy, Precision, Recall, F1, confusion matrix.

**Task B (subtype) metrics:**
- Accuracy, Macro Precision, Macro Recall, Macro F1, per-class precision/recall/F1, confusion matrix.

Save all metrics using the same format/file structure as Experiment 1 so results can be directly
tabulated side-by-side in the ablation study (Section 17 of the master plan).

## 9. Deliverables Checklist

- [ ] Confirmed reuse of Experiment 1's `split.csv` (no re-splitting)
- [ ] Documented ViT checkpoint source/library/version used
- [ ] Trained ViT-B/16 notebook (`.ipynb`)
- [ ] Best checkpoint file (saved to Drive)
- [ ] Test-set metrics (Task A + Task B), same format as Experiment 1
- [ ] Documented embedding-extraction method ([CLS] token, for Experiment 3 reuse)
- [ ] Recorded hyperparameters/config (seed, LR, warmup schedule, batch size/accumulation, λ,
      optimizer, epochs run)

## 10. Explicit Guardrails

- Do not re-split the dataset — reuse Experiment 1's `split.csv` exactly.
- Do not change the class-imbalance strategy from what Experiment 1 established.
- Do not change the two-head multi-task structure — must mirror Experiment 1 for fusion
  compatibility in Experiment 3.
- Do not touch the test set until final evaluation.
- Do not add attention/complex fusion logic here — this is a standalone ViT baseline only.
- If normalization stats, checkpoint source, or input resolution differ from assumptions above,
  document the deviation explicitly rather than silently changing it — downstream fusion work
  needs to know exactly how each backbone was prepared.
