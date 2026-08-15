# Experiment 1 Plan: DenseNet-201 CNN Baseline (BreakHis)

## 0. Role of This Document

This is a build spec for a single Google Colab notebook (`.ipynb`). It is meant to be handed to an
implementation agent with no other context. Follow it in order. Do not add architecture complexity,
alternate models, or shortcuts not listed here. This is Experiment 1 of 3 in a larger CNN-ViT
feature-fusion research project. Experiment 2 (ViT-B/16) uses an identical dataset treatment and
evaluation protocol — do not diverge from what's specified here, or the two baselines won't be
comparable.

## 1. Objective

Fine-tune a pretrained DenseNet-201 on BreakHis to produce:
1. A primary classification baseline (benign vs malignant).
2. A histopathological subtype classification baseline.
3. A saved, reusable feature extractor (penultimate-layer embeddings) for later use in Experiment 3
   (CNN+ViT fusion).

This notebook's output is a baseline result, not the final research contribution. Keep it clean and
reproducible — Experiment 3 depends on this checkpoint.

## 2. Dataset Scope (Locked Decisions — Do Not Change)

- **Dataset:** BreakHis (Breast Cancer Histopathological Database).
- **Magnification:** Use **200X only** for the core pipeline. Do not pool magnifications and do not
  substitute a different magnification without explicit instruction. (Reason: controls for a
  confound not yet relevant to this baseline; multi-magnification is a later ablation, not part of
  the core three experiments.)
- **Storage/access:** Dataset lives in a **private Kaggle dataset**. Fetch it into Colab via the
  Kaggle API (`kaggle.json` credentials uploaded to Colab session or stored in Colab secrets — do
  not hardcode API keys in the notebook). Do not manually upload the full dataset each session.
- **Tasks (both required, not optional):**
  - Task A: benign vs malignant (primary classification).
  - Task B: histopathological subtype (8-class: adenosis, fibroadenoma, phyllodes tumor, tubular
    adenoma [benign]; ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma
    [malignant]).
- **Model structure for two tasks:** Single shared DenseNet-201 backbone, **two classification
  heads** (multi-task, not two separate models). This is a locked architectural decision for
  Colab efficiency and consistency with the fusion model's design in Experiment 3. Do not
  reintroduce this as an open question.

## 3. Mandatory Pre-Training Data Audit (Do This Before Any Model Code)

Do not skip or reorder this section. Produce a short `data_audit.md` (or notebook section with
printed output saved) covering:

1. **Patient/case ID extraction.** BreakHis filenames encode patient/case info
   (e.g. `SOB_B_A-14-22549AB-40-001.png`). Parse out a stable patient/case identifier for every
   image in the 200X subset.
2. **Patient-level split, not image-level.** All images from a single patient/case must land in
   only one of train/val/test. Verify programmatically after splitting (assert no ID overlap
   across the three sets) — this is not optional, it is a correctness requirement for a
   publishable result.
3. **Split ratios.** Use a stratified-by-class (at patient level, best effort) split — recommended
   70/15/15 or 80/10/10 train/val/test. Document the exact counts (patients and images) per split,
   per class, per subtype.
4. **Class distribution report.** Print and save counts for both Task A (benign/malignant) and
   Task B (8 subtypes) within the 200X subset. This determines the loss/sampling strategy in
   Section 6 below — do not pick a class-imbalance method before this step.
5. **Directory structure verification.** Confirm the downloaded Kaggle dataset structure matches
   expectations (folder layout, file counts) before writing the Dataset/DataLoader class — fail
   loudly if it doesn't.
6. **Save the split** (e.g. as a CSV with columns: filepath, patient_id, magnification, primary_label,
   subtype_label, split) so Experiment 2 and Experiment 3 can load the exact same split. This file
   is the single source of truth for all three experiments — treat it as a fixed artifact.

## 4. Data Pipeline

- Kaggle API → download private dataset into Colab (`kaggle datasets download ...`), unzip to a
  working directory.
- Build the split CSV from Section 3.
- PyTorch `Dataset`/`DataLoader` (assume PyTorch unless the team has a stated TensorFlow preference
  — flag this assumption to the user if uncertain).
- Preprocessing:
  - Resize to 224×224 (DenseNet-201 ImageNet-pretrained input size).
  - Normalize using ImageNet mean/std (since backbone is ImageNet-pretrained).
- Augmentation (train split only):
  - Random horizontal/vertical flip (histopathology images have no canonical orientation — this is
    standard and safe for this domain).
  - Random rotation (small angle, e.g. ±20°).
  - Mild color jitter (brightness/contrast) — keep conservative, stain color carries diagnostic
    signal, don't distort it aggressively.
  - Do not apply heavy augmentation until the class-distribution audit shows it's needed for the
    minority classes.

## 5. Model Architecture

- Backbone: `torchvision.models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)`.
- Remove/replace the original 1000-class ImageNet classifier.
- Add two heads on top of the pooled feature vector (1920-dim for DenseNet-201):
  - Head A: Linear → 2 classes (benign/malignant).
  - Head B: Linear → 8 classes (subtype). Optionally one hidden layer (e.g. 1920→512→8) if
    validation performance suggests the linear head underfits — start simple first.
- Keep a hook or named intermediate to extract the 1920-dim pooled embedding later for Experiment 3
  (feature fusion). Document the exact layer name used.

## 6. Training Strategy

- **Phase 1 (head-only warmup):** Freeze the backbone entirely. Train only the two heads for a few
  epochs (e.g. 3–5) to stabilize before unfreezing — cheap on Colab and prevents early destructive
  gradients into the pretrained backbone.
- **Phase 2 (gradual unfreeze):** Unfreeze the last DenseNet block(s) first, then progressively
  more if val performance and compute budget allow. Use a lower learning rate for backbone layers
  than for the heads (discriminative learning rates).
- **Loss:** Sum or weighted sum of Task A loss + Task B loss (e.g. `total_loss = loss_A + λ * loss_B`,
  start with λ=1, tune later — record whatever value is used).
  - Once Section 3's class distribution is known: if imbalance is significant, use weighted
    cross-entropy (class weights inverse to frequency) or focal loss for the affected task(s).
    Do not assume which method wins before comparing — but pick one as the working default per
    the audit and document the choice.
- **Optimizer:** AdamW, weight decay ~1e-4 (tune later, record whatever value is used).
- **LR schedule:** Cosine annealing or ReduceLROnPlateau on validation macro-F1 (subtype task).
- **Batch size:** As large as fits in Colab GPU memory at 224×224 for DenseNet-201 — typically
  16–32 depending on the assigned GPU (T4/A100). Use mixed precision (`torch.cuda.amp`) to increase
  headroom.
- **Early stopping:** On validation macro-F1 for Task B (subtype), since that's the harder/more
  imbalance-sensitive task. Patience ~5–10 epochs.
- **Checkpointing:** Save best checkpoint by validation macro-F1. Also save the final embedding
  extraction function/module cleanly — Experiment 3 will import or replicate this.
- **Reproducibility:** Fix random seeds (Python, NumPy, PyTorch, CUDA). Log the seed used.
- **Test set:** Touch only once, at the very end, after all model selection is done on val.

## 7. Colab-Specific Practicalities

- Mount Google Drive (or use Colab's ephemeral disk + re-download from Kaggle each session) to
  persist checkpoints — do not lose a trained checkpoint to session timeout.
- Save checkpoints periodically (e.g. every N epochs) in addition to "best" checkpoint, in case of
  disconnection.
- Log GPU memory usage once at model-build time to catch OOM risk before a full training run.
- Use mixed-precision training by default.

## 8. Evaluation Protocol (Must Match Experiment 2 Exactly)

Run once on the held-out test set after model selection is complete.

**Task A (primary) metrics:**
- Accuracy, Precision, Recall, F1, confusion matrix.

**Task B (subtype) metrics:**
- Accuracy, Macro Precision, Macro Recall, Macro F1, per-class precision/recall/F1, confusion matrix.

Save all metrics and confusion matrices to disk (CSV/JSON + plotted images) — Experiment 3's paper
section will directly compare against these numbers.

## 9. Deliverables Checklist

- [ ] `data_audit.md` / audit output (patient counts, class counts, split verification)
- [ ] `split.csv` (shared source of truth for all three experiments)
- [ ] Trained DenseNet-201 notebook (`.ipynb`)
- [ ] Best checkpoint file (saved to Drive)
- [ ] Test-set metrics (Task A + Task B) saved to disk
- [ ] Documented embedding-extraction method (for Experiment 3 reuse)
- [ ] Recorded hyperparameters/config (seed, LR, batch size, λ, optimizer, epochs run)

## 10. Explicit Guardrails

- Do not use magnifications other than 200X without explicit instruction.
- Do not split by image — patient-level split only, verified.
- Do not skip the class-distribution audit before choosing a loss strategy.
- Do not touch the test set until final evaluation.
- Do not change the two-head multi-task structure to two separate models without instruction —
  Experiment 2 and 3 assume this structure for comparability.
- Do not add attention/complex fusion logic here — this is a standalone CNN baseline only.
