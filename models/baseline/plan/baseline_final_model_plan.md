# Final Model Plan: Gated DenseNet-201 + ViT-B/16 Fusion with Class-Balanced Subtype Loss

## 1. Final Model Decision

The final model is:

**DenseNet-201 + ViT-B/16 → feature-wise learnable gated fusion → multi-task heads (primary +
subtype), with class-balanced loss applied to the subtype task only.**

This is a single, committed architecture. It will be trained once, under the existing 5-fold
patient-level protocol for Task B and the existing single held-out split for Task A, exactly as
Baselines 1–3 were. There is no subsequent architecture, no conditional second phase, and no
follow-up experiment planned after this. Both the gating mechanism and the class-balanced loss are
included in this one design — not staged as separate trials — because the evidence for each was
already established by prior completed work, and this project has budget for one training
campaign, not a sequence of them.

## 2. Research Motivation

Three completed experiments produced a specific, non-ambiguous diagnosis:

- DenseNet-201 (0.2938 ± 0.0343) and ViT-B/16 (0.2688 ± 0.0540) each plateau on subtype
  classification, with especially poor minority-class recall (phyllodes tumor near 0%, mucinous
  carcinoma ~19%, papillary carcinoma ~18%).
- Static concatenation fusion (0.2752 ± 0.0203) did not beat DenseNet-201 and missed the
  pre-registered threshold (0.3281). Critically, it did not fail uniformly — it improved tubular
  adenoma and papillary carcinoma while degrading lobular and ductal carcinoma. This pattern rules
  out "CNN+ViT fusion doesn't work" and instead implicates the **fusion mechanism**: a fixed
  concatenation cannot adapt which branch to trust for which input.
- Independently, all three completed experiments show the same minority-class failure pattern
  regardless of architecture (CNN, ViT, or static fusion), which implicates the **loss function's**
  handling of severe patient-level imbalance (phyllodes: 3 patients, adenosis: 4, lobular: 5,
  papillary: 6, tubular: 7) as a second, separate cause that architecture changes alone will not
  fix.

Two distinct problems, two distinct pieces of evidence, one committed final design addressing both
in the single remaining training campaign.

## 3. Architecture

```
                         Input (224x224)
                           |
                 +---------+---------+
                 |                   |
                 v                   v
            DenseNet-201          ViT-B/16
           (pretrained,          (pretrained,
          partially fine-tuned)  partially fine-tuned)
                 |                   |
             1920-d              768-d
                 |                   |
                 v                   v
          CNN Projection        ViT Projection
        1920->384, BatchNorm    768->384, LayerNorm
        ReLU, Dropout(0.3)      ReLU, Dropout(0.3)
                 |                   |
                 +---------+---------+
                           |
                    Gating Module
              z = concat(c, t)        [768-d]
              g = sigmoid(Linear(768->384)(z))
                           |
                    Adaptive Fusion
          fused = g * c + (1 - g) * t   [384-d]
                           |
                 Fusion/Task MLP
              384 -> 256, ReLU, Dropout(0.3)
                           |
                  +--------+--------+
                  |                 |
                  v                 v
             Primary Head      Subtype Head
           256 -> 2 classes   256 -> 8 classes
           (standard CE)      (class-balanced CE)
```

## 4. Why This Architecture

- **DenseNet-201 + ViT-B/16 backbones, pretrained, not trained from scratch:** locked decision
  since project inception; both prior baselines validated these as reasonable individual feature
  extractors (primary accuracy 82–86%).
- **Projection dimensions unchanged from Baseline 3 (1920→384, 768→384):** keeps the new result
  attributable to the fusion mechanism and loss change, not to an incidental capacity change.
- **Feature-wise gating replacing static concatenation:** direct, minimal response to Baseline 3's
  documented per-class evidence that a fixed combination helps some subtypes and hurts others. A
  single learnable gate vector lets the model weight CNN versus ViT contribution per feature
  dimension, per input — the simplest mechanism that can express "trust CNN more here, ViT more
  there," which is exactly the behavior static concatenation cannot express. Cross-attention was
  considered and rejected: it adds substantially more parameters and compute for a Colab-Free
  budget without evidence from the completed experiments that a richer interaction than gating is
  needed.
- **Class-balanced loss on the subtype task only:** justified by a pattern present across all
  three completed experiments, not just one — minority-class failure did not go away when the
  architecture changed, which is the signature of a loss/data problem rather than an architecture
  problem. Effective-number-of-samples weighting (Cui et al.) is selected specifically because it
  is designed for the regime this project is actually in: very small per-class sample counts
  (single digits of patients for several subtypes), where plain inverse-frequency weighting
  over-corrects and focal loss does not address the root scarcity. Applied only to Task B: Task A
  (benign/malignant) is only moderately imbalanced (~69/31 in BreakHis) and was not diagnosed as a
  failure point in any completed experiment — adding class-balancing there would be an unjustified,
  untested change.
- **Both changes bundled into one architecture rather than tested separately:** this project has
  budget for one campaign. Bundling is the correct compute-constrained decision given that both
  changes are independently evidenced. The tradeoff — that this run cannot cleanly attribute any
  improvement between "gating" and "class-balanced loss" in isolation — is accepted and disclosed
  (Section 18), not hidden.

## 5. Dataset

- BreakHis, 200X magnification only. No change from all prior experiments.
- Task A (primary, benign/malignant): reuse `split_task_a.csv` exactly as produced for Baseline 3.
  Do not regenerate.
- Task B (subtype, 8-class): reuse `folds_task_b.csv` exactly as produced for Baseline 3. Do not
  regenerate. Patient-level fold assignments, no leakage, identical to Baselines 1–3.
- Known, accepted data constraint, not to be treated as a bug: several subtypes have very few
  independent patients (phyllodes: 3, adenosis: 4, lobular: 5, papillary: 6, tubular: 7). Lobular
  carcinoma has only 1 patient in 4 of 5 folds and 0 in the fifth. Class-balanced loss (Section 8)
  is the project's one designed response to this; if lobular carcinoma remains near-random after
  this run, that is an expected, reportable data-support ceiling, not a failure of this
  architecture to diagnose further.

## 6. Evaluation Protocol

Identical operational protocol to Baselines 1–3 — no changes:

- **Task A:** single patient-level held-out split (`split_task_a.csv`). Report Accuracy,
  Precision, Recall, Macro-F1, confusion matrix, computed once on the held-out test partition
  after model selection on validation.
- **Task B:** patient-level 5-fold cross-validation (`folds_task_b.csv`). One multi-task model
  trained per fold (same operational pattern as Baseline 3: each fold's training partition is used
  to train the full multi-task model; Task A's designated held-out split remains the source of the
  official single Task A number, exactly as in prior baselines). Report per fold and aggregated
  (mean ± std): Accuracy, Macro Precision, Macro Recall, Macro F1, per-class Precision/Recall/F1,
  confusion matrices, and class support per fold — do not hide folds where rare classes have
  insufficient patients.
- Test/held-out data touched only once, after all model selection and early stopping decisions are
  finalized on validation data.

## 7. Loss Function

```
L_total = L_primary + λ * L_subtype_CB

L_primary   = standard cross-entropy (2-class)
L_subtype_CB = class-balanced cross-entropy (8-class), weighting per Section 8
λ = 1.0 (fixed, matching all three completed experiments for comparability)
```

λ is not tuned in this campaign. Changing it would be a second experiment.

## 8. Class Imbalance Strategy

**Method: Class-Balanced Loss via Effective Number of Samples (Cui et al., 2019), applied to
Task B (subtype) only.**

For each fold's training partition, compute per-class image counts `n_1 ... n_8`. Effective
number per class:

```
E_i = (1 - beta^{n_i}) / (1 - beta),   beta = 0.9999
```

Per-class weight:

```
w_i = (1 - beta) / (1 - beta^{n_i})
```

Normalize weights so they sum to the number of classes (8), then use as the per-class weight
vector in the subtype cross-entropy loss. Recompute weights per fold, since class counts differ
slightly across folds (`n_i` = image count in that fold's training partition, not patient count —
patient count is what determines eligibility/severity of imbalance, but the loss operates at the
image level since that is the training unit).

`beta = 0.9999` is fixed for this campaign (standard default from the source method for
small-to-moderate dataset sizes; not swept or tuned — tuning it would constitute a second
experiment).

No other imbalance method (focal loss, balanced sampling, targeted augmentation) is used in this
campaign. This is the one, singular, justified imbalance mechanism per the project constraint.

## 9. Training Configuration

- **Backbone initialization:** load DenseNet-201 and ViT-B/16 weights from the checkpoints used in
  Baseline 3 (themselves fine-tuned from Baseline 1/2). Do not reinitialize from raw ImageNet
  weights.
- **New-component initialization:** projections, gating module, fusion MLP, both heads —
  random init (standard framework defaults).
- **Fixed two-stage schedule, executed unconditionally within each fold's training run (not a
  decision point — both stages always run):**

  **Stage A — epochs 1–8, frozen backbones.** Freeze DenseNet-201 and ViT-B/16 entirely. Train
  projections, gating module, fusion MLP, both heads.

  **Stage B — epochs 9–20, partial unfreeze.** Unfreeze DenseNet-201's final dense block
  (denseblock4) and ViT-B/16's final 2 transformer blocks. Continue training all components from
  Stage A plus these unfrozen backbone layers.

- **Optimizer:** AdamW throughout.
- **Learning rates (discriminative, fixed for the whole run, not tuned):**
  - Stage A: `1e-3` for projections/gate/fusion/heads (backbone frozen, no backbone LR needed).
  - Stage B: backbone (unfrozen layers) `1e-5`; projections/gate/fusion/heads `1e-3`.
- **Weight decay:** `0.01` for all trainable parameters.
- **Batch size:** 16 in Stage A (frozen features are cheap), 8 in Stage B (gradients now flow
  through backbone layers, higher memory cost), with gradient accumulation of 2 steps in both
  stages to reach an effective batch size of 32 / 16 respectively.
- **Mixed precision:** enabled throughout (`torch.cuda.amp`).
- **Max epochs:** 20 total per fold (8 + 12), as specified above — not open-ended.
- **Early stopping:** monitor validation subtype Macro-F1 within each fold; patience = 5 epochs;
  restore best checkpoint by this metric before moving to test/held-out evaluation for that fold.
  Early stopping can end a fold before 20 epochs; it does not extend beyond 20.
- **Reproducibility:** fix random seed (Python, NumPy, framework, CUDA) to the same value used in
  Baselines 1–3. Log the seed.

## 10. Google Colab Optimization

- **Frozen-feature caching:** during Stage A, DenseNet-201 and ViT-B/16 outputs do not change per
  epoch (backbones frozen) — precompute and cache both backbones' feature vectors for the entire
  fold's training/validation data once, then train Stage A epochs directly on cached features.
  This is the single largest compute saving available and is what makes 5-fold training feasible
  on Colab Free; use it.
- **Stage B cannot use cached features** (backbone layers now update) — expect Stage B to be the
  compute-heavy portion of each fold.
- Mount Google Drive for checkpoint and metric persistence; do not rely on Colab's ephemeral disk
  surviving a session.
- Save a checkpoint after every epoch of Stage B (not just at the end) so a session
  disconnect loses at most one epoch, not a whole fold.
- Save per-fold results (metrics, confusion matrices, gate-value logs) to Drive immediately after
  each fold completes, before starting the next fold — do not hold all 5 folds' results in memory
  only.
- Expect this to span multiple Colab Free sessions across several days given the 12-hour session
  limit; this is planned for, not an exception-handling scenario. Structure the notebook so it can
  resume from the last completed fold and the last saved epoch within an in-progress fold.

## 11. Exact Implementation Pipeline

1. Load `split_task_a.csv` and `folds_task_b.csv`. Verify row counts and class distributions match
   the values recorded in the Baseline 3 audit. Do not regenerate either file.
2. Fetch BreakHis 200X images from the private Kaggle dataset via Kaggle API (identical to
   Baselines 1–3's data-fetch step).
3. Build the DenseNet-201 and ViT-B/16 backbones, loading Baseline 3's fine-tuned checkpoint
   weights.
4. Build the projection layers, gating module, fusion MLP, and both classification heads as
   specified in Section 3.
5. Implement the class-balanced weight computation (Section 8) as a function of a given fold's
   training-partition class counts; recompute per fold.
6. For each of the 5 folds in `folds_task_b.csv`:
   a. Precompute and cache frozen-backbone features for that fold's train/val data.
   b. Run Stage A (epochs 1–8) on cached features.
   c. Run Stage B (epochs 9–20) with backbone layers unfrozen as specified, with early stopping.
   d. Evaluate the fold's best checkpoint on the fold's held-out validation partition (Task B
      metrics) and log gate-value distributions (Section 12).
   e. Save fold results and checkpoint to Drive.
7. After all 5 folds: aggregate Task B metrics (mean ± std) across folds.
8. Separately, using `split_task_a.csv`'s designated train/val/test partition, run the same
   architecture and training schedule once to produce the official single-split Task A metrics
   (Accuracy, Precision, Recall, Macro-F1, confusion matrix) on the held-out test set.
9. Compile the comparison table (Section 13) and ablation summary (Section 14).
10. Freeze all results as final — no further tuning against these numbers.

## 12. Metrics

**Task A:** Accuracy, Precision, Recall, Macro-F1, confusion matrix (single held-out test set).

**Task B, per fold and aggregated (mean ± std):** Accuracy, Macro Precision, Macro Recall, Macro
F1, per-class Precision/Recall/F1, confusion matrices, class support per fold.

**Additional diagnostic (not a standard metric, but required output):** distribution of gate
values `g` across the validation set, per fold, and where possible broken down by subtype — this
provides interpretable evidence of whether the gate is behaving adaptively (values shifting by
input/class) or has collapsed toward a near-constant value (indicating the gate is not adding
real adaptivity, which is directly relevant to Section 16's interpretation of results).

## 13. Comparison With Existing Baselines

| Model | Task A Accuracy | Task B Macro-F1 (mean ± std) |
|---|---:|---:|
| DenseNet-201 | 0.8605 | 0.2938 ± 0.0343 |
| ViT-B/16 | 0.8240 | 0.2688 ± 0.0540 |
| Static CNN+ViT Fusion | 0.8336 | 0.2752 ± 0.0203 |
| **Final Model (Gated Fusion + CB Loss)** | measure | measure |

Pre-registered success threshold, unchanged: **0.3281** (DenseNet-201 mean + 1 std, computed at
the time Baseline 3 was evaluated; not recomputed or adjusted after seeing the final model's
result).

## 14. Ablation Strategy Using Existing Results

Comparisons obtainable **without any additional training**, directly from artifacts already
produced by Baselines 1–3:

- Does CNN alone help? — DenseNet-201 result, already available.
- Does ViT alone help? — ViT-B/16 result, already available.
- Does naive fusion help? — Static concatenation result, already available.
- Per-class effect of naive fusion versus each individual backbone — already available from
  Baseline 3's per-class F1 comparison artifact.

Comparison requiring the one new campaign in this plan (no additional training beyond what
Section 9–11 already specifies):

- Does gated fusion + class-balanced loss improve over all three prior results, overall and
  per-class? — this run's result, once complete.

**Explicitly not performed, and disclosed as a limitation (Section 18):** an isolated ablation of
"gating alone, without class-balanced loss" or "class-balanced loss alone, without gating" — this
would require a second training campaign, which is out of scope for this project's compute budget.
The final result cannot cleanly attribute improvement (if any) between the two components. This
is a deliberate, disclosed tradeoff, not an oversight.

## 15. Success Criteria

The final model is considered to have **succeeded** if:

- Task B aggregated Macro-F1 (mean) ≥ 0.3281 (clears the original pre-registered threshold), and
- The improvement is not concentrated solely in the dominant ductal carcinoma class — at least
  some of the previously worst-performing minority classes (mucinous, papillary, phyllodes tumor
  where fold support allows) show improved per-class F1 relative to the best of the three prior
  models.

Both conditions matter: a threshold-clearing result driven entirely by the majority class would
not represent genuine progress on the stated research question (subtype discrimination, not just
aggregate accuracy).

## 16. Failure Interpretation

This section is prepared in advance so the paper's interpretation does not depend on the outcome.

- **If the final model clears 0.3281 and improves minority-class F1 broadly:** report it as the
  proposed model. State plainly that the improvement combines two mechanisms (gating and
  class-balanced loss) whose individual contributions were not separately isolated, per Section 14.
- **If the final model beats Baseline 3 (0.2752) and/or DenseNet-201 (0.2938) but does not clear
  0.3281:** report this as a genuine, if partial, improvement. The correct paper framing is that
  adaptive fusion combined with class-balanced learning measurably helps subtype discrimination
  under BreakHis's severe patient-level imbalance, but the underlying data scarcity for several
  subtypes (single-digit patient counts) remains a hard ceiling that architecture and loss design
  alone cannot fully overcome. This is not framed as a failure; it is framed as a bounded,
  evidence-based result consistent with the diagnosed data constraint in Section 5.
- **If the final model does not beat Baseline 3:** report this directly and honestly. The paper's
  contribution in this case shifts to the diagnostic value of the full experimental sequence
  itself: three architectures and one class-imbalance mechanism were tested under a rigorous,
  leakage-free, patient-level evaluation protocol, and the consistent finding is that BreakHis's
  per-patient data scarcity for minority histopathological subtypes is the binding constraint on
  subtype classification performance — a finding with its own value for the field, independent of
  whether any single model "won." Do not follow this outcome with a proposal for a further model;
  present it as the paper's evidence-based conclusion.

## 17. Publication Positioning

Frame the paper around the full experimental sequence, not solely the final number:

> A controlled comparison of pretrained DenseNet-201, ViT-B/16, static feature-concatenation
> fusion, and gated feature fusion with class-balanced learning for breast histopathological
> subtype classification on BreakHis, under a leakage-free patient-level evaluation protocol,
> showing [state the actual Section 16 outcome once known] and quantifying the impact of
> per-patient class scarcity on achievable subtype-classification performance.

The rigor of the evaluation protocol (patient-level splitting, 5-fold CV, pre-registered
threshold, honest reporting of insufficient-support folds) is itself a defensible contribution
regardless of which Section 16 branch the numeric result falls into.

## 18. Limitations

To be disclosed in the paper, regardless of outcome:

- BreakHis provides only 82 total patients (24 benign, 58 malignant); several subtypes have
  single-digit patient counts (phyllodes: 3, adenosis: 4, lobular: 5, papillary: 6, tubular: 7),
  which fundamentally limits the achievable reliability of subtype-level evaluation, independent
  of model architecture.
- Lobular carcinoma has only 1 patient in 4 of 5 folds and 0 in the fifth; its cross-validated
  metric should be interpreted with this in mind, not treated as equivalent in reliability to
  better-supported classes.
- The final model bundles two independently-motivated changes (gated fusion, class-balanced loss)
  in a single training campaign; their individual contributions were not isolated via separate
  ablation, due to Colab Free compute constraints. This is a deliberate scope limitation, not an
  unnoticed gap.
- This is a 2D histopathology study; it does not generalize to or make claims about breast MRI or
  other imaging modalities.
- Results are obtained on Google Colab Free-tier compute, which constrained batch size, epoch
  budget, and the extent of backbone fine-tuning (only the final block/2 transformer blocks
  unfrozen); a higher-compute environment might shift absolute numbers, though the relative
  comparison methodology remains valid.
- The class-balanced loss hyperparameter (`beta = 0.9999`) was fixed at a standard literature
  default and not tuned for this dataset, due to the single-campaign constraint.

## 19. Reproducibility Requirements

- Use the exact same random seed as Baselines 1–3.
- Log and save: `beta` value, all learning rates, batch sizes, gradient accumulation steps,
  epoch counts per stage, early-stopping patience, and per-fold class-balanced weight vectors.
- Save `split_task_a.csv` and `folds_task_b.csv` alongside the final model's results directory —
  do not rely on them existing only in the Baseline 3 artifacts location.
- Save per-fold checkpoints (best-by-validation-Macro-F1), not only the final epoch's weights.
- Save gate-value distribution logs per fold as a distinct artifact, not just summary statistics —
  needed for the interpretive discussion in Section 16/17.
- Record exact library versions (PyTorch/torchvision or timm, CUDA version) used for training, for
  the paper's reproducibility statement.

## 20. Final Implementation Checklist

- [ ] `split_task_a.csv` and `folds_task_b.csv` loaded and verified against Baseline 3's audit
      (no regeneration)
- [ ] DenseNet-201 and ViT-B/16 backbones initialized from Baseline 3's fine-tuned checkpoints
- [ ] Projection layers, gating module, fusion MLP, both heads implemented per Section 3
- [ ] Class-balanced weight computation implemented per Section 8, recomputed per fold
- [ ] Stage A (frozen, cached features) and Stage B (partial unfreeze) both implemented as a fixed,
      unconditional two-stage schedule per fold
- [ ] All 5 folds trained, evaluated, and results saved to Drive incrementally
- [ ] Single Task A held-out run completed using `split_task_a.csv`
- [ ] Gate-value distribution logged per fold
- [ ] Comparison table (Section 13) filled in with actual numbers
- [ ] Section 15 success criteria evaluated explicitly (met / partially met / not met)
- [ ] Section 16 interpretation branch selected and written based on actual outcome — not decided
      in advance of the result
- [ ] All hyperparameters, seed, and library versions logged per Section 19
- [ ] No further architecture, loss, or hyperparameter changes made after this campaign completes
