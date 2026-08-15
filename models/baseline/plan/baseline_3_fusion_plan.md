# Experiment 3 Plan: DenseNet-201 + ViT-B/16 Feature-Level Fusion (BreakHis) — Corrected Protocol

## 0. Role of This Document

This is a build spec for a single Google Colab notebook (`.ipynb`), handed to an implementation
agent with no other context. This is Experiment 3 of 3 in the CNN-ViT feature-fusion research
project. It has two jobs, not one:

1. Build and evaluate the first hybrid model: DenseNet-201 + ViT-B/16 feature-level fusion.
2. **Correct a methodology flaw discovered in Baseline 1 (DenseNet-201) and Baseline 2 (ViT-B/16):**
   the subtype classification test set had zero samples for phyllodes tumor, meaning the 8-class
   subtype evaluation was not actually fully evaluated in either baseline.

Do not run this notebook against the old split or the old evaluation protocol. Everything in
Section 2 must be resolved and confirmed working *before* the hybrid model in Section 5 onward is
built. This is not optional groundwork — it is the actual point of this experiment, per direct
instruction. The output of this notebook is what the next stage (final model, intended for
publication) will be built on top of. Get the evaluation right here or every downstream number
is compromised.

## 1. Why the Old Protocol Broke

BreakHis has only 82 patients total (24 benign, 58 malignant) spread across 8 subtypes. Some
subtypes — phyllodes tumor in particular — have very few patients. A single fixed patient-level
train/val/test split can, and did, leave a subtype with zero patients in the test partition. When
that happens, the reported subtype metrics (accuracy, macro-F1, etc.) are not a real evaluation —
`support = 0` for that class means no evaluation happened for it, even though a number gets
printed. This is a dataset-structure problem, not a bug in the DenseNet/ViT notebooks. It cannot be
fixed by re-splitting once and hoping for a better random draw — with this few patients in the
rarest classes, a single 3-way split is inherently unstable. It needs a different evaluation
strategy, not just a different split.

## 2. Corrected Evaluation Strategy (Mandatory — Build This First)

### 2.1 Task A (primary: benign vs malignant) — keep a single held-out split

Benign/malignant has enough patients per class that a single stratified patient-level
train/val/test split remains valid and stable. No change needed here beyond re-verifying the split
(Section 3).

### 2.2 Task B (subtype, 8-class) — switch to patient-level stratified k-fold cross-validation

Do not evaluate subtype classification on a single fixed test split. Instead:

1. Use **patient-level stratified k-fold cross-validation** (k=5, matching the screening
   infrastructure already built in Baseline 1/2) as the actual reported subtype evaluation —
   not just a stability check run on the side.
2. For each fold: patients assigned to that fold's held-out portion must not appear in the
   training portion for that fold (same patient-level leakage rule as before, applied per fold
   instead of once).
3. Train and evaluate the model once per fold, report **mean ± standard deviation** across folds
   for every subtype metric (accuracy, macro precision, macro recall, macro F1, per-class
   precision/recall/F1).
4. **If any subtype still has zero (or only one) patients in a given fold's held-out portion,**
   do not silently report a 0.0 or omit the class. Instead:
   - Flag that class explicitly as "insufficient patient count for stable cross-validated
     evaluation" in the results output.
   - Report whatever folds *do* contain that class, and note how many folds contributed.
   - This honest flag is a legitimate, reportable limitation — it is not a failure of the
     experiment. Do not try to force a number into existence for a class that genuinely lacks
     enough independent patients.
5. Keep the primary-task evaluation (2.1) and subtype evaluation (2.2) methodologically separate —
   do not force Task A into cross-validation just because Task B needs it, and vice versa.

### 2.3 Define the success threshold before looking at results

Because subtype macro-F1 will now come with a standard deviation across folds, define in advance
what counts as a real improvement versus noise:

> **A hybrid subtype macro-F1 improvement only counts as a real finding if it exceeds
> approximately one cross-fold standard deviation above the better of the two baselines.**

Write this threshold into the notebook/results doc *before* running the hybrid model. Do not decide
the threshold after seeing the fusion result.

## 3. Dataset Scope

- **Dataset:** BreakHis, **200X magnification only** — same subset as Baseline 1 and 2.
- **Storage/access:** Private Kaggle dataset, fetched via Kaggle API into Colab. No hardcoded
  credentials.
- **Patient/case ID extraction:** Same method as Baseline 1 (parsed from filenames).
- **Re-audit before building anything:**
  1. Recompute unique patient count per subtype in the 200X subset.
  2. Confirm which subtypes can support a stable single held-out fold and which cannot — document
     this explicitly (this becomes the "insufficient data" list referenced in 2.2.4).
  3. Regenerate `split_task_a.csv` (single 3-way patient split, Task A only) and
     `folds_task_b.csv` (k=5 patient-level fold assignments, Task B only) as the new shared
     artifacts. Do not reuse the old `split.csv` from Baseline 1/2 as-is — it produced the
     phyllodes zero-support failure.
  4. These two new CSVs become the fixed source of truth going forward. If the final model
     (post-Baseline-3) is built later, it must also use these same artifacts.

## 4. Data Pipeline

- Preprocessing: resize to 224×224. Normalize DenseNet-201 branch with ImageNet stats; normalize
  ViT-B/16 branch with the stats matching whatever pretrained checkpoint was used in Baseline 2
  (confirm they match — do not assume identical to DenseNet's).
- Augmentation (train only): same policy as Baseline 1/2 — horizontal/vertical flip, small-angle
  rotation (±20°), mild color jitter. Keep identical to the baselines for fair comparison; this is
  not the place to introduce new augmentation.

## 5. Hybrid Model Architecture

```
                    Input Image
                        |
              +---------+---------+
              |                   |
              v                   v
        DenseNet-201           ViT-B/16
        CNN branch            Transformer branch
              |                   |
         1920-d feature        768-d feature
              |                   |
              v                   v
       Projection Layer     Projection Layer
        (1920 -> 384)         (768 -> 384)
              |                   |
              +---------+---------+
                        |
                Feature Concatenation
                        |
                     768-d
                  fused feature
                        |
                 Fusion MLP (768 -> 384 -> dropout)
                        |
              +---------+---------+
              |                   |
              v                   v
        Primary Head         Subtype Head
          2 classes            8 classes
```

- Keep this simple first version exactly as specified. No attention, no gating, no cross-attention,
  no transformer-on-top-of-transformer. That is explicitly out of scope for this experiment.
- If Phase 1 (frozen-backbone) results land suspiciously close to the better single baseline,
  treat the 384-dim projection width as a documented, explicit tuning step before concluding
  fusion doesn't help — don't let an under-sized bottleneck masquerade as a negative result.

## 6. Initialization

- Load the **best surviving checkpoint** from Baseline 1 (DenseNet-201) and Baseline 2 (ViT-B/16).
  Do not train either backbone from scratch inside this notebook.
- Note: because the evaluation protocol changed (Section 2), the baseline checkpoints being reused
  were trained under the *old* split. This is acceptable for initializing backbone weights (the
  representations themselves are still valid pretrained-then-fine-tuned features), but the hybrid
  model's own training and evaluation in this notebook must use the *new* split/fold artifacts
  from Section 3. Do not evaluate the hybrid on the old split for comparability — comparability
  comes from re-running the metric computation the same way for all three models, not from reusing
  a broken split.
- **Practical implication:** for a clean apples-to-apples final comparison, DenseNet-201 and
  ViT-B/16 should also be re-evaluated (not necessarily retrained) under the new Task A split and
  Task B folds, so all three models' reported numbers come from the same evaluation protocol. If
  full retraining under the new folds is too costly for Colab, at minimum re-run evaluation-only
  passes of the existing baseline checkpoints through the new split/fold artifacts before declaring
  the three-way comparison in Section 10. Flag clearly in the results doc whether baseline numbers
  are "retrained on new folds" or "existing checkpoint, re-evaluated on new split" — don't blur the
  two.

## 7. Training Strategy

### Phase 1 — Frozen backbones (run first, cheapest, most informative)

Freeze both DenseNet-201 and ViT-B/16 entirely. Train only:
- DenseNet projection layer
- ViT projection layer
- Fusion MLP
- Primary head
- Subtype head

This tests whether the two independently-learned representations are complementary before any
further fine-tuning cost is spent. This should be the first result reported.

### Phase 2 — Partial unfreeze (only if Phase 1 shows promise)

Unfreeze only the final block(s) of each backbone. Use a lower learning rate for backbone layers
than for fusion/head layers. Do not fully unfreeze both networks on the first hybrid run — this is
both a cost-control measure for Colab and a controlled-experiment discipline (don't change backbone
weights and fusion architecture in the same uncontrolled step).

### For Task B specifically (cross-validated)

Both Phase 1 and Phase 2 (if reached) must be run once per fold under the k-fold protocol from
Section 2.2. This is more Colab-expensive than a single split — budget for it. If full 5-fold
training of Phase 2 is not feasible on available Colab compute, it is acceptable to run Phase 1
(frozen, cheap) across all 5 folds, and reserve Phase 2 (unfrozen, expensive) for a single
representative fold or a reduced fold count, clearly labeled as such in the results.

## 8. Loss

```
total_loss = primary_loss + λ × subtype_loss
```

- λ = 1.0 initially, matching Baseline 1/2 for comparability.
- Class-imbalance handling: reuse whatever strategy (weighted CE / focal loss / etc.) was
  established from the Baseline 1 class-distribution audit. Do not introduce a new imbalance
  method and a new architecture in the same experiment — that confounds which change caused any
  observed effect.
- Do not tune λ against results in this first run. If λ tuning is warranted later, it's a
  documented follow-up ablation, not part of establishing the first fusion baseline.

## 9. Evaluation Protocol

### Task A (primary) — single held-out test set (per Section 2.1)
- Accuracy, Precision, Recall, Macro-F1, confusion matrix.

### Task B (subtype) — patient-level 5-fold cross-validation (per Section 2.2)
- Per fold: Accuracy, Macro Precision, Macro Recall, Macro F1, per-class Precision/Recall/F1,
  confusion matrix.
- Aggregate: mean ± std for every metric above across folds.
- Explicit flag for any subtype with insufficient patients for stable cross-fold evaluation
  (per 2.2.4) — do not omit or silently zero these out.

## 10. Primary Comparison and Success Criterion

The central scientific question of this experiment:

```
DenseNet-201 subtype Macro-F1 (re-evaluated, new folds)
        vs
ViT-B/16 subtype Macro-F1 (re-evaluated, new folds)
        vs
DenseNet+ViT fusion subtype Macro-F1 (new folds)
```

Do not judge success by Task A primary accuracy alone — both baselines already perform reasonably
there. The open problem this experiment is testing is subtype classification, particularly the
previously-failing minority classes (lobular carcinoma, mucinous carcinoma, papillary carcinoma,
and phyllodes tumor if it now has stable fold coverage).

Apply the pre-registered threshold from Section 2.3: a fusion improvement only counts as real if
it clears roughly one cross-fold standard deviation above the stronger baseline. Report the
comparison honestly even if the result is negative or inconclusive — a fusion model that doesn't
beat the threshold is still a valid, useful, reportable outcome that shapes what the final model
needs to do differently (e.g., pursue explicit class-imbalance handling before or alongside
architectural changes).

## 11. Colab-Specific Practicalities

- Mount Drive for checkpoint/results persistence — do not lose fold-by-fold results to a session
  timeout, since 5-fold training is the most expensive run so far.
- Cache DenseNet and ViT feature embeddings where possible during Phase 1 (frozen-backbone) runs —
  since backbones don't update in Phase 1, embeddings can be precomputed once per fold rather than
  recomputed every epoch. This significantly reduces Phase 1 compute cost and is the main lever for
  making 5-fold training Colab-feasible.
- Use mixed precision (`torch.cuda.amp`) throughout.
- Save per-fold checkpoints and per-fold metrics separately before aggregating — if a session
  disconnects mid-fold-sweep, you should be able to resume from the last completed fold rather than
  restarting the whole sweep.

## 12. Deliverables Checklist

- [ ] Documented re-audit of patient counts per subtype (Section 3)
- [ ] `split_task_a.csv` (new, single 3-way split for primary task)
- [ ] `folds_task_b.csv` (new, 5-fold patient-level assignments for subtype task)
- [ ] Explicit list of any subtypes flagged as insufficient for stable CV, with justification
- [ ] Re-evaluated (or retrained) DenseNet-201 and ViT-B/16 numbers under the new protocol,
      clearly labeled as retrained vs. re-evaluated-only
- [ ] Trained hybrid model notebook (`.ipynb`), Phase 1 minimum, Phase 2 if feasible
- [ ] Per-fold and aggregated (mean ± std) subtype metrics for all three models
- [ ] Single held-out primary-task metrics for all three models
- [ ] Explicit statement of whether the fusion result clears the pre-registered success threshold
- [ ] Recorded hyperparameters/config (seed, LR, λ, optimizer, projection dim, epochs per fold)

## 13. Explicit Guardrails

- Do not reuse the old `split.csv` from Baseline 1/2 — it is known to be broken for subtype
  evaluation.
- Do not report a subtype metric for a class with zero test/fold support as if it were a valid
  evaluation — flag it instead.
- Do not decide the success threshold after seeing results.
- Do not add attention/gated fusion, new augmentations, or a new imbalance method in this
  experiment — simple fusion only, isolate one architectural change at a time.
- Do not skip re-evaluating (at minimum) the two baselines under the new protocol — a fusion vs.
  baseline comparison across two different evaluation protocols is not a valid comparison.
- Do not proceed to the final model until this notebook's results and the pre-registered threshold
  comparison are complete and documented.

## 14. What Comes After This Notebook

The output of this experiment — specifically, whether fusion clears the success threshold on
subtype macro-F1, and which minority classes remain unresolved — is what determines the direction
of the final model. Do not decide the final model's improvement path (attention fusion vs.
class-balanced loss vs. both) before these results exist. That decision is explicitly deferred to
the next planning stage, after this notebook's output is in hand.
