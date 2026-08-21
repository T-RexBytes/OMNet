# FINAL IMPLEMENTATION SPECIFICATION
## `breast_cancer_research.ipynb` — Multi-Task Breast Histopathology Detection & Subtype Classification

This document is the complete, locked build specification for a Google Colab notebook. It supersedes the prior model plan wherever the two disagree; disagreements and the reasoning behind each change are called out explicitly in the re-audit below. Nothing in this document should require the coding agent to invent a missing decision — any true residual ambiguity is isolated in the **ASSUMPTIONS REQUIRING USER CONFIGURATION** section at the end.

---

## 0. RE-AUDIT OF THE PRIOR PROPOSAL (changes made, and why)

| Prior design element | Audit finding | Resolution |
|---|---|---|
| Deep Equilibrium (DEQ) fixed-point refinement with implicit differentiation | Requires a custom solver (Anderson acceleration), custom backward pass, ~2.8× training-time overhead, and documented rare non-convergence even in the source paper's controlled setting. This is a fragile component for an unattended Colab Free session — directly against the reliability mandate. | **Replaced with a weight-tied, fixed-iteration unrolled refinement block** (`N_ITERS`, default 6, config-controlled). Same scientific claim — implicit-depth-style iterative feature refinement from a small shared block — implemented as an ordinary `for` loop with standard autograd, no custom solver, no convergence risk. This is a conscious simplification, stated here rather than silently substituted. |
| Multi-task Detection + Subtype heads with a "consistency loss" | In BreakHis, the detection label is a **deterministic function** of the subtype label (grouping, not independent annotation). Framing this as two independently-informative tasks overstates what the data supports. The consistency loss term, in particular, penalizes disagreement between two things that are trivially derivable from each other and adds little beyond what weighting the subtype loss more heavily already achieves. | **Consistency loss term is removed.** The detection head is retained only as an *auxiliary regularizing signal* (a genuinely testable, literature-motivated design choice — auxiliary heads on a coarser version of a fine label are a known regularization strategy), described honestly as such, and its value is tested directly in the ablation hierarchy (A2 vs A3). Primary evaluation and model selection is on the subtype task. |
| DenseNet-201 as the only backbone option | DenseNet-201 is not the lightest option for a T4 with limited session time; no fallback was specified if it doesn't fit. | Backbone is now a config field with `densenet201` (default) and `efficientnet_b0` (fallback) both implemented and tested paths, not just a hardcoded choice. |
| No explicit Kaggle/Drive/config-centralization/dev-test separation | Missing entirely from the prior plan | Added in full below (Sections 2–7). |
| Custom `DataLoader` behavior unspecified | Risk of accidental `num_workers>0` defaults causing Colab worker crashes | `num_workers=0` is the hard default; increasing it requires an explicit, documented, tested opt-in (Section 4). |

Everything else — the dataset choice (BreakHis primary, IDC cross-dataset evaluation only), the patient-grouped leakage protocol, the literature benchmark table and its fair-comparison caveats, the Fourier-KAN + attention components, and the baseline/ablation logic — is preserved because it remains the strongest scientifically justified part of the proposal and nothing in this re-audit undermines it.

---

## 1. Research Question

> On the public BreakHis dataset, does a shared-backbone architecture combining an ImageNet-pretrained CNN encoder with an attention-enhanced Fourier-KAN residual block and a weight-tied, fixed-iteration refinement stage (a reliability-motivated simplification of DEQ-style implicit-depth refinement, extending Ali et al. 2026 — which validated the FKAN+attention+DEQ combination only for binary classification) improve 8-class subtype macro-F1 beyond from-scratch and pretrained single-task baselines, and does an auxiliary binary detection head improve that result further — all under a patient-level, leakage-safe evaluation protocol, with a frozen final test set evaluated exactly once?

Two falsifiable sub-claims, each with its own ablation (Section 18): (a) the FKAN + iterative-refinement combination improves subtype macro-F1 over a plain pretrained backbone; (b) the auxiliary detection head improves subtype performance over the single-task subtype model.

---

## 2. Final Model

```
Input image (224×224×3, ImageNet-normalized)
        ↓
Shared CNN Backbone — config: "densenet201" (default) or "efficientnet_b0" (fallback), ImageNet-pretrained
        ↓  global-avg-pool → feature vector (1920-d for densenet201 / 1280-d for efficientnet_b0)
Linear projection → 256-d
        ↓
Attention-enhanced Fourier-KAN residual block (Fourier grid size g=8; LCBAM: channel attention reduction r=16 + depthwise 7×7 spatial attention)
        ↓
Weight-tied fixed-iteration refinement stage (N_ITERS=6 default, same block reapplied, standard backprop through the unrolled loop — NOT an implicit-differentiation DEQ solver)
        ↓ shared 256-d representation
        ├── Subtype Head (primary): Linear(256→8), softmax
        └── Detection Head (auxiliary): Linear(256→2), softmax
```

No separate "main-type" head: BreakHis's binary label is a strict grouping of its 8-way subtype label, not an independent annotation (see Section 0 audit). Forcing a third, redundant head is not scientifically justified and is not built.

---

## 3. Dataset Source and Acquisition (Kaggle-first)

### 3.1 Primary dataset — BreakHis
- **Kaggle slug (default, config-overridable):** `KAGGLE_DATASET_BREAKHIS = "ambarish/breakhis"` — the coding agent must treat this as a config default, not a hardcoded certainty, and verify the actual directory layout after download (Section 3.4) rather than assuming it matches documentation exactly.
- **Task support:** binary detection + 8-way subtype (both derivable from the same folder structure).

### 3.2 Secondary dataset — IDC (cross-dataset evaluation only, never used in training)
- **Kaggle slug (default, config-overridable):** `KAGGLE_DATASET_IDC = "paultimothymooney/breast-histopathology-images"`.
- **Task support:** binary detection only. No subtype labels — never used for the subtype task.

### 3.3 Kaggle credential workflow

```python
# --- Kaggle credential configuration ------------------------------------
# Preferred: Colab Secrets (Colab > Secrets panel, key icon in left sidebar)
#   Add two secrets named KAGGLE_USERNAME and KAGGLE_KEY.
# The notebook will try Colab Secrets first, then fall back to a manually
# uploaded kaggle.json, and will NEVER print, log, or persist the token
# value anywhere in cell output, files, or saved artifacts.

from google.colab import userdata  # only available in Colab

def load_kaggle_credentials():
    try:
        username = userdata.get('KAGGLE_USERNAME')
        key = userdata.get('KAGGLE_KEY')
        source = "colab_secrets"
    except Exception:
        username, key, source = None, None, None

    if not username or not key:
        # Fallback: user uploads kaggle.json manually via the Colab file
        # upload widget. The notebook must clearly print:
        #   "Kaggle credentials not found in Colab Secrets.
        #    Please upload your kaggle.json file now."
        # and use `files.upload()` (ipywidgets-free, standard Colab widget,
        # not a multiprocessing/background mechanism) to receive it.
        source = "uploaded_file"
        # kaggle.json is written to ~/.kaggle/kaggle.json with mode 600
        # and is NEVER echoed back or logged.

    return source  # the notebook logs only which source was used, never the values
```

- **Rule, non-negotiable:** the actual `KAGGLE_USERNAME`/`KAGGLE_KEY` values must never appear in any printed cell output, any saved file under the Drive run directory, or any log line. Only the *source* of the credential (`colab_secrets` or `uploaded_file`) is logged.
- **Installation:** check first (`import kaggle` inside a `try/except`) before running `pip install -q kaggle`; do not reinstall if already present.

### 3.4 Download, extraction, and root discovery

```python
# 1. kaggle datasets download -d {slug} -p {raw_data_dir} --unzip
# 2. Verify the download produced at least one file; if zero files, stop
#    with a clear error naming the slug and the expected local path.
# 3. Discover the actual dataset root by walking the extracted directory
#    and searching for the expected BreakHis folder signature
#    ("SOB" directory present, subfolders "benign"/"malignant" present)
#    rather than assuming a fixed nesting depth — Kaggle mirrors of
#    BreakHis have historically varied in nesting.
# 4. If the expected signature is not found after search, STOP and print
#    the actual directory tree (first 3 levels) so the user can update
#    DATA_ROOT_OVERRIDE in Config rather than the notebook guessing wrong.
```

- **Verification of downloaded files:** file count check, corrupt-archive check (extraction must not silently produce zero images), and a printed summary (`N files found, M under benign/, K under malignant/`).
- **What happens if the dataset structure differs:** the notebook does not guess silently. It prints the discovered tree and raises a clear, actionable error asking the user to set `Config.DATA_ROOT_OVERRIDE` to the correct path. It never proceeds on an unverified path.
- **Updating the dataset source:** exclusively through `Config.KAGGLE_DATASET_BREAKHIS` / `Config.KAGGLE_DATASET_IDC` / `Config.DATA_ROOT_OVERRIDE` — no other code path changes the dataset source.

---

## 4. Google Drive Output Storage

### 4.1 Structure

```text
MyDrive/
└── breast_cancer_research/
    └── runs/
        └── {RUN_ID}/                      # RUN_ID = YYYYMMDD_HHMMSS, generated once at notebook start
            ├── config.json
            ├── environment.json
            ├── dataset_manifest.csv
            ├── split_manifest.csv
            ├── class_distribution.csv
            ├── leakage_report.json
            ├── training_history/
            │     ├── {model_name}_fold{k}_seed{s}.csv
            ├── checkpoints/
            │     ├── {model_name}_fold{k}_seed{s}_best.pt
            │     ├── {model_name}_fold{k}_seed{s}_final.pt
            ├── predictions/
            │     ├── {model_name}_test_predictions.csv
            ├── figures/
            │     ├── {figure_id}.pdf   (vector, primary)
            │     ├── {figure_id}.png   (raster fallback, 300 DPI)
            ├── tables/
            │     ├── baseline_results.csv
            │     ├── ablation_results.csv
            │     ├── statistical_comparison.csv
            │     ├── final_test_results.csv
            ├── reports/
            │     ├── final_report.md
            └── logs/
                  └── run_log.txt
```

### 4.2 Behavior
1. Mount Drive at notebook start (`drive.mount('/content/drive')`), inside a `try/except` — if mounting fails, the notebook does not silently switch to ephemeral-only storage without telling the user; it prints a clear warning, writes to `/content/local_run_backup/{RUN_ID}/` instead, and reminds the user at the end of the run to manually copy that directory before the Colab session ends (since `/content` is wiped on disconnect).
2. `RUN_ID` and the full run directory are created before any training begins.
3. `config.json` is written **before** training starts (Section 7).
4. `environment.json` is written immediately after (Section 21).
5. Checkpoints are written after every epoch that improves the model-selection metric (Section 9), not just at the end — protects against session interruption.
6. Training history is appended incrementally (one row per epoch, flushed to disk every epoch), not held in memory until the end.
7. Final metrics, predictions, and figures are written once the corresponding stage completes.
8. **Never rely solely on `/content` (ephemeral Colab local disk) for anything in the artifact list above.** Local disk is used only as a fast intermediate cache (e.g., resized-image cache) that can be safely regenerated.

---

## 5. Central Configuration Object

A single `Config` object (Python `dataclass` or nested `dict` — coding agent's choice, but it must be a **single object**, not scattered constants) holds everything below. No magic numbers elsewhere in the notebook.

```python
Config = {
    # --- Environment ---
    "SEED": 42,
    "DETERMINISTIC": True,          # torch.use_deterministic_algorithms where feasible
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MIXED_PRECISION": True,        # torch.cuda.amp, auto-disabled if DEVICE == "cpu"

    # --- Dataset ---
    "KAGGLE_DATASET_BREAKHIS": "ambarish/breakhis",
    "KAGGLE_DATASET_IDC": "paultimothymooney/breast-histopathology-images",
    "DATA_ROOT_OVERRIDE": None,     # set explicitly if auto-discovery (Sec 3.4) fails
    "OUTPUT_ROOT": "MyDrive/breast_cancer_research/runs",
    "IMAGE_SIZE": 224,
    "SPLIT_RATIOS": {"train": 0.70, "val": 0.15, "test": 0.15},
    "GROUPING_KEY": "patient_id",
    "N_CV_FOLDS": 5,
    "SUBTYPE_MAP": {  # fixed, documented, never inferred at runtime
        "adenosis": 0, "fibroadenoma": 1, "phyllodes_tumor": 2, "tubular_adenoma": 3,
        "ductal_carcinoma": 4, "lobular_carcinoma": 5, "mucinous_carcinoma": 6, "papillary_carcinoma": 7,
    },
    "BENIGN_SUBTYPE_IDS": [0, 1, 2, 3],   # detection label derived from this, not re-annotated

    # --- DataLoader ---
    "BATCH_SIZE": 32,               # auto-halved on CUDA OOM, see Sec 20
    "NUM_WORKERS": 0,               # hard default; see Sec 6 for the opt-in path
    "PIN_MEMORY": None,             # None -> resolved to (DEVICE=="cuda") at runtime
    "PERSISTENT_WORKERS": False,

    # --- Augmentation (train split only) ---
    "AUG_HFLIP": True,              # tissue orientation is not diagnostic -> safe
    "AUG_VFLIP": True,              # same reasoning
    "AUG_ROTATION_DEG": 15,         # mild, matches literature convention
    "AUG_COLOR_JITTER": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.0},
    # hue left at 0.0 deliberately: H&E stain hue carries diagnostic information
    "AUG_CROP": None,               # no cropping: risks removing the diagnostic region; not used

    # --- Model ---
    "BACKBONE": "densenet201",      # or "efficientnet_b0" (fallback, config-swappable)
    "PRETRAINED": True,
    "DROPOUT": 0.2,
    "PROJECTION_DIM": 256,
    "FKAN_GRID_SIZE": 8,
    "ATTENTION_REDUCTION_RATIO": 16,
    "N_REFINEMENT_ITERS": 6,        # weight-tied unrolled refinement (Sec 0/2), not implicit DEQ
    "NUM_SUBTYPE_CLASSES": 8,
    "NUM_DETECTION_CLASSES": 2,
    "USE_DETECTION_HEAD": True,     # toggled off for the A2 (subtype-only) ablation

    # --- Training ---
    "OPTIMIZER": "adam",
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-5,
    "SCHEDULER": "reduce_on_plateau",   # monitors val macro-F1, patience 5, factor 0.5
    "MAX_EPOCHS": 50,
    "EARLY_STOP_PATIENCE": 10,
    "EARLY_STOP_METRIC": "val_macro_f1",
    "GRAD_CLIP_NORM": 1.0,
    "GRAD_ACCUM_STEPS": 1,          # auto-increased if batch size is auto-halved, see Sec 20
    "CHECKPOINT_EVERY_IMPROVEMENT": True,

    # --- Loss ---
    "SUBTYPE_LOSS": "weighted_ce",  # or "focal" (ablation-tested alternative, gamma=2.0)
    "FOCAL_GAMMA": 2.0,
    "DETECTION_LOSS": "weighted_ce",
    "LOSS_WEIGHT_SUBTYPE": 1.0,
    "LOSS_WEIGHT_DETECTION": 0.3,   # only active when USE_DETECTION_HEAD=True

    # --- Evaluation ---
    "METRICS_SUBTYPE": ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                         "per_class_f1", "per_class_precision_recall", "ovr_roc_auc", "confusion_matrix"],
    "METRICS_DETECTION": ["accuracy", "balanced_accuracy", "precision", "recall",
                           "specificity", "f1", "roc_auc", "confusion_matrix"],
    "FINAL_MODEL_SEEDS": [42, 7, 123],   # 3-seed repetition for the final proposed model only
    "STAT_TEST": "wilcoxon_signed_rank",
    "STAT_ALPHA": 0.05,
    "STAT_CORRECTION": "holm_bonferroni",

    # --- Visualization ---
    "FIGURE_DPI": 300,
    "FIGURE_FORMATS": ["pdf", "png"],
    "FIGURE_STYLE": "seaborn-v0_8-whitegrid",
    "FIGURE_DIR": "figures",  # relative to run directory
}
```

Everything downstream references `Config[...]`; no re-derivation elsewhere in the notebook.

---

## 6. DataLoader Reliability Policy (Colab-specific)

- **`num_workers=0` is the hard default.** The coding agent may add a single, clearly commented, opt-in cell that tests `num_workers=2` on a small dummy batch inside a `try/except` and reports whether it worked — but training itself proceeds with `num_workers=0` unless the user has manually confirmed the opt-in path succeeded on their runtime. This must never be silently auto-enabled.
- **`persistent_workers=False` always when `num_workers=0`** (required by PyTorch; the notebook must not attempt an invalid combination).
- **`pin_memory`** resolves to `True` only when `Config["DEVICE"]=="cuda"`; `False` on CPU.
- **No multiprocessing pools, no forked subprocesses, no distributed samplers, no background prefetch threads beyond what `DataLoader`'s own (disabled-by-default) workers provide.**
- **Sequential, deterministic indexing** for validation/test loaders (`shuffle=False`) so that prediction rows can be matched back to the manifest without ambiguity.

---

## 7. Data Loading, QC, Preprocessing, Augmentation Pipeline

`raw → QC → filtering → resize/normalize → augmentation (train only) → tensor`

| Step | Operation | Detail |
|---|---|---|
| Discovery | Recursive directory walk | Parse `patient_id`, `subtype`, `magnification` from the verified path structure (Sec 3.4), build one manifest `DataFrame` before any model code runs |
| QC — corrupt files | `PIL.Image.open` + `.verify()` per file | Log to `corrupt_files.csv`, exclude, continue — never crash the whole run on one bad file |
| QC — exact duplicates | MD5 hash across the full manifest | Log to `dedup_report.json`; exact duplicates across splits would be a hard leakage failure (Sec 8) |
| QC — near-duplicates | Perceptual hash (`imagehash.phash`), Hamming distance ≤5 (config) | Same-patient, cross-magnification near-duplicates are **expected and normal** for BreakHis — this check exists to confirm the split (Sec 8) correctly groups them, not to remove them |
| Resize | Bilinear → 224×224 | `Config["IMAGE_SIZE"]` |
| Normalize | ImageNet mean/std, fixed constants | Applied identically to train/val/test — no data-fit statistics, so no leakage risk here |
| Augmentation | Train split only, per Config Section 5 | Every augmentation choice has a stated reason inline in Config's comments |
| Caching | Resized+normalized tensors cached to local `/content` disk as `.pt` shards after first pass | BreakHis (~8K images) fits comfortably; cache is regenerable, never the source of truth |

---

## 8. Leakage Prevention and Splitting Protocol

### 8.1 Split
- Grouping key: `patient_id` (BreakHis), `wsi_id` (IDC, evaluation-only so no split needed — IDC is used whole, frozen).
- Ratios: 70/15/15, patient-grouped, stratified by subtype where group sizes allow (`StratifiedGroupKFold`-style logic).
- **Hard rule enforced by an executable assertion, not a comment:** `assert set(train_patients) & set(val_patients) == set() and set(train_patients) & set(test_patients) == set() and set(val_patients) & set(test_patients) == set()`. If this assertion fails, the notebook **stops** — it does not continue with a compromised split.
- 5-fold patient-grouped CV for baseline/ablation model selection and comparison.
- The final test split (15%) is created once, immediately after the manifest is built, and is **never touched, inspected for model-development decisions, or used for hyperparameter selection** — only cross-validation folds carved from the remaining 85% are used for that.

### 8.2 Machine-readable leakage report (`leakage_report.json`)
```json
{
  "exact_duplicates_found": <int>,
  "exact_duplicates_crossing_splits": <int>,
  "near_duplicates_found": <int>,
  "near_duplicates_crossing_splits": <int>,
  "patient_overlap_train_val": <int>,
  "patient_overlap_train_test": <int>,
  "patient_overlap_val_test": <int>,
  "magnification_distribution_per_split": {...},
  "class_distribution_per_split": {...},
  "verdict": "PASS" | "FAIL"
}
```
- **If `exact_duplicates_crossing_splits > 0` or any `patient_overlap_* > 0`: `verdict = "FAIL"` and the notebook raises and halts before any model training cell executes.**
- `near_duplicates_crossing_splits` should be 0 by construction (patient grouping handles this); if it is not, this indicates a bug in the grouping logic, and the notebook halts with that message rather than proceeding.

---

## 9. Training Schedule, Monitoring, and Model Selection

### 9.1 Development vs. final evaluation — explicit separation
```
Dataset verification
→ Exploratory analysis (Figures A–D, Section 15)
→ Patient-grouped split (train+val pool [85%] vs. frozen test [15%], created once)
→ Baseline training (B1, B2) — 5-fold CV on the 85% pool
→ Baseline evaluation — CV metrics only, test set untouched
→ Reproduction gate (B3, literature protocol) — 5-fold CV on the pool
→ Ablation training (A0–A3) — 5-fold CV on the pool
→ Proposed model training (A4/full) — 5-fold CV × 3 seeds on the pool
→ Model selection — by mean CV validation macro-F1 across folds, pool only
→ Configuration locked (no further architecture/hyperparameter changes after this point)
→ Final test evaluation — the locked configuration, retrained on the full 85% pool (all CV folds combined), evaluated exactly once on the untouched 15% test set
→ Error analysis (on test-set predictions)
→ Robustness analysis (IDC cross-dataset)
→ Statistical comparison (on CV fold-level metrics, not test-set-derived — a single test-set evaluation has no fold structure to test statistically)
→ Visualization
→ Final report
```
- **Model selection criterion, stated explicitly and used consistently:** mean validation macro-F1 across the 5 CV folds, computed on the 85% development pool only. The test set plays no role in choosing architecture, hyperparameters, or which ablation "wins."
- **How CV interacts with final evaluation:** CV folds are used only to compare candidate configurations against each other (baselines vs. ablations vs. proposed model) with variance estimates. Once the winning configuration is locked, it is retrained one more time on the *entire* 85% pool (not just one fold) and evaluated once on the frozen test set — this is the number reported as "final test performance," clearly distinguished from CV development numbers in every table.

### 9.2 Monitoring (every epoch, every model)
- Training loss, validation loss (both task losses separately when multi-task).
- Validation macro-F1, validation balanced accuracy.
- Current learning rate (post-scheduler).
- All values appended to `training_history/{model_name}_fold{k}_seed{s}.csv` every epoch (not held in memory).

### 9.3 Checkpointing
- **Best checkpoint:** saved whenever `val_macro_f1` improves (`{model}_fold{k}_seed{s}_best.pt`).
- **Final checkpoint:** saved at training end regardless of whether it's the best (`{model}_fold{k}_seed{s}_final.pt`), for resumability diagnostics.
- **Resume-from-checkpoint:** if a training cell is interrupted (Colab disconnect), rerunning that cell must detect an existing `_final.pt` or partial `training_history` CSV for the same `(model, fold, seed)` key and offer to resume from the last completed epoch rather than silently restarting from scratch and silently overwriting history.

---

## 10. Loss Functions

- **Subtype (primary):** weighted cross-entropy, class weights = inverse frequency computed from the **current CV fold's training partition only** (recomputed per fold, never leaked from val/test). `"focal"` is a config-swappable, ablation-tested alternative (γ=2.0).
- **Detection (auxiliary, `USE_DETECTION_HEAD` toggle):** weighted cross-entropy, same per-fold weighting policy.
- **Combined:** `L = LOSS_WEIGHT_SUBTYPE * L_subtype + LOSS_WEIGHT_DETECTION * L_detection` (no consistency term — see Section 0 audit).
- **Gradient clipping:** global-norm clip at `GRAD_CLIP_NORM=1.0` for training stability, applied to every model including baselines (fair comparison).

---

## 11. Baselines (mandatory, in order of cost)

| ID | Description | Init | Purpose |
|---|---|---|---|
| B1 | From-scratch backbone (`Config["BACKBONE"]`), plain linear subtype head only | Random | Absolute floor |
| B2 | Pretrained backbone, plain linear subtype head only | ImageNet | Standard transfer-learning baseline |
| B3 | Literature reproduction: AttFKAN-DEQ, **binary only**, random init, exact published hyperparameters (Adam lr=1e-3, wd=1e-5, batch=32, max 50 epochs) | Random | Calibration gate — must land within ~3 points of the published 96.60% BreakHis accuracy before any multi-class extension is trusted. If it does not, the notebook halts model development and reports the discrepancy rather than proceeding as if calibrated. |

---

## 12. Ablation Hierarchy

| ID | Configuration | Isolates |
|---|---|---|
| A0 | Backbone (pretrained) + linear subtype head | Same as B2 — the ablation floor |
| A1 | + Fourier-KAN residual block (no attention, no iterative refinement, `N_REFINEMENT_ITERS=1`) | Does the Fourier parameterization alone help? |
| A2 | + LCBAM attention (still `N_REFINEMENT_ITERS=1`) | Does attention add beyond FKAN? |
| A3 | + weight-tied iterative refinement (`N_REFINEMENT_ITERS=6`), subtype-only (`USE_DETECTION_HEAD=False`) | Does iterative refinement help, isolated from multi-task effects? |
| A4 | Full proposed model: A3 + auxiliary detection head (`USE_DETECTION_HEAD=True`) | Does the auxiliary detection signal help beyond A3? |

- Every step reports mean ± std macro-F1 across the 5 CV folds, and wall-clock training time per fold (computational cost, per instructions).
- **A component that does not improve macro-F1 beyond its predecessor, or that improves it by less than the practical-significance threshold noted in Section 13, is reported as not useful — it is not dropped from the write-up, it is reported as a negative/neutral finding.**

---

## 13. Statistical Comparison

- **Test:** Wilcoxon signed-rank test on fold-level metric pairs (macro-F1 for subtype, accuracy for detection), comparing each ablation/baseline pair in the hierarchy (A0 vs A1, A1 vs A2, A2 vs A3, A3 vs A4; and A4 vs B1/B2/B3).
- **Correction:** Holm-Bonferroni across the full comparison family run in one analysis pass.
- **Practical-significance threshold:** a macro-F1 delta below 0.5 points is flagged as "statistically significant but practically marginal" if it clears the p-value threshold — the notebook reports both the p-value and the raw delta together, never the p-value alone.
- **Confidence intervals:** report the CV fold-level 95% CI (normal approximation from the 5-fold mean/std, or bootstrap over folds if the coding agent prefers — either must be documented in the output table's column header).

---

## 14. Failure and Robustness Analysis

- **Confusion analysis:** top-3 most-confused subtype pairs from the locked test-set confusion matrix.
- **Low-confidence review:** test predictions with max-softmax <0.5, sample up to 20 per confused pair, saved with image path + predicted/true label to `predictions/low_confidence_review.csv`.
- **Per-magnification breakdown:** subtype macro-F1 computed separately per magnification (40x/100x/200x/400x) on the test set, to check for a magnification-specific weakness.
- **Cross-dataset robustness (IDC):** the locked, test-set-evaluated model's detection head (only) is run once, frozen, on the full IDC dataset. Report AUC/AUPR/accuracy and compare against BreakHis test-set detection numbers — a large drop is reported plainly as a domain-shift limitation, not reframed.
- **Negative-result handling:** if any ablation shows no improvement, if the auxiliary detection head hurts subtype performance, if a minority subtype's F1 collapses despite class weighting, or if IDC generalization fails outright — each is written into `final_report.md` as a first-class finding with its own subsection, not omitted or buried.

---

## 15. Input / Data Visualizations (publication-quality)

All figures: `FIGURE_DPI=300`, saved as both PDF (vector, primary) and PNG (raster fallback), consistent font family/size across all figures (`FIGURE_STYLE` in Config), every figure has a title, axis labels with units where applicable, a legend where needed, and a stated sample count in the caption.

- **Figure A — Dataset Overview:** grid of representative example images, one row per subtype (8 rows), 3–4 columns showing different magnifications of the same subtype where available; each tile captioned with class label, subtype, magnification.
- **Figure B — Dataset Distribution:** two-panel bar plot — (left) image count per subtype, (right) patient count per subtype — both with benign/malignant color-coded, sample counts printed on bars.
- **Figure C — Data Pipeline Illustration:** schematic (built with `matplotlib` boxes/arrows, not a decorative image) showing `Raw image → QC → resize/normalize → augmentation (train only) → model input`.
- **Figure D — Split Visualization:** stacked bar showing train/val/test sample counts per subtype, plus a separate panel explicitly demonstrating patient-level group separation (e.g., a matrix or Venn-style plot showing zero patient overlap across splits, directly reflecting the `leakage_report.json` verdict).

---

## 16. Output / Results Visualizations (publication-quality)

- **Figure E — Main Model Performance:** grouped bar chart, proposed model vs. B1/B2/B3, mean ± std error bars (from 5-fold CV) for macro-F1 (subtype) and accuracy (detection), with the literature benchmark value (96.60%) marked as a reference line, explicitly labeled "published (not directly comparable dataset/protocol)" per Section 11's fair-comparison rule.
- **Figure F — Confusion Matrices:** two separate normalized confusion matrices (subtype 8×8, detection 2×2), test-set predictions only.
- **Figure G — ROC/PR Curves:** binary detection ROC+PR; subtype one-vs-rest ROC for each of the 8 classes on one panel (or small multiples if 8 overlapping curves are unreadable — coding agent's judgment, documented in the caption).
- **Figure H — Learning Curves:** training/validation loss and validation macro-F1 vs. epoch, for the final locked model's full-pool retraining run.
- **Figure I — Ablation Study:** line or bar chart, macro-F1 vs. ablation stage (A0→A4), with error bars and the statistical significance markers from Section 13.
- **Figure J — Explainability:** for 4–6 representative test examples (mix of correct and incorrect predictions across subtypes), show input image, predicted label, true label, and the LCBAM spatial attention overlay plus a Fourier-coefficient frequency-band summary per example.
- **Figure K — Error Analysis:** grid of representative high-confidence misclassifications (from Section 14's low-confidence/confusion review), each captioned with predicted/true label and softmax confidence.

**Explicitly prohibited:** 3D plots, pie charts, decorative-only charts, any figure built only from successful predictions, any axis scaling chosen to visually exaggerate a small delta.

---

## 17. PUBLISHABLE FIGURE PLAN

| Fig | Scientific question | Data used | Plot type | Variables | Expected interpretation | Caveat | Notebook cell | Output filename |
|---|---|---|---|---|---|---|---|---|
| A | What does the raw data look like across classes? | Manifest + sample images | Image grid | subtype × magnification | Establishes visual baseline for morphological differences | Selection of examples must be random-seeded, not cherry-picked | Cell 12 | `fig_A_dataset_overview` |
| B | How imbalanced is the dataset? | Manifest | Grouped bar | class/subtype × count | Motivates class-weighted loss (Sec 10) | Patient count ≠ image count — both shown to avoid overstating diversity | Cell 12 | `fig_B_dataset_distribution` |
| C | What transformations does an image undergo? | Schematic only | Flow diagram | pipeline stages | Documents reproducible pipeline | Illustrative, not data-derived | Cell 12 | `fig_C_pipeline_schematic` |
| D | Is the split leakage-safe? | Split manifest + leakage report | Stacked bar + group-separation panel | split × subtype × patient | Directly evidences the leakage-report PASS verdict | Only as strong as the underlying assertion check (Sec 8) | Cell 9 | `fig_D_split_visualization` |
| E | Does the proposed model beat baselines? | CV fold-level metrics | Grouped bar, error bars | model × macro-F1/accuracy | Primary result figure | Literature reference line is not a fair same-protocol comparison (Sec 11) | Cell 23 | `fig_E_model_performance` |
| F | Where does the model fail, structurally? | Test-set predictions | Normalized confusion matrix ×2 | predicted × true | Identifies systematic subtype confusions | Test set is small for some minority subtypes — cell counts shown | Cell 24 | `fig_F_confusion_matrices` |
| G | How well does the model rank/threshold? | Test-set probabilities | ROC/PR curves | TPR/FPR, precision/recall | Threshold-independent performance view | Multiclass OvR curves can visually flatter minority classes — per-class N noted | Cell 24 | `fig_G_roc_pr_curves` |
| H | Did training behave realistically (no memorization spike, stable convergence)? | Per-epoch history | Line plot | epoch × loss/macro-F1 | Confirms realistic learning dynamics | Single locked run, not averaged across folds | Cell 22 | `fig_H_learning_curves` |
| I | Which components actually help? | Ablation CV results | Bar/line + significance markers | ablation stage × macro-F1 | Directly supports/refutes the novelty claims (Sec 1) | Some deltas may be statistically significant but practically marginal (Sec 13) | Cell 25 | `fig_I_ablation_study` |
| J | Is the model using plausible signal? | 4–6 representative test examples | Image + attention overlay panel | input/attention/prediction | Sanity check against shortcut learning | Small qualitative sample, not a quantitative claim | Cell 27 | `fig_J_explainability` |
| K | What do high-confidence errors look like? | Low-confidence/confusion review set | Image grid with captions | predicted/true/confidence | Concrete failure characterization for the paper's limitations section | Illustrative, not exhaustive | Cell 28 | `fig_K_error_analysis` |

Eleven figures total (A–K), within the "6–10 strong figures" guidance once B/C are optionally merged into a single dataset-characterization panel at the coding agent's discretion — documented either way in `final_report.md`.

---

## 18. Colab Resource Strategy

- Assume Colab Free T4 (16GB) as the baseline target; the notebook must not assume more.
- `MIXED_PRECISION=True` by default on CUDA.
- Local `/content` disk used only for the regenerable image-tensor cache (Section 7) — never for anything in the Drive artifact list.
- Order of execution follows cost: B1/B2 (cheap) → B3 reproduction gate (cheap sanity check) → A0–A3 ablations (moderate) → A4/full model × 3 seeds (most expensive) — so a bad architectural idea is invalidated before the expensive multi-seed run, per the original resource-plan rationale.
- If a training cell exceeds a reasonable single-session time budget, the checkpoint/resume mechanism (Section 9.3) allows splitting a run across multiple Colab sessions without losing progress.

---

## 19. Error-Resilient Design and Fallback Handling

- **GPU check:** first executable cell reports `torch.cuda.is_available()`, GPU name if present, and falls back to `Config["DEVICE"]="cpu"` with a clear warning (and a note that training time will increase substantially) rather than failing.
- **Dataset existence checks:** every download/extraction step verifies output before proceeding (Section 3.4); a missing or empty dataset halts with a specific, actionable message.
- **Corrupted-file handling:** logged and excluded, never a full-run crash (Section 7).
- **RAM/resource warnings:** before caching the full image set to `/content`, estimate size and compare against typical Colab Free local-disk limits; warn if projected usage is high.
- **CUDA OOM handling:** wrap the training step in a `try/except torch.cuda.OutOfMemoryError`; on catch, halve `Config["BATCH_SIZE"]`, double `Config["GRAD_ACCUM_STEPS"]` to preserve the effective batch size, clear the CUDA cache, and retry the epoch — log this adjustment explicitly in `run_log.txt`, never silently.
- **Optional-package fallback:** if any non-core package fails to import (e.g., a specific `imagehash` backend), catch the import error, degrade to the best available alternative if one exists, and log the degradation — never silently skip the check it was meant to perform.
- **No silent failure, no fake outputs:** any stage that cannot complete (e.g., the B3 reproduction gate failing its 3-point tolerance check) **stops downstream dependent stages** and reports the specific failure, likely cause, and the documented fallback (Section 0's table / Section 11's B3 gate) rather than substituting a placeholder number.
- **"Do not fix research problems by silently changing the experimental protocol"** — e.g., if patient-grouped stratification cannot achieve a target ratio for a minority subtype, the notebook does not quietly fall back to ungrouped splitting; it reports the achieved ratio, flags the affected class, and proceeds with the best achievable patient-grouped split, exactly as scoped in the earlier plan's risk table.

---

## 20. Reproducibility Record (`environment.json`)

```json
{
  "run_id": "...",
  "seed": 42,
  "deterministic_mode": true,
  "python_version": "...",
  "torch_version": "...",
  "cuda_version": "... or null",
  "gpu_name": "... or 'cpu'",
  "package_versions": {"torchvision": "...", "timm": "...", "scikit-learn": "...", "imagehash": "..."},
  "dataset_sources": {"breakhis_kaggle_slug": "...", "idc_kaggle_slug": "..."},
  "config_snapshot_file": "config.json"
}
```
Written immediately after Drive mount and before any data download, alongside `config.json` (the full resolved `Config` object as JSON, credentials excluded by construction since they are never stored in `Config`).

---

## 21. Exact Notebook Cell Structure

1. Title / research question (Section 1, restated).
2. Environment setup — package checks, installs only if missing, GPU check (Section 19).
3. Imports.
4. `Config` object definition (Section 5) — single source of truth.
5. Drive mount + `RUN_ID` + run-directory creation (Section 4) + `environment.json` + `config.json` written.
6. Kaggle credential loading (Section 3.3) — no values printed/logged.
7. Dataset download + extraction + root discovery/verification (Section 3.4) for BreakHis.
8. Dataset download + extraction + root discovery/verification for IDC.
9. Manifest construction (BreakHis) + QC (corrupt/exact-dup/near-dup) + `leakage_report.json` groundwork.
10. Patient-grouped split creation + leakage assertions (Section 8) — **halts here if `verdict=="FAIL"`**.
11. Preprocessing + augmentation pipeline definitions (Section 7).
12. Exploratory visualization — Figures A, B, C (Section 15/17).
13. Figure D — split visualization (depends on cell 10's manifest).
14. `Dataset`/`DataLoader` classes, `num_workers=0` default (Section 6).
15. Model architecture definitions — backbone wrapper, FKAN+LCBAM block, weight-tied refinement stage, dual-head model class, config-toggleable ablation variants (A0–A4) and B1–B3 variants.
16. Loss functions (Section 10).
17. Shared training utilities — epoch loop, early stopping, checkpointing (with resume detection), AMP wrapper, OOM auto-recovery (Section 19).
18. Train B1, B2 — 5-fold CV.
19. Train B3 — reproduction gate; explicit check against the 96.60% literature figure; **halts subsequent proposed-model cells if the gate fails**, per Section 19.
20. Train A0–A3 ablations — 5-fold CV.
21. Train A4/full proposed model — 5-fold CV × 3 seeds.
22. Model selection (mean CV val macro-F1) + lock configuration + retrain locked config on full 85% pool + Figure H (learning curves of this final run).
23. Final, single evaluation on the frozen test set + Figure E.
24. Confusion matrices + ROC/PR — Figures F, G.
25. Statistical comparison (CV fold-level) — Figure I + `statistical_comparison.csv`.
26. Cross-dataset (IDC) robustness evaluation of the detection head.
27. Explainability — Figure J.
28. Failure/error analysis — Figure K + `low_confidence_review.csv`.
29. Master results table (all models × all metrics, CV and final-test clearly separated columns) exported to `tables/`.
30. `final_report.md` generation — restates the research question, the benchmark caveats, every ablation result including negative ones, the final test numbers, and the IDC robustness finding, without omission.
31. Final artifact save/verification cell — confirms every file in Section 4.1's structure exists in the Drive run directory and prints a manifest of what was saved.

---

## 22. Exact Output Artifacts

`config.json`, `environment.json`, `dataset_manifest.csv`, `split_manifest.csv`, `class_distribution.csv`, `leakage_report.json`, `corrupt_files.csv`, `dedup_report.json`, `training_history/{model}_fold{k}_seed{s}.csv`, `checkpoints/{model}_fold{k}_seed{s}_{best|final}.pt`, `predictions/{model}_test_predictions.csv`, `predictions/low_confidence_review.csv`, `figures/fig_{A–K}_*.{pdf,png}`, `tables/baseline_results.csv`, `tables/ablation_results.csv`, `tables/statistical_comparison.csv`, `tables/final_test_results.csv`, `reports/final_report.md`, `logs/run_log.txt`.

---

## ASSUMPTIONS REQUIRING USER CONFIGURATION

These are the only decisions this specification cannot make on the user's behalf; every other decision in the notebook is fixed by the sections above.

1. **Exact Kaggle dataset slugs** — the defaults given (`ambarish/breakhis`, `paultimothymooney/breast-histopathology-images`) are the most common public mirrors at the time of writing but must be verified by the user against their own Kaggle account access before the first run; update `Config["KAGGLE_DATASET_BREAKHIS"]` / `Config["KAGGLE_DATASET_IDC"]` if different.
2. **Kaggle credentials** — must be supplied by the user via Colab Secrets or a manually uploaded `kaggle.json`; cannot be preset.
3. **`DATA_ROOT_OVERRIDE`** — only needed if the automatic root-discovery step (Section 3.4) fails to locate the expected `benign/`/`malignant/` structure; the user sets this after inspecting the printed directory tree.
4. **GPU availability at runtime** — the notebook adapts (Section 19) but cannot guarantee a T4-class GPU is actually allocated by Colab in a given session; expensive stages (A4, final retrain) may need to be rerun across sessions if only CPU is allocated.
5. **`num_workers` opt-in** — remains 0 unless the user has personally verified `num_workers>0` is stable on their specific runtime via the documented opt-in test cell (Section 6); this specification does not enable it by default.
6. **Backbone choice under memory pressure** — `densenet201` is the default; if OOM persists even after the auto-recovery in Section 19, the user should manually set `Config["BACKBONE"]="efficientnet_b0"`.
7. **Whether Figures B and C are merged into one panel** — left to the coding agent's layout judgment (Section 17), documented in `final_report.md` either way.

---

## FINAL AGENT HANDOFF INSTRUCTION

Build `breast_cancer_research.ipynb` exactly from this specification, in the exact cell order given in Section 21. Do not introduce `num_workers>0`, multiprocessing, or distributed training by default. Do not implement a true implicit-differentiation DEQ solver — implement the weight-tied fixed-iteration refinement block specified in Section 2/Section 0's audit. Do not add a consistency-loss term between the detection and subtype heads. Do not proceed past a `leakage_report.json` `FAIL` verdict or a failed B3 reproduction gate. Do not touch the frozen test set for any purpose other than the single final evaluation in Section 9.1/Cell 23. Every configuration value must come from the single `Config` object in Section 5. Every artifact in Section 22 must be written to the Google Drive run directory structure in Section 4.1. Report every ablation and robustness result — including negative ones — in `final_report.md` without omission or reframing.
