# OMNet-V3 — Final Research Paper Master Requirements & Evidence Checklist

**Document status:** Master checklist / evidence specification  
**Study status:** Pre-experiment research agenda; no experimental results are assumed  
**Primary dataset:** BreaKHis v1  
**Primary model:** EfficientNet-B0 + ViT-Tiny + Magnification-Aware Adaptive Fusion + hierarchical heads  
**Primary evaluation:** Patient/specimen-disjoint 5-fold cross-validation  

> **Rule:** Nothing in this document is a result. Any `[TBD]`, `[TO COLLECT]`, or `[TO VERIFY]` item must be completed from the actual implementation, experiment logs, saved checkpoints, generated figures, or verified literature before the final paper is written.

---

## 0. Evidence-status convention

Every item is classified in one of three ways:

- **Required Information** — facts, configuration, definitions, or methodological description that the final paper must contain.
- **Expected Result** — the scientific behavior the experiment is intended to test; **not a claim that the result will occur**.
- **Actual Result to Collect** — the measured value, figure, table, statistical output, or implementation evidence that must be obtained before a conclusion can be written.

Use these labels consistently in experiment notes and the final manuscript.

---

# 1. Overall research problem

## 1.1 Research problem statement

**Required Information**

- [ ] State the task as automated breast histopathology image classification.
- [ ] State that the study addresses both benign/malignant discrimination and eight-class subtype classification.
- [ ] Explain that diagnostically relevant information exists at multiple visual scales, including tissue architecture and cellular/nuclear morphology.
- [ ] Explain that BreaKHis images are available at 40×, 100×, 200×, and 400× magnifications.
- [ ] Explain that image appearance can vary because of staining/acquisition differences.
- [ ] Establish the need for evaluation that is resistant to patient/specimen leakage.
- [ ] Keep the scope at college-project level: classification and controlled robustness analysis, not clinical deployment.

**Expected Result**

- [ ] A model that combines local and global visual information should be better positioned to handle multi-scale histopathology patterns than either representation alone.
- [ ] Explicit magnification conditioning is expected to help the model adapt fusion behavior to known imaging scale.
- [ ] Patient-disjoint evaluation is expected to produce a more realistic estimate of generalization than image-level random splitting.

**Actual Result to Collect**

- [ ] Final problem statement used in the manuscript: `[FINAL TEXT]`
- [ ] Exact experimental task definitions: `[TO COLLECT]`
- [ ] Any scope limitations agreed by the project team: `[TO COLLECT]`

---

# 2. Research objective, gap, proposed solution, hypotheses, success criteria

## 2.1 Research objective

**Required Information**

- [ ] Define the primary objective as testing whether a lightweight CNN–ViT model with magnification-conditioned adaptive fusion improves patient-disjoint eight-class BreaKHis classification compared with appropriate baselines.
- [ ] Define secondary objectives: hierarchy, magnification behavior, stain robustness, patient-level performance, interpretability, and computational cost.

**Actual Result to Collect**

- [ ] Final objective wording: `[TO COLLECT]`

## 2.2 Research gap

**Required Information**

The final paper must establish, from the literature review, whether the following combination remains insufficiently studied:

- [ ] lightweight CNN local representation;
- [ ] lightweight ViT global representation;
- [ ] explicit magnification metadata conditioning;
- [ ] adaptive local/global fusion rather than simple concatenation;
- [ ] hierarchical benign/malignant + subtype objectives;
- [ ] patient/specimen-disjoint evaluation;
- [ ] controlled stain/domain robustness evaluation.

**Critical rule**

- [ ] Do **not** claim “first,” “novel,” or “no prior work” until the literature matrix has been completed.
- [ ] Phrase the final gap as the absence of identified prior work covering the specific tested combination/protocol, if that is what the completed review supports.

**Actual Result to Collect**

- [ ] Literature matrix with at least: paper, dataset, CNN, ViT, hybrid, magnification conditioning, hierarchy, split protocol, robustness evaluation, principal result.
- [ ] `[TO VERIFY]` exact overlap with the closest competing methods.
- [ ] `[TO COLLECT]` final defensible gap statement.

## 2.3 Proposed solution

**Required Information**

- [ ] EfficientNet-B0 branch for local morphological representation.
- [ ] ViT-Tiny branch for global contextual representation.
- [ ] 256-D projection of each branch.
- [ ] Four-level learnable magnification embedding.
- [ ] Magnification-Aware Adaptive Fusion (MAF).
- [ ] Residual feature enrichment.
- [ ] Binary benign/malignant head.
- [ ] Eight-class subtype head.
- [ ] Weighted joint objective with consistency term.
- [ ] Patient-disjoint five-fold evaluation.
- [ ] Online Macenko stain perturbation at the locked probability, unless implementation explicitly changes it.

## 2.4 Hypotheses / research questions

**Required Information**

### H1 — Local/global fusion
- [ ] Test whether the shared CNN + ViT fusion outperforms individual CNN and ViT branches on eight-class classification.

### H2 — Magnification awareness
- [ ] Test whether magnification-conditioned fusion outperforms the equivalent magnification-agnostic fusion.

### H3 — Hierarchical learning
- [ ] Test whether the joint binary/subtype objective improves subtype classification and/or hierarchical consistency versus the flat subtype objective.

### H4 — Stain robustness
- [ ] Test whether stain augmentation reduces performance degradation under controlled stain perturbation relative to no stain augmentation.

### Evaluation protocol statement
- [ ] Treat patient/specimen-disjoint validation as an evaluation design requirement, not as a model hypothesis.

**Actual Result to Collect**

- [ ] Hypothesis-by-hypothesis result table: hypothesis, comparison, primary metric, observed difference, uncertainty, supported/not supported.

## 2.5 What constitutes success

**Required Information**

Success is not defined as raw BreaKHis accuracy alone. The final study should seek evidence for:

- [ ] statistically/uncertainty-supported improvement on the primary metric relative to appropriate baselines;
- [ ] strong patient-level performance;
- [ ] stable per-class performance rather than domination by the largest class;
- [ ] meaningful magnification-wise behavior;
- [ ] measurable contribution of the adaptive fusion mechanism;
- [ ] interpretable differences or complementary behavior between CNN and ViT branches;
- [ ] reduced degradation under controlled stain/domain shift, if that experiment is completed;
- [ ] reproducible results under the stated patient-disjoint protocol;
- [ ] acceptable computational cost for the intended college-project setting.

**Actual Result to Collect**

- [ ] One final success/failure table tied directly to the research hypotheses.

---

# 3. Abstract requirements

## 3.1 Background

**Required Information**

- [ ] One concise statement of breast histopathology classification difficulty.
- [ ] Multi-scale/magnification challenge.
- [ ] Local CNN vs global ViT motivation.

## 3.2 Objective

- [ ] State the precise primary research objective.

## 3.3 Methods

- [ ] BreaKHis v1.
- [ ] Number of images and patients: `[TO VERIFY / COLLECT FROM ACTUAL DATASET]`.
- [ ] Four magnifications.
- [ ] Patient-disjoint five-fold validation.
- [ ] EfficientNet-B0 + ViT-Tiny + MAF.
- [ ] Hierarchical classification.
- [ ] Stain augmentation.
- [ ] Primary evaluation metric.

## 3.4 Results

**Actual Result to Collect**

- [ ] Binary accuracy / macro-F1: `[TBD]`
- [ ] 8-class accuracy / macro-F1: `[TBD]`
- [ ] Primary confidence interval: `[TBD]`
- [ ] Best/mean fold performance as defined by the final reporting policy: `[TBD]`
- [ ] Patient-level primary performance: `[TBD]`
- [ ] Robustness result, if tested: `[TBD]`

## 3.5 Conclusion

- [ ] Only evidence-supported claims.
- [ ] No clinical-readiness claim.
- [ ] No superiority claim unless supported by the experimental comparison.

---

# 4. Introduction and motivation

## 4.1 Clinical/technical background

**Required Information**

- [ ] Explain histopathology classification as a computer-aided analysis problem.
- [ ] Explain benign/malignant and subtype recognition.
- [ ] Introduce H&E-stained tissue imagery.
- [ ] Explain why morphology exists at multiple scales.
- [ ] Explain why magnification is relevant in BreaKHis.

## 4.2 CNN motivation

**Required Information**

- [ ] Local feature extraction.
- [ ] Hierarchical visual representations.
- [ ] Transfer-learning efficiency.
- [ ] Limitation relevant to the project: limited direct modeling of long-range/global relationships.

## 4.3 ViT motivation

**Required Information**

- [ ] Patch-based representation.
- [ ] Self-attention/global contextual relationships.
- [ ] Motivation for complementing CNN features.
- [ ] Caveat: performance depends on training regime, data, and pretraining.

## 4.4 Why simple hybrids are insufficient for this objective

**Required Information**

- [ ] Literature evidence that CNN+ViT hybrids already exist.
- [ ] Explain that naive concatenation/fixed fusion does not explicitly condition the local/global balance on known magnification.
- [ ] Explain that a hybrid alone does not establish robustness or leakage-resistant evaluation.
- [ ] Avoid implying that all existing hybrid approaches are inadequate generally; restrict the critique to this project's objective.

**Actual Result to Collect**

- [ ] Literature evidence supporting each limitation: `[TO COLLECT]`

---

# 5. Literature review and research-gap evidence

## 5.1 CNN-based BreaKHis work

- [ ] Papers using CNNs on BreaKHis.
- [ ] Architectures used.
- [ ] Transfer learning strategies.
- [ ] Split protocols.
- [ ] Metrics.
- [ ] Reported limitations.

## 5.2 ViT/Transformer work

- [ ] ViT and related Transformer studies relevant to breast histopathology.
- [ ] Pretraining and optimization assumptions.
- [ ] BreaKHis-specific results where comparable.
- [ ] Evaluation protocol.

## 5.3 CNN–ViT hybrids

- [ ] Existing fusion mechanisms.
- [ ] Concatenation vs learned/adaptive fusion.
- [ ] Whether magnification metadata is explicitly conditioned.
- [ ] Whether hierarchy is used.
- [ ] Patient-disjoint evaluation.

## 5.4 Magnification/multi-scale approaches

- [ ] Independent per-magnification models.
- [ ] Multi-scale concatenation.
- [ ] Scale-aware conditioning.
- [ ] Multi-branch approaches.
- [ ] Exact overlap with MAF.

## 5.5 Hierarchical classification

- [ ] Relevant benign/malignant + subtype learning approaches.
- [ ] Hierarchical consistency approaches.
- [ ] Whether the hierarchy is imposed architecturally, through loss, or both.

## 5.6 Validation/leakage

- [ ] Patient-level vs image-level splitting in prior studies.
- [ ] Evidence that same-patient images can create optimistic estimates if split across partitions.
- [ ] Comparability rules for literature benchmarks.

## 5.7 Stain/domain robustness

- [ ] Stain normalization/augmentation literature.
- [ ] Macenko method and exact role in this study.
- [ ] Internal stain perturbation vs external-domain evaluation distinction.

## 5.8 Literature comparison table

**Actual Result to Collect**

- [ ] Final comparison matrix with at least 10 relevant/closest papers if available.
- [ ] `[TBD]` closest prior method.
- [ ] `[TBD]` strongest methodological overlap.
- [ ] `[TBD]` unresolved gap supported by the literature.
- [ ] `[TBD]` final novelty wording.

---

# 6. BreaKHis dataset and data characteristics

## 6.1 Core dataset facts

**Required Information**

- [ ] Dataset: BreaKHis v1.
- [ ] Total images: 7,909 (verify from the actual downloaded dataset and source).
- [ ] Patients: 82 (verify).
- [ ] Benign patients: 24 (verify).
- [ ] Malignant patients: 58 (verify).
- [ ] Four magnifications: 40×, 100×, 200×, 400×.
- [ ] RGB PNG images.
- [ ] Nominal image resolution: 700×460.
- [ ] Surgical Open Biopsy (SOB).

## 6.2 Eight subtype labels

### Benign
- [ ] Adenosis (A)
- [ ] Fibroadenoma (F)
- [ ] Phyllodes Tumor (PT)
- [ ] Tubular Adenoma (TA)

### Malignant
- [ ] Ductal Carcinoma (DC)
- [ ] Lobular Carcinoma (LC)
- [ ] Mucinous Carcinoma (MC)
- [ ] Papillary Carcinoma (PC)

## 6.3 Actual dataset distribution

**Actual Result to Collect**

- [ ] Image count by subtype.
- [ ] Patient count by subtype, if recoverable from metadata.
- [ ] Image count by magnification.
- [ ] Patient count represented at each magnification.
- [ ] Benign/malignant counts at image level.
- [ ] Benign/malignant counts at patient level.
- [ ] Any duplicate/corrupt/missing files found.

## 6.4 Filename/metadata parsing

**Required Information**

Document exactly how the implementation extracts:

- [ ] biopsy procedure;
- [ ] tumor class;
- [ ] subtype;
- [ ] year;
- [ ] slide/patient identifier;
- [ ] magnification;
- [ ] sequence number.

**Actual Result to Collect**

- [ ] Example filename and parser output.
- [ ] Number of successfully parsed files.
- [ ] Number of parsing failures: `[TBD]`.
- [ ] Final patient-ID extraction rule.

---

# 7. Why BreaKHis is suitable and what challenges it introduces

## 7.1 Suitability

**Required Information**

- [ ] Publicly documented histopathology dataset.
- [ ] Eight pathological subtypes.
- [ ] Benign/malignant hierarchy.
- [ ] Four known magnifications.
- [ ] Multiple patients and multiple images per patient.
- [ ] Sufficient size for a controlled student-scale transfer-learning experiment.

## 7.2 H&E image challenges

**Required Information**

- [ ] Staining variability / appearance variation.
- [ ] Cellular and tissue morphology at different scales.
- [ ] Class imbalance.
- [ ] Visually similar subtypes where applicable.
- [ ] Potential acquisition/domain differences.

**Actual Result to Collect**

- [ ] Representative image panel showing all four magnifications.
- [ ] Representative benign and malignant examples.
- [ ] Representative minority-class examples.
- [ ] Optional visual examples of stain variation.

## 7.3 Class hierarchy challenge

- [ ] Explicit benign vs malignant superclass.
- [ ] Eight subtype labels.
- [ ] Mapping of each subtype to superclass documented in code and paper.

## 7.4 Magnification challenge

- [ ] State that magnification is known metadata.
- [ ] State that the model does not infer magnification from pixels.
- [ ] Explain why different magnifications can expose different morphological information.

## 7.5 Imbalance challenge

- [ ] State exact class frequencies.
- [ ] Explain why macro-F1/balanced accuracy are necessary.
- [ ] Record training-fold class weights.

## 7.6 Domain/stain challenge

- [ ] Separate internal stain variation from true external-domain generalization.
- [ ] Use “stain robustness” unless a genuine domain-shift experiment is completed.

---

# 8. Dataset split and patient/specimen-disjoint validation

## 8.1 Primary split protocol

**Required Information**

- [ ] StratifiedGroupKFold.
- [ ] 5 folds.
- [ ] Group variable = patient ID.
- [ ] Stratification label = eight-class subtype, subject to feasibility.
- [ ] Every patient appears in exactly one test fold.
- [ ] No patient appears in both training and test within a fold.
- [ ] 10% of the training patients are reserved for validation within each fold.

## 8.2 Split integrity checks

**Actual Result to Collect**

For every fold:

- [ ] train patient count;
- [ ] validation patient count;
- [ ] test patient count;
- [ ] train image count;
- [ ] validation image count;
- [ ] test image count;
- [ ] class distribution in each partition;
- [ ] magnification distribution in each partition;
- [ ] intersection of train/validation/test patient IDs = empty;
- [ ] script output proving no patient leakage.

## 8.3 Leakage evidence

- [ ] Save fold assignments.
- [ ] Save a machine-readable patient-to-fold mapping.
- [ ] Save an automated assertion/check that no patient crosses partitions.
- [ ] Include a short leakage-prevention statement in the manuscript.

## 8.4 Validation leakage safeguards

- [ ] Compute class weights using training data only.
- [ ] Fit any learned preprocessing/statistical normalization only from training data if applicable.
- [ ] Never use test labels to tune hyperparameters.
- [ ] Do not choose the final model by looking at test performance.

---

# 9. Preprocessing and augmentation

## 9.1 Input pipeline

**Required Information**

### Training
- [ ] Resize to 256×256.
- [ ] Random resized crop to 224×224.
- [ ] Random horizontal flip.
- [ ] Random vertical flip.
- [ ] Random 90° rotation.
- [ ] Online stain augmentation.
- [ ] ImageNet normalization.

### Validation/test
- [ ] Resize to 256×256.
- [ ] Center crop to 224×224.
- [ ] ImageNet normalization.
- [ ] No stochastic train-only augmentation at evaluation.

## 9.2 Exact augmentation configuration

**Actual Result to Collect**

- [ ] Crop scale range.
- [ ] Crop aspect-ratio range.
- [ ] Horizontal flip probability.
- [ ] Vertical flip probability.
- [ ] Rotation probability / implementation details.
- [ ] Macenko application probability = 0.5 unless implementation changes.
- [ ] Exact Macenko perturbation parameter ranges.
- [ ] Exact interpolation/resampling method.
- [ ] Exact normalization mean/std values.

## 9.3 Stain augmentation

**Required Information**

- [ ] Optical density transformation.
- [ ] H&E stain estimation.
- [ ] Random perturbation of stain concentrations/matrix according to actual implementation.
- [ ] Reconstruction procedure.
- [ ] Probability.

**Actual Result to Collect**

- [ ] Exact implementation/library.
- [ ] Version.
- [ ] Parameter ranges.
- [ ] Example original/augmented images.

---

# 10. Proposed OMNet-V3 architecture

## 10.1 Architecture diagram

**Actual Result to Collect**

- [ ] Final publication-quality architecture figure.
- [ ] Tensor dimensions annotated at every major stage.
- [ ] CNN branch.
- [ ] ViT branch.
- [ ] Magnification embedding.
- [ ] MAF gate.
- [ ] Residual enrichment.
- [ ] Binary head.
- [ ] Subtype head.

## 10.2 EfficientNet-B0 branch

**Required Information**

- [ ] ImageNet pretrained.
- [ ] Classification head removed.
- [ ] Global average pooling.
- [ ] 1280-D feature representation.
- [ ] Role: local morphological feature extraction.

**Actual Result to Collect**

- [ ] Exact model/library implementation.
- [ ] Pretrained weights identifier.
- [ ] Number of parameters.
- [ ] Trainable parameters during each training stage.
- [ ] Any implementation deviations.

## 10.3 ViT-Tiny branch

**Required Information**

- [ ] ViT-Tiny.
- [ ] 16×16 patch size.
- [ ] ImageNet pretrained.
- [ ] Classification head removed.
- [ ] 192-D CLS representation.

**Actual Result to Collect**

- [ ] Exact model/library implementation.
- [ ] Exact pretrained weights identifier.
- [ ] Parameter count.
- [ ] Any implementation deviations.

## 10.4 Feature projections

**Required Information**

- [ ] CNN: 1280→256 + LayerNorm.
- [ ] ViT: 192→256 + LayerNorm.

## 10.5 Magnification embedding

**Required Information**

- [ ] `Embedding(4,64)`.
- [ ] Explicit mapping of `{40,100,200,400}` to embedding indices.

**Actual Result to Collect**

- [ ] Exact index mapping used in code.

## 10.6 Magnification-Aware Adaptive Fusion

**Required Information**

Gate input:

- [ ] CNN 256-D feature;
- [ ] ViT 256-D feature;
- [ ] magnification 64-D embedding;
- [ ] concatenated input dimension = 576.

Locked formulation:

\[
\alpha = \sigma(W[f_{cnn};f_{vit};e_m])
\]

\[
f_{fused}=\alpha f_{cnn} + (1-\alpha)f_{vit}
\]

**Actual Result to Collect**

- [ ] Exact gate implementation.
- [ ] Gate parameter count.
- [ ] Whether a bias term is used.
- [ ] Alpha tensor shape.
- [ ] Alpha distribution over test/validation examples.

## 10.7 Residual enrichment

**Required Information**

- [ ] Concatenate fused feature + magnification embedding.
- [ ] 320→256 linear layer.
- [ ] GeLU.
- [ ] Dropout 0.3.
- [ ] 256→256 linear layer.
- [ ] Residual addition to fused feature.

**Actual Result to Collect**

- [ ] Exact implementation and parameter count.
- [ ] Confirmation that residual dimensions match.

## 10.8 Classification heads

**Required Information**

- [ ] Binary head: 256→2.
- [ ] Subtype head: 256→8.

**Actual Result to Collect**

- [ ] Exact activation/output interpretation.
- [ ] Loss function implementation.

---

# 11. Mathematical formulation of hierarchy and losses

## 11.1 Binary objective

**Required Information**

- [ ] Define the binary target mapping.
- [ ] Define exact loss function: `[CE or BCE — TO VERIFY FROM IMPLEMENTATION]`.
- [ ] Define class weighting formula.

**Actual Result to Collect**

- [ ] Exact equation used by code.
- [ ] Fold-specific weight values.

## 11.2 Subtype objective

**Required Information**

- [ ] Eight-class weighted cross-entropy.
- [ ] Class weights derived from training data only.

## 11.3 Subtype-to-binary marginalization

**Required Information**

Define:

\[
P(B)=P(A)+P(F)+P(PT)+P(TA)
\]

\[
P(M)=P(DC)+P(LC)+P(MC)+P(PC)
\]

Then specify how the marginalized subtype distribution is compared with the binary head.

**Actual Result to Collect**

- [ ] Exact consistency-loss formulation.
- [ ] Confirm whether KL direction is fixed or symmetric.
- [ ] Confirm numerical stabilization procedure if used.

## 11.4 Total loss

Locked intended form:

\[
L_{total}=0.3L_{binary}+0.6L_{subtype}+0.1L_{consistency}
\]

**Actual Result to Collect**

- [ ] Exact code equation.
- [ ] Loss values logged per epoch/fold.
- [ ] Relative magnitude of each loss component, if logged.

---

# 12. Training configuration

## 12.1 Locked configuration

**Required Information**

| Setting | Intended value | Actual value |
|---|---|---|
| Optimizer | AdamW | `[TBD]` |
| Backbone LR | 1e-4 | `[TBD]` |
| Head LR | 5e-4 | `[TBD]` |
| Weight decay | 1e-4 | `[TBD]` |
| Scheduler | cosine + 5-epoch warmup | `[TBD]` |
| Batch size | 32 subject to VRAM | `[TBD]` |
| Max epochs | 50 | `[TBD]` |
| Early stopping | patience 10 | `[TBD]` |
| Monitor | validation subtype macro-F1 | `[TBD]` |
| AMP | FP16 | `[TBD]` |
| Gradient clipping | 1.0 | `[TBD]` |
| Seed | 42 | `[TBD]` |
| Progressive unfreezing | planned | `[TBD exact schedule]` |

## 12.2 Exact training schedule

**Actual Result to Collect**

- [ ] Frozen/unfrozen components by epoch.
- [ ] Warmup implementation.
- [ ] Scheduler implementation.
- [ ] LR after warmup.
- [ ] Number of epochs actually trained per fold.
- [ ] Best epoch per fold.
- [ ] Early-stopping epoch per fold.
- [ ] Training time per fold.
- [ ] Peak VRAM/RAM usage.
- [ ] Training hardware.

## 12.3 Training curves

**Actual Result to Collect**

- [ ] Train loss curve.
- [ ] Validation loss curve.
- [ ] Train macro-F1 curve.
- [ ] Validation macro-F1 curve.
- [ ] Optional binary metrics per epoch.
- [ ] Learning-rate curve.

---

# 13. Baseline and comparison requirements

## 13.1 Required baseline models

The final study should include controlled baselines sufficient to isolate the claimed contribution:

- [ ] EfficientNet-B0 only.
- [ ] ViT-Tiny only.
- [ ] CNN + ViT simple concatenation.
- [ ] CNN + ViT fixed fusion, if implemented.
- [ ] OMNet-V3 full model.

## 13.2 Required controlled comparisons

- [ ] Same patient folds.
- [ ] Same preprocessing wherever applicable.
- [ ] Same evaluation metrics.
- [ ] Same early-stopping policy where feasible.
- [ ] Same class-weighting policy.
- [ ] Same seed policy.
- [ ] Differences in model capacity explicitly recorded.

## 13.3 Literature comparison

**Required Information**

Only compare numerical results when:

- [ ] dataset version matches;
- [ ] task definition matches;
- [ ] split protocol is sufficiently comparable;
- [ ] metric definition is comparable.

**Actual Result to Collect**

- [ ] Table of comparable published results: `[TBD]`
- [ ] Explicit protocol mismatch notes where direct comparison is invalid.

---

# 14. Ablation evidence

## 14.1 Minimum ablation matrix

| Variant | CNN | ViT | Simple/Fixed/Adaptive fusion | Magnification | Hierarchy | Stain augmentation |
|---|---:|---:|---|---:|---:|---:|
| CNN baseline | ✓ | | — | | | |
| ViT baseline | | ✓ | — | | | |
| Simple hybrid | ✓ | ✓ | Concatenation | | | |
| Fixed fusion | ✓ | ✓ | Fixed | | | |
| No magnification | ✓ | ✓ | Adaptive | | ✓ | |
| No hierarchy | ✓ | ✓ | Adaptive | ✓ | | |
| No stain augmentation | ✓ | ✓ | Adaptive | ✓ | ✓ | |
| **Full OMNet-V3** | ✓ | ✓ | Adaptive | ✓ | ✓ | ✓ |

## 14.2 Actual ablation results to collect

For each variant:

- [ ] mean fold macro-F1;
- [ ] standard deviation;
- [ ] confidence interval if selected for primary comparison;
- [ ] accuracy;
- [ ] balanced accuracy;
- [ ] MCC;
- [ ] macro-AUC where valid;
- [ ] patient-level metrics;
- [ ] training time;
- [ ] parameter count if architecture changes;
- [ ] statistical comparison against the reference model, where planned.

## 14.3 Critical ablations

- [ ] Adaptive vs fixed fusion.
- [ ] Magnification-aware vs magnification-agnostic fusion.
- [ ] Hierarchical vs flat classification.
- [ ] Stain augmentation vs no augmentation.

---

# 15. Main quantitative evaluation

## 15.1 Primary metric

**Required Information**

- [ ] Select one primary endpoint before looking at final test results; intended candidate = eight-class macro-F1.
- [ ] Define why the selected primary endpoint is appropriate for class imbalance.

**Actual Result to Collect**

- [ ] Mean 5-fold primary score: `[TBD]`
- [ ] Fold 1–5 scores: `[TBD]`
- [ ] SD: `[TBD]`
- [ ] 95% CI: `[TBD]`

## 15.2 Secondary metrics

**Actual Result to Collect**

For binary and eight-class tasks as applicable:

- [ ] Accuracy.
- [ ] Macro-F1.
- [ ] Weighted-F1.
- [ ] Balanced accuracy.
- [ ] Macro-AUC, one-vs-rest.
- [ ] MCC.
- [ ] Precision.
- [ ] Recall/sensitivity.
- [ ] Specificity for binary task.
- [ ] Per-class precision/recall/F1.

## 15.3 Fold-level results

- [ ] Complete fold-level results, not only the mean.
- [ ] Fold patient counts and image counts.
- [ ] Fold best epoch.
- [ ] Fold checkpoint identifier.

---

# 16. Per-class results and confusion matrices

## 16.1 Per-class quantitative results

**Actual Result to Collect**

For all eight subtypes:

- [ ] support/image count;
- [ ] precision;
- [ ] recall;
- [ ] F1;
- [ ] one-vs-rest AUC if included;
- [ ] fold variability.

## 16.2 Confusion matrices

- [ ] Binary confusion matrix.
- [ ] Eight-class confusion matrix.
- [ ] Prefer normalized and raw-count versions if space permits.
- [ ] Patient-level confusion matrix where patient-level subtype aggregation is performed.

## 16.3 Error concentration

- [ ] Identify largest confusion pairs.
- [ ] Identify minority classes with unstable performance.
- [ ] Quantify whether errors are concentrated at specific magnifications.

---

# 17. Magnification-wise analysis

## 17.1 Required groups

- [ ] 40×
- [ ] 100×
- [ ] 200×
- [ ] 400×

## 17.2 Quantitative results

**Actual Result to Collect**

For every magnification:

- [ ] sample count;
- [ ] patient count;
- [ ] accuracy;
- [ ] macro-F1;
- [ ] balanced accuracy;
- [ ] MCC;
- [ ] per-class results where sample sizes permit.

## 17.3 Magnification-conditioning experiment

- [ ] Compare full MAF with magnification-agnostic equivalent.
- [ ] Keep all other factors fixed.

**Actual Result to Collect**

- [ ] Performance difference at each magnification.
- [ ] Aggregate improvement.
- [ ] Uncertainty of the difference.

## 17.4 Magnification distribution figure/table

- [ ] Dataset magnification distribution table.
- [ ] Performance-vs-magnification plot.

---

# 18. CNN/ViT fusion-gate analysis

## 18.1 Alpha collection

**Actual Result to Collect**

- [ ] Alpha value for each evaluated image.
- [ ] Patient identifier associated with each alpha.
- [ ] Magnification.
- [ ] Predicted subtype.
- [ ] True subtype.
- [ ] Correct/incorrect indicator.
- [ ] Confidence/probability if logged.

## 18.2 Aggregate gate analysis

- [ ] Mean alpha by magnification.
- [ ] Median alpha by magnification.
- [ ] Standard deviation/IQR.
- [ ] Alpha distributions by magnification.
- [ ] Alpha distributions for correct vs incorrect predictions.

## 18.3 Interpretation requirement

- [ ] Do not equate alpha directly with “percentage of information” from a branch.
- [ ] Describe alpha as the learned scalar fusion coefficient unless stronger evidence is available.

## 18.4 Figures

- [ ] Alpha distribution plot by magnification.
- [ ] Optional alpha vs confidence/error plot.

---

# 19. Interpretability results

## 19.1 CNN branch — Grad-CAM

**Required Information**

- [ ] Target layer selected and named.
- [ ] Target class definition.
- [ ] Grad-CAM procedure.

**Actual Result to Collect**

- [ ] Representative correctly classified examples.
- [ ] Representative incorrect examples.
- [ ] At least examples across relevant magnifications/classes.
- [ ] Original image + heatmap overlay.

## 19.2 ViT branch — attention rollout

**Required Information**

- [ ] Exact layer/block aggregation method.
- [ ] Attention rollout implementation details.

**Actual Result to Collect**

- [ ] Representative correct cases.
- [ ] Representative failure cases.
- [ ] Heatmap overlays.

## 19.3 Fused representation

- [ ] t-SNE or equivalent embedding visualization.
- [ ] Color/label by subtype.
- [ ] Optional second plot by magnification.

**Actual Result to Collect**

- [ ] Random seed.
- [ ] Number of samples used.
- [ ] Dimensionality-reduction parameters.
- [ ] Figure and interpretation.

## 19.4 Interpretation discipline

- [ ] Do not claim clinical reasoning from visualization alone.
- [ ] Describe visualizations as model-relevance/localization evidence.

---

# 20. Error analysis

## 20.1 Required error categories

**Actual Result to Collect**

- [ ] False positives.
- [ ] False negatives.
- [ ] Major subtype confusions.
- [ ] Low-confidence predictions.
- [ ] Unusual fusion-gate behavior.
- [ ] Minority-class failures.
- [ ] Magnification-specific failures.

## 20.2 Case table

For selected representative cases collect:

- [ ] image ID;
- [ ] patient ID;
- [ ] magnification;
- [ ] true class;
- [ ] predicted class;
- [ ] predicted probability/confidence;
- [ ] alpha;
- [ ] explanation/visualization;
- [ ] observed failure pattern;
- [ ] whether a plausible visual ambiguity is present.

## 20.3 Error-analysis summary

- [ ] Top recurring failure modes.
- [ ] Whether failures disproportionately affect minority subtypes.
- [ ] Whether errors cluster by magnification.

---

# 21. Statistical evidence and uncertainty

## 21.1 Cross-fold reporting

**Required Information**

- [ ] Report all five folds.
- [ ] Mean.
- [ ] Standard deviation.
- [ ] Confidence interval methodology.

## 21.2 Confidence intervals

**Actual Result to Collect**

- [ ] 95% CI for primary metric.
- [ ] 95% CI for key baseline comparison.
- [ ] Exact CI method used.
- [ ] Unit of resampling if bootstrapping is used: preferably patient-level for patient-correlated data.

## 21.3 Statistical significance

**Required Information**

- [ ] Predefine which comparisons will receive significance testing.
- [ ] Avoid running many unplanned tests simply because they are available.

**Actual Result to Collect**

- [ ] Test name.
- [ ] Statistic.
- [ ] p-value.
- [ ] effect size.
- [ ] confidence interval of difference where applicable.
- [ ] multiple-comparison correction if multiple tests are performed.

## 21.4 Interpretation

- [ ] Distinguish statistical significance from practical importance.
- [ ] Do not declare a winner based on a tiny numerical difference without uncertainty/context.

---

# 22. External / cross-dataset validation

## 22.1 Status

**Required Information**

- [ ] Treat external validation as optional unless the project team explicitly adds it.
- [ ] If no external dataset is used, state this as a limitation.

## 22.2 If performed

**Actual Result to Collect**

- [ ] External dataset name/version.
- [ ] Source and license/access details.
- [ ] Class mapping.
- [ ] Magnification availability.
- [ ] Stain/domain characteristics.
- [ ] Preprocessing compatibility.
- [ ] Whether magnification metadata is available; if not, document how the model is evaluated without it.
- [ ] No parameter tuning on external test data.
- [ ] External accuracy/macro-F1/balanced accuracy/MCC.
- [ ] Per-class performance where valid.
- [ ] Performance drop from internal to external domain.

## 22.3 Internal robustness if external validation is not performed

- [ ] Controlled test-time stain perturbation experiment.
- [ ] Compare clean vs perturbed performance.
- [ ] Report absolute and relative performance degradation.

---

# 23. Computational complexity and resource usage

## 23.1 Model complexity

**Actual Result to Collect**

- [ ] Total trainable parameters.
- [ ] Total parameters.
- [ ] Parameters by branch.
- [ ] Parameters in MAF.
- [ ] Parameters in residual module.
- [ ] FLOPs/GFLOPs or equivalent estimate at 224×224.

## 23.2 Runtime/resource measurements

- [ ] GPU/accelerator model.
- [ ] GPU VRAM.
- [ ] CPU model if relevant.
- [ ] System RAM.
- [ ] Software environment.
- [ ] Training time per fold.
- [ ] Total training time.
- [ ] Inference time per image or images/sec.
- [ ] Peak VRAM.

## 23.3 Lightweight claim

- [ ] Define “lightweight” quantitatively in the final paper.
- [ ] Avoid qualitative claims without parameter/compute evidence.

---

# 24. Reproducibility and implementation details

## 24.1 Environment

**Actual Result to Collect**

- [ ] Python version.
- [ ] PyTorch version.
- [ ] torchvision version.
- [ ] timm version if used.
- [ ] scikit-learn version.
- [ ] NumPy version.
- [ ] Pillow/OpenCV version if used.
- [ ] stain augmentation library/version if used.
- [ ] CUDA/driver details if applicable.
- [ ] Colab runtime type.

## 24.2 Data organization

- [ ] Dataset root.
- [ ] Output directory.
- [ ] Checkpoint naming convention.
- [ ] Fold assignment files.

## 24.3 Randomness

- [ ] Global seed.
- [ ] DataLoader worker seed behavior.
- [ ] NumPy/PyTorch/Python seeds if applicable.
- [ ] Deterministic settings if used.

## 24.4 Checkpoints and artifacts

- [ ] Saved model checkpoint per fold.
- [ ] Saved best epoch.
- [ ] Saved optimizer/scheduler state if available.
- [ ] Saved configuration file.
- [ ] Saved metrics JSON/CSV.
- [ ] Saved predictions.
- [ ] Saved fold assignments.
- [ ] Saved figures.

## 24.5 Final implementation statement

**Required Information**

The final paper should report that the study used:

- Google Colab, if that remains the actual environment;
- BreaKHis from the documented project storage/location;
- fixed seed 42 unless changed and documented;
- patient-disjoint five-fold validation;
- saved per-fold checkpoints;
- stored experiment configuration;
- exported metrics/figures;
- a self-contained notebook or executable implementation.

**Actual Result to Collect**

- [ ] Exact versions and final repository/notebook path: `[TBD]`

---

# 25. Main paper tables required

- [ ] **Table 1:** BreaKHis dataset characteristics.
- [ ] **Table 2:** Eight-class distribution.
- [ ] **Table 3:** Literature comparison / research-gap matrix.
- [ ] **Table 4:** Final architecture/configuration.
- [ ] **Table 5:** Training configuration.
- [ ] **Table 6:** Main five-fold classification results.
- [ ] **Table 7:** Baseline comparison.
- [ ] **Table 8:** Ablation results.
- [ ] **Table 9:** Per-class precision/recall/F1.
- [ ] **Table 10:** Magnification-wise performance.
- [ ] **Table 11:** Patient-level results.
- [ ] **Table 12:** Robustness/domain-shift results, if performed.
- [ ] **Table 13:** Computational cost.
- [ ] **Table 14:** Optional external validation results.

Every table must have:

- [ ] exact metric definition;
- [ ] units where applicable;
- [ ] sample counts;
- [ ] mean ± variability where appropriate;
- [ ] CI or significance notation where used;
- [ ] clearly defined best/selected model convention.

---

# 26. Main paper figures required

- [ ] **Figure 1:** Overall research pipeline.
- [ ] **Figure 2:** Representative BreaKHis images across magnifications/classes.
- [ ] **Figure 3:** OMNet-V3 architecture.
- [ ] **Figure 4:** MAF mechanism and equations.
- [ ] **Figure 5:** Training/validation curves.
- [ ] **Figure 6:** Binary confusion matrix.
- [ ] **Figure 7:** Eight-class confusion matrix.
- [ ] **Figure 8:** Magnification-wise performance.
- [ ] **Figure 9:** Fusion-gate alpha distributions.
- [ ] **Figure 10:** Grad-CAM examples.
- [ ] **Figure 11:** ViT attention-rollout examples.
- [ ] **Figure 12:** Final fused-representation t-SNE/embedding plot.
- [ ] **Figure 13:** Representative error cases.
- [ ] **Figure 14:** Optional clean-vs-stain-shift robustness plot.

For every figure:

- [ ] source data recorded;
- [ ] figure-generation script/notebook saved;
- [ ] train/test contamination ruled out;
- [ ] axes/legends/sample counts included;
- [ ] captions explain what the figure demonstrates without overstating it.

---

# 27. Results section writing requirements

## 27.1 Dataset statistics

**Actual Result to Collect**

- [ ] Verified counts from the actual experiment files, not copied blindly from a paper.
- [ ] Final split statistics.

## 27.2 Training behavior

- [ ] Curves.
- [ ] Best epoch.
- [ ] Early-stopping behavior.
- [ ] Any instability/failed folds.

## 27.3 Primary classification result

- [ ] Mean + variability.
- [ ] Fold-level results.
- [ ] Confidence interval.
- [ ] Baseline comparison.

## 27.4 Secondary analyses

- [ ] Per-class.
- [ ] Magnification.
- [ ] patient-level.
- [ ] hierarchy consistency.
- [ ] MAF alpha.
- [ ] interpretability.
- [ ] error analysis.
- [ ] robustness.

## 27.5 Result integrity

- [ ] Every reported number traceable to saved data/log.
- [ ] No manually typed number without source artifact.
- [ ] No result changed after statistical analysis without documenting the change.

---

# 28. Discussion requirements

## 28.1 Main finding

- [ ] Answer whether the primary hypothesis was supported.
- [ ] State effect size/direction.
- [ ] Include uncertainty.

## 28.2 Local/global fusion

- [ ] Discuss whether fusion improves over individual branches.
- [ ] Discuss whether the evidence suggests complementary representations.
- [ ] Avoid claiming causality beyond the ablation evidence.

## 28.3 Magnification-aware behavior

- [ ] Discuss performance by magnification.
- [ ] Discuss gate behavior by magnification.
- [ ] Do not predetermine which branch “should” dominate at any magnification.

## 28.4 Hierarchy

- [ ] Discuss binary performance.
- [ ] Discuss subtype performance.
- [ ] Discuss consistency.
- [ ] State whether hierarchy actually improved the primary endpoint.

## 28.5 Robustness/generalization

- [ ] Distinguish internal stain robustness from external generalization.
- [ ] Quantify degradation rather than describing robustness qualitatively.

## 28.6 Literature comparison

- [ ] Only compare comparable protocols.
- [ ] Explain important split/metric differences.

---

# 29. Limitations

**Required Information — include only limitations that remain applicable**

- [ ] 82-patient dataset size.
- [ ] Class imbalance.
- [ ] Minority-class statistical power.
- [ ] Known magnification metadata requirement.
- [ ] Potential domain differences between BreaKHis and external data.
- [ ] Single-seed limitation if only seed 42 is used.
- [ ] Computational constraints.
- [ ] Lack of external validation if not performed.
- [ ] Explainability-method limitations.
- [ ] No clinical deployment/diagnostic validation.

**Actual Result to Collect**

- [ ] Final limitations list supported by the completed experiments.

---

# 30. Conclusion

**Required Information**

The conclusion must answer:

- [ ] What problem was addressed?
- [ ] What was proposed?
- [ ] What evidence was obtained?
- [ ] Which hypotheses were supported/not supported?
- [ ] What is the most defensible methodological contribution?
- [ ] What remains unresolved?

**Prohibited until supported**

- [ ] No fabricated accuracy.
- [ ] No unverified superiority.
- [ ] No unsupported generalization claim.
- [ ] No clinical-readiness claim.
- [ ] No unsupported novelty claim.

---

# 31. Future work

Keep future work proportional to the project. Candidate directions only; do not present them as completed work:

- [ ] External multi-dataset validation.
- [ ] Larger/more diverse histopathology datasets.
- [ ] Stronger domain adaptation/generalization testing.
- [ ] Multi-scale image/patch modeling beyond metadata-conditioned fusion.
- [ ] More advanced stain normalization/augmentation.
- [ ] Multi-seed/repeated evaluation.
- [ ] Prospective/clinical validation only as long-term future work.

---

# 32. Final evidence matrix — master collection sheet

| Evidence item | Required? | Status | Source/artifact | Final value/location |
|---|---|---|---|---|
| Dataset counts | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Patient IDs/fold assignments | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Leakage check | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Class distribution | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Magnification distribution | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Exact preprocessing | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Exact Macenko configuration | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Architecture implementation | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Loss implementation | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Fold checkpoints | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Training curves | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Main metrics | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Fold-level metrics | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Confidence intervals | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Baselines | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Ablations | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Per-class results | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Confusion matrices | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Magnification analysis | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Alpha/gate analysis | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Grad-CAM | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Attention rollout | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| t-SNE/embedding analysis | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Error cases | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Robustness experiment | Yes for robustness claim | `[TBD]` | `[TBD]` | `[TBD]` |
| External validation | Optional | `[TBD]` | `[TBD]` | `[TBD]` |
| Parameter count | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| FLOPs/compute | Yes if “lightweight” claimed | `[TBD]` | `[TBD]` | `[TBD]` |
| Runtime/VRAM | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Software versions | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Reproducibility artifacts | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Literature-gap matrix | Yes | `[TBD]` | `[TBD]` | `[TBD]` |
| Novelty wording | Yes, after literature review | `[TBD]` | `[TBD]` | `[TBD]` |

---

# 33. Final manuscript section map

Use this as the writing order after experiments are complete:

1. [ ] Title
2. [ ] Abstract
3. [ ] Keywords
4. [ ] Introduction
5. [ ] Research objective / questions / hypotheses
6. [ ] Related work
7. [ ] Research gap and contributions
8. [ ] Dataset
9. [ ] Preprocessing and augmentation
10. [ ] Patient-disjoint validation protocol
11. [ ] Proposed OMNet-V3 architecture
12. [ ] Mathematical formulation
13. [ ] Training methodology
14. [ ] Baselines and ablations
15. [ ] Evaluation metrics and statistical analysis
16. [ ] Main results
17. [ ] Per-class and confusion-matrix analysis
18. [ ] Magnification analysis
19. [ ] Fusion-gate analysis
20. [ ] Interpretability
21. [ ] Error analysis
22. [ ] Robustness / external validation
23. [ ] Computational analysis
24. [ ] Discussion
25. [ ] Limitations
26. [ ] Conclusion
27. [ ] Future work
28. [ ] Reproducibility statement
29. [ ] References
30. [ ] Supplementary material / appendix, if required

---

# 34. Research agenda vs actual evidence

## Research objective

> Determine whether a lightweight EfficientNet-B0 + ViT-Tiny architecture with magnification-conditioned adaptive local–global fusion and hierarchical learning can improve patient-disjoint breast histopathology classification on BreaKHis while providing measurable robustness and interpretable fusion behavior.

## Research gap

> The study targets the intersection of local/global representation balance, explicit magnification conditioning, hierarchical pathology learning, leakage-resistant patient-level evaluation, and stain/domain robustness. The final literature review must verify the extent to which this exact combination has already been explored.

## Proposed solution

> EfficientNet-B0 + ViT-Tiny → 256-D projections → 64-D magnification embedding → adaptive fusion gate → residual enrichment → binary and eight-class heads, trained with the locked weighted hierarchical objective.

## Hypotheses

> H1: fusion benefit.  
> H2: magnification-conditioning benefit.  
> H3: hierarchical-learning benefit.  
> H4: stain-robustness benefit.

## Expected outcomes

These are expectations, not results:

- [ ] CNN and ViT provide complementary information.
- [ ] Adaptive fusion improves over individual/fixed baselines.
- [ ] Magnification conditioning changes fusion behavior and may improve performance.
- [ ] Hierarchical learning may improve subtype recognition and consistency.
- [ ] Stain augmentation may reduce degradation under controlled stain shift.
- [ ] Patient-level metrics may provide a stricter view than image-level metrics.

## Evidence required

The final paper can support these statements only after obtaining:

- [ ] controlled baseline comparisons;
- [ ] ablations;
- [ ] five-fold patient-disjoint results;
- [ ] uncertainty/statistical evidence;
- [ ] magnification-wise analysis;
- [ ] fusion-gate analysis;
- [ ] interpretability evidence;
- [ ] error analysis;
- [ ] robustness/domain-shift evaluation;
- [ ] reproducibility records.

---

# 35. Source and methodology register

This register records the main externally verified methodological anchors used while preparing this checklist. It does **not** replace the final literature review.

1. **BreaKHis dataset:** official database documentation and original dataset literature should be used to verify dataset counts, magnifications, acquisition details, and filename conventions. The official documentation lists the BreaKHis collection and four magnifications; the final experiment must verify counts from the actual downloaded files.
   - Official database: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
   - Original dataset paper: https://pubmed.ncbi.nlm.nih.gov/26540668/

2. **EfficientNet:** the selected EfficientNet-B0 backbone is grounded in the EfficientNet family’s efficiency/scaling methodology.
   - https://arxiv.org/abs/1905.11946
   - https://research.google/pubs/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks/

3. **Vision Transformer:** ViT establishes the patch-based Transformer approach used as the global-representation branch.
   - https://arxiv.org/abs/2010.11929
   - https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/

4. **StratifiedGroupKFold:** the evaluation design uses grouped, class-stratified folds so that groups do not cross folds while class proportions are preserved as far as feasible.
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

5. **Grad-CAM:** the CNN interpretation method is based on gradient-weighted localization of class-relevant regions.
   - https://arxiv.org/abs/1610.02391

6. **Attention rollout:** the ViT interpretation method is based on propagating attention information across Transformer layers; raw attention alone should not automatically be treated as a faithful explanation.
   - https://arxiv.org/abs/2005.00928

7. **Macenko stain method:** the planned stain augmentation should be documented from the actual implementation and parameters; the paper must distinguish stain perturbation/augmentation from true external-domain generalization.
   - Final reference to be inserted after verifying the exact Macenko implementation/paper used by the project: `[TO VERIFY]`

---

# 36. Final pre-writing gate

The final research paper should **not** be drafted as a publication claim until all applicable boxes below are complete:

- [ ] Dataset independently verified from actual files.
- [ ] Patient IDs correctly extracted.
- [ ] Five-fold split generated and saved.
- [ ] No patient leakage confirmed automatically.
- [ ] Train/validation/test counts recorded.
- [ ] Final preprocessing frozen.
- [ ] Exact augmentation configuration frozen.
- [ ] Exact Macenko implementation documented.
- [ ] OMNet-V3 implementation frozen.
- [ ] Mathematical formulation matches code.
- [ ] Training configuration matches code.
- [ ] CNN baseline completed.
- [ ] ViT baseline completed.
- [ ] Hybrid baseline completed.
- [ ] Adaptive/fixed fusion comparison completed.
- [ ] Magnification ablation completed.
- [ ] Hierarchy ablation completed.
- [ ] Stain augmentation ablation completed.
- [ ] Five-fold primary results exported.
- [ ] Secondary metrics exported.
- [ ] Per-class metrics exported.
- [ ] Confusion matrices generated.
- [ ] Patient-level aggregation defined and evaluated.
- [ ] Magnification analysis completed.
- [ ] Fusion-gate analysis completed.
- [ ] Grad-CAM generated.
- [ ] Attention rollout generated.
- [ ] Embedding analysis generated.
- [ ] Error analysis completed.
- [ ] Confidence intervals calculated.
- [ ] Planned statistical comparisons completed.
- [ ] Robustness/domain-shift experiment completed or claim narrowed.
- [ ] External validation completed or explicitly omitted as a limitation.
- [ ] Parameter/compute/resource measurements collected.
- [ ] Software versions recorded.
- [ ] Checkpoints/metrics/configurations archived.
- [ ] Literature-gap matrix finalized.
- [ ] Novelty wording reviewed against literature.
- [ ] All manuscript numerical claims traceable to evidence artifacts.
- [ ] Results, discussion, and conclusion written only after the evidence above is complete.

---

# 37. Non-negotiable integrity rules

1. **Never invent results.**
2. **Never convert an expected outcome into a factual claim.**
3. **Never report a significance test without the actual calculation.**
4. **Never report a confidence interval without documenting the method.**
5. **Never claim clinical effectiveness from BreaKHis alone.**
6. **Never claim external generalization without external or controlled domain-shift evidence.**
7. **Never claim architectural novelty without the completed literature review.**
8. **Never compare published numbers without checking task, dataset, split, and metric compatibility.**
9. **Never use test data to choose hyperparameters or model variants.**
10. **Keep the final conclusions strictly proportional to the collected evidence.**

---

## End-state definition

The completed project should produce **a reproducible OMNet-V3 implementation, saved fold-level evidence, controlled baseline/ablation comparisons, patient-level and image-level evaluation, magnification/fusion analysis, interpretability and error-analysis artifacts, and a defensible literature-grounded research gap**. Only then should the final publication-quality paper be written.
