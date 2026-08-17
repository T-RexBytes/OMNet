# Final Model Plan: Patient-Aware Hierarchical Attention Network

## 1. Final Decision

This project will stop the CNN-ViT fusion line after the completed
experiments and use one final model architecture for the remaining
training campaign.

The final model is:

**Patient-Aware Hierarchical Attention Network (PAHAN)**

Core design:

``` text
Patient
  |
  +-- Multiple 200X BreakHis images
          |
          v
    Shared DenseNet-201
      pretrained ImageNet
          |
          v
    Image-level embeddings
          |
          v
    Attention-based MIL pooling
          |
          v
    Patient-level representation
          |
          +----------------------+
          |                      |
          v                      v
 Primary classification     Hierarchical subtype
 Benign / Malignant              routing
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              Benign expert        Malignant expert
                 4-way                 4-way
                    |                     |
                    +----------+----------+
                               |
                               v
                        8-class subtype
                               |
                               v
                       Consistency loss
```

This is the final planned architecture. No additional CNN-ViT fusion
experiment, ensemble, or backbone search should be introduced unless a
serious implementation failure makes the model impossible to train.

The goal is to spend the remaining compute budget on one scientifically
motivated model rather than another sequence of speculative
architectures.

------------------------------------------------------------------------

# 2. Project Objective

The project has two simultaneous classification objectives.

## Task A: Primary cancer classification

Binary classification:

-   Benign
-   Malignant

## Task B: Histological subtype classification

Eight classes:

1.  Adenosis
2.  Fibroadenoma
3.  Phyllodes tumor
4.  Tubular adenoma
5.  Ductal carcinoma
6.  Lobular carcinoma
7.  Mucinous carcinoma
8.  Papillary carcinoma

The final model must produce both:

``` text
Primary prediction
    benign / malignant

Subtype prediction
    one of 8 histological subtypes
```

The model must remain evaluated using patient-level separation.

------------------------------------------------------------------------

# 3. Experimental History

The final architecture is based on the observed behavior of the
completed experiments.

## Baseline 1: DenseNet-201

Pretrained DenseNet-201 was fine-tuned for the multi-task problem.

Observed subtype Macro-F1:

``` text
approximately 0.294
```

Primary held-out test:

``` text
Accuracy ≈ 0.768
F1 ≈ 0.698
```

DenseNet was a strong baseline, particularly for subtype classification.

## Baseline 2: ViT-B/16

Pretrained ViT-B/16 was fine-tuned for the same tasks.

Observed subtype Macro-F1:

``` text
approximately 0.307
```

Primary held-out test:

``` text
Accuracy ≈ 0.879
F1 ≈ 0.854
```

ViT was competitive and achieved the strongest original subtype result.

## Baseline 3: Static DenseNet + ViT Fusion

DenseNet and ViT features were projected and concatenated before
classification.

Observed subtype Macro-F1:

``` text
approximately 0.275
```

Static fusion therefore failed to improve over either backbone.

## Final Gated Fusion + Class-Balanced Loss

A learned CNN/ViT gate was introduced together with class-balanced
subtype learning.

Observed subtype Macro-F1:

``` text
approximately 0.281
```

This remained below the original DenseNet and ViT baselines.

The learned gate was predominantly around 0.60--0.70 across subtypes,
indicating consistent CNN preference rather than strong subtype-specific
CNN/ViT specialization.

The class-balanced objective improved some minority classes but reduced
performance on several other classes and did not improve overall
Macro-F1.

------------------------------------------------------------------------

# 4. Diagnosis From Previous Experiments

The previous experiments indicate that the main problem should not be
attacked by adding another fusion mechanism.

The evidence suggests:

1.  CNN and ViT feature fusion is not producing useful complementary
    information under the current dataset and training setup.
2.  The learned fusion mechanism tends to rely on the CNN
    representation.
3.  Class reweighting alone does not solve the subtype problem.
4.  The subtype task is substantially harder than the primary
    benign/malignant task.
5.  The dataset contains severe patient-level imbalance for several
    subtypes.
6.  Multiple images belong to the same patient, meaning image-level
    prediction does not fully exploit the structure available in the
    dataset.
7.  A flat 8-class classifier forces all subtypes into one competition.
8.  The benign/malignant hierarchy provides meaningful structure that
    can be incorporated into the model.

The final architecture therefore changes the modeling unit from an
isolated image to a patient-level bag of images and changes subtype
prediction from a flat 8-way problem to a hierarchy-aware prediction
problem.

------------------------------------------------------------------------

# 5. Research Hypothesis

The final research hypothesis is:

> A patient-aware hierarchical attention model can improve fine-grained
> breast tumor subtype recognition by aggregating multiple tissue
> observations from the same patient and constraining subtype
> predictions according to the benign/malignant hierarchy, addressing
> limitations observed in flat image-level CNN, ViT, and CNN-ViT fusion
> models.

The hypothesis has three components:

1.  Patient-level aggregation should provide a more complete
    representation than isolated image classification.
2.  Attention pooling should allow the model to identify informative
    images rather than treating every image equally.
3.  Hierarchical subtype prediction should reduce unnecessary
    competition between biologically distinct benign and malignant
    groups.

The experiment must test this hypothesis rather than assume it is true.

------------------------------------------------------------------------

# 6. Dataset

Dataset:

**BreakHis**

Locked magnification:

``` text
200X
```

Modality:

``` text
2D breast histopathology images
```

No additional magnifications will be introduced in the final experiment.

The existing patient identifiers must be preserved.

Images belonging to the same patient must never be distributed across
train, validation, and test partitions.

------------------------------------------------------------------------

# 7. Fundamental Data Unit

The most important architectural change is the definition of a sample.

Previous models effectively used:

``` text
one image = one sample
```

The final model uses:

``` text
one patient = one bag
```

A patient bag contains all eligible 200X images associated with that
patient.

Example:

``` text
Patient A
    image_1
    image_2
    image_3
    image_4
    ...
```

These images are processed by the same shared feature extractor.

The model then learns how much attention each image should receive.

------------------------------------------------------------------------

# 8. Patient Bag Construction

Create a patient-level metadata table.

Required fields:

``` text
patient_id
filepath
magnification
primary_label
subtype_label
split
```

For each patient:

``` text
patient_id
    |
    +-- image paths
    +-- primary label
    +-- subtype label
```

A patient must have exactly one primary label and one subtype label.

If contradictory labels are discovered within a patient, fail the audit
instead of silently resolving the conflict.

------------------------------------------------------------------------

# 9. Patient-Level Data Splitting

The existing corrected patient-level split artifacts must be reused
where possible.

Task A:

``` text
70% train
15% validation
15% held-out test
```

The split is performed at patient level.

Task B:

``` text
5-fold patient-level cross-validation
```

No image from a patient may appear in a different fold.

Mandatory assertions:

``` text
train ∩ validation = empty
train ∩ test = empty
validation ∩ test = empty
```

For Task B:

``` text
fold_i ∩ fold_j = empty
```

for every pair of folds.

------------------------------------------------------------------------

# 10. Rare-Class Handling

The following classes require explicit monitoring:

``` text
phyllodes_tumor
adenosis
```

The dataset contains very few patients for some rare subtypes.

This creates a statistical limitation that architecture alone cannot
remove.

The implementation must:

1.  Report patient counts per subtype.
2.  Report image counts per subtype.
3.  Report the number of patients represented in each fold.
4.  Flag folds with insufficient class support.
5.  Never fabricate samples.
6.  Never split images from one patient across folds.

Macro-F1 must be interpreted together with per-class F1 and support.

------------------------------------------------------------------------

# 11. Final Architecture

## 11.1 Backbone

Use:

``` text
DenseNet-201 pretrained on ImageNet
```

Do not train DenseNet-201 from scratch.

The DenseNet classifier is removed.

The convolutional feature extractor produces a feature map.

Apply adaptive global average pooling to obtain:

``` text
1920-dimensional image embedding
```

Therefore:

``` text
image
  -> DenseNet-201
  -> feature map
  -> adaptive average pooling
  -> 1920-d embedding
```

------------------------------------------------------------------------

# 12. Image-Level Projection

The raw 1920-dimensional embedding is projected into a lower-dimensional
representation before attention.

Recommended dimension:

``` text
384
```

Projection:

``` text
Linear(1920, 384)
BatchNorm1d(384)
ReLU
Dropout(0.2-0.3)
```

Output:

``` text
384-d image representation
```

This keeps the attention module computationally small.

------------------------------------------------------------------------

# 13. Attention-Based Multiple Instance Learning

For a patient containing N images:

``` text
h1
h2
...
hN
```

where each:

``` text
hi ∈ R^384
```

The model calculates an attention score for every image.

Use gated attention:

``` text
a_i =
softmax(
    w^T(
        tanh(V h_i)
        ⊙
        sigmoid(U h_i)
    )
)
```

where:

``` text
V: 384 -> attention_dim
U: 384 -> attention_dim
w: attention_dim -> 1
```

Recommended:

``` text
attention_dim = 128
```

The attention weights must sum to 1 within each patient bag.

The patient representation is:

``` text
z_patient = Σ a_i h_i
```

Therefore the network learns which images contribute most to the
patient-level diagnosis.

------------------------------------------------------------------------

# 14. Attention Interpretation

The model must export attention weights.

For every patient, save:

``` text
patient_id
image_path
attention_weight
primary_label
subtype_label
```

This provides a useful interpretability artifact.

The highest-attention images can be inspected after training.

Attention weights must not be described as medical explanations or
lesion localization unless separately validated.

They should be described as model-derived importance weights.

------------------------------------------------------------------------

# 15. Patient Representation

After attention pooling:

``` text
patient embedding = 384 dimensions
```

Optional lightweight refinement:

``` text
Linear(384, 384)
LayerNorm
ReLU
Dropout(0.2)
```

Do not add a large transformer or additional backbone here.

The patient representation must remain compact.

------------------------------------------------------------------------

# 16. Primary Classification Head

The primary head predicts:

``` text
Benign
Malignant
```

Architecture:

``` text
384
 |
Dropout
 |
Linear(384, 2)
```

Output:

``` text
logits_primary ∈ R^2
```

------------------------------------------------------------------------

# 17. Hierarchical Subtype Architecture

The eight subtypes are divided into two conditional groups.

## Benign branch

``` text
adenosis
fibroadenoma
phyllodes_tumor
tubular_adenoma
```

## Malignant branch

``` text
ductal_carcinoma
lobular_carcinoma
mucinous_carcinoma
papillary_carcinoma
```

Two independent subtype experts are used.

------------------------------------------------------------------------

# 18. Benign Subtype Expert

Input:

``` text
384-d patient representation
```

Architecture:

``` text
384
 |
Linear(384, 256)
 |
ReLU
 |
Dropout(0.3)
 |
Linear(256, 4)
```

Output:

``` text
P(subtype | benign)
```

Classes:

``` text
adenosis
fibroadenoma
phyllodes_tumor
tubular_adenoma
```

------------------------------------------------------------------------

# 19. Malignant Subtype Expert

Input:

``` text
384-d patient representation
```

Architecture:

``` text
384
 |
Linear(384, 256)
 |
ReLU
 |
Dropout(0.3)
 |
Linear(256, 4)
```

Output:

``` text
P(subtype | malignant)
```

Classes:

``` text
ductal_carcinoma
lobular_carcinoma
mucinous_carcinoma
papillary_carcinoma
```

------------------------------------------------------------------------

# 20. Soft Hierarchical Routing

Do not hard-route the patient through one branch.

For example:

``` text
P(Benign) = 0.45
P(Malignant) = 0.55
```

Both subtype experts remain relevant.

Let:

``` text
P_B = primary benign probability
P_M = primary malignant probability
```

For a benign subtype:

``` text
P(subtype) = P_B × P(subtype | benign)
```

For a malignant subtype:

``` text
P(subtype) = P_M × P(subtype | malignant)
```

The eight resulting probabilities are concatenated and normalized if
required.

This prevents a primary classification mistake from automatically
forcing the subtype into the wrong branch.

------------------------------------------------------------------------

# 21. Hierarchical Consistency

The subtype distribution implies a primary probability.

For example:

``` text
P_M_from_subtype =
    P(ductal)
  + P(lobular)
  + P(mucinous)
  + P(papillary)
```

Similarly:

``` text
P_B_from_subtype =
    P(adenosis)
  + P(fibroadenoma)
  + P(phyllodes)
  + P(tubular)
```

The primary prediction should agree with these aggregated subtype
probabilities.

------------------------------------------------------------------------

# 22. Consistency Loss

Use:

``` text
L_consistency =
KL(
    P_primary
    ||
    P_primary_from_subtype
)
```

or an equivalent symmetric probability-distribution consistency loss.

The implementation must keep the consistency coefficient small enough
that the primary objective is not dominated by subtype supervision.

Recommended starting value:

``` text
lambda_consistency = 0.1
```

This should be treated as the locked final configuration unless a
numerical stability issue requires adjustment.

------------------------------------------------------------------------

# 23. Final Loss

The overall loss is:

``` text
L_total =
L_primary
+
lambda_subtype * L_subtype
+
lambda_consistency * L_consistency
```

Recommended:

``` text
lambda_subtype = 1.0
lambda_consistency = 0.1
```

For subtype loss:

``` text
L_subtype
=
L_benign_branch
+
L_malignant_branch
```

However, only the relevant branch should contribute directly to the
supervised subtype loss for a sample.

For a benign patient:

``` text
L_subtype = L_benign_branch
```

For a malignant patient:

``` text
L_subtype = L_malignant_branch
```

The other branch can still participate in forward inference but should
not receive a direct target for that patient.

------------------------------------------------------------------------

# 24. Class Imbalance Strategy

Do not repeat the aggressive class-balanced loss experiment used in the
previous final model.

The previous experiment demonstrated that class reweighting can shift
performance between classes without improving overall Macro-F1.

The final model should instead use:

1.  Hierarchical decomposition.
2.  Mild inverse-frequency class weights within each four-class expert.
3.  Patient-level sampling where necessary.
4.  Macro-F1 as the primary subtype selection metric.

Weights must be calculated using training patients only.

Validation and test distributions must never influence class weights.

------------------------------------------------------------------------

# 25. Patient-Level Sampling

The primary sampling unit is the patient.

Do not oversample individual images independently.

If weighted sampling is required, assign each patient a weight based on
its subtype frequency.

Example:

``` text
patient_weight =
1 / number_of_training_patients_in_subtype
```

This ensures that a patient with many images does not automatically
dominate training simply because it contains more images.

Avoid aggressive oversampling if it causes the same patient to appear
excessively often within an epoch.

------------------------------------------------------------------------

# 26. Variable Number of Images Per Patient

Patients can contain different numbers of images.

The DataLoader must support variable-length patient bags.

Preferred implementation:

``` text
custom collate_fn
```

or controlled padding.

For a batch:

``` text
patient_1 -> N1 images
patient_2 -> N2 images
...
patient_B -> NB images
```

Pad to:

``` text
N_max
```

and provide an attention mask.

Padded images must never receive attention.

------------------------------------------------------------------------

# 27. Memory-Constrained Strategy

Because this is being trained on Google Colab Free:

Use:

``` text
mixed precision
gradient accumulation
num_workers = 0
pin_memory = True when supported
```

The batch size is the number of patients, not the number of images.

Recommended starting point:

``` text
patient_batch_size = 2-4
```

depending on the actual GPU memory.

Use gradient accumulation to reach an effective batch size of
approximately:

``` text
8-16 patients
```

without requiring that many patient bags in GPU memory simultaneously.

------------------------------------------------------------------------

# 28. Feature Caching Option

If end-to-end training repeatedly causes GPU memory or runtime problems,
implement an optional feature-cache mode.

Pipeline:

``` text
Image
  ↓
Frozen DenseNet-201
  ↓
1920-d embedding
  ↓
save embedding
```

Then:

``` text
cached embeddings
  ↓
projection
  ↓
attention MIL
  ↓
hierarchical heads
```

This can substantially reduce repeated backbone computation.

However, the final preferred training mode is:

``` text
Phase 1: frozen backbone + train patient-level architecture
Phase 2: partial DenseNet fine-tuning
```

------------------------------------------------------------------------

# 29. Training Strategy

Use two phases.

## Phase 1: Patient-level head training

Freeze DenseNet-201.

Train:

-   projection
-   attention module
-   patient representation
-   primary head
-   benign subtype expert
-   malignant subtype expert
-   consistency mechanism

Recommended:

``` text
epochs = 8-12
```

Early stopping:

``` text
patience = 3-4
```

Selection metric:

``` text
validation subtype Macro-F1
```

------------------------------------------------------------------------

# 30. Phase 2: Partial Fine-Tuning

If Phase 1 is stable, unfreeze only the later DenseNet layers.

Recommended:

``` text
DenseNet denseblock4
```

Keep earlier DenseNet layers frozen.

Use discriminative learning rates.

Recommended:

``` text
backbone_lr = 1e-5
head_lr = 1e-3
```

Use AdamW.

Recommended:

``` text
weight_decay = 1e-4
```

Recommended maximum:

``` text
epochs = 12-15
```

Early stopping:

``` text
patience = 4
```

The best checkpoint is selected using validation subtype Macro-F1.

Do not use test performance for checkpoint selection.

------------------------------------------------------------------------

# 31. Optimizer

Use:

``` text
AdamW
```

Parameter groups:

``` text
DenseNet backbone:
lr = 1e-5

New architecture:
lr = 1e-3
```

Weight decay:

``` text
1e-4
```

A cosine learning-rate schedule may be used.

Do not perform a broad LR search.

------------------------------------------------------------------------

# 32. Data Augmentation

Use the existing 200X preprocessing pipeline for comparability.

Recommended training augmentations:

``` text
Resize / controlled crop to 224x224
RandomHorizontalFlip
RandomVerticalFlip
RandomRotation(±20 degrees)
mild ColorJitter
```

Validation and test:

``` text
resize
center crop if required
ImageNet normalization
```

No augmentation may alter patient identity or labels.

------------------------------------------------------------------------

# 33. Input Resolution

Keep:

``` text
224x224
```

for the primary final experiment.

Do not simultaneously introduce high-resolution multi-crop processing,
multi-magnification processing, and the new hierarchy.

That would make it impossible to attribute improvement to the proposed
architecture and would significantly increase compute.

If a higher-resolution implementation is explored later, it must be
treated as a separate experiment, not silently incorporated into the
final model.

------------------------------------------------------------------------

# 34. Evaluation Protocol

## Task A

Evaluate on the held-out patient-level test set.

Report:

``` text
Accuracy
Precision
Recall
F1
Macro-F1
Confusion Matrix
Classification Report
```

## Task B

Perform 5-fold patient-level evaluation.

For each fold report:

``` text
Accuracy
Macro Precision
Macro Recall
Macro-F1
Per-class Precision
Per-class Recall
Per-class F1
Support
Confusion Matrix
```

Aggregate:

``` text
mean ± standard deviation
```

Do not report only the mean.

------------------------------------------------------------------------

# 35. Additional Patient-Level Metrics

Because the model is explicitly patient-aware, report:

``` text
number of patients
number of images
images per patient distribution
```

Report whether each subtype has adequate patient support.

Where possible, report confidence intervals or variability across folds.

Do not claim high certainty for classes represented by extremely few
patients.

------------------------------------------------------------------------

# 36. Comparison Against Existing Experiments

The final comparison table must include:

``` text
DenseNet-201
ViT-B/16
Static CNN+ViT Fusion
Gated CNN+ViT + CB Loss
Final Patient-Aware Hierarchical Attention Model
```

For each:

``` text
Task A
Accuracy
F1

Task B
Macro Precision
Macro Recall
Macro-F1
Std
```

The comparison must use the corrected evaluation protocol wherever
possible.

If an older baseline was trained/evaluated under a different protocol,
clearly mark it rather than pretending the numbers are directly
equivalent.

------------------------------------------------------------------------

# 37. Required Ablation Analysis

Do not train a large collection of new models.

Use the already completed experiments as historical ablations:

``` text
DenseNet
vs
ViT
vs
Static Fusion
vs
Gated Fusion
```

The new final model adds:

``` text
Patient-level attention
+
Hierarchical subtype routing
+
Consistency constraint
```

The paper can therefore explain the progression conceptually.

If compute permits only one additional controlled ablation, prioritize:

``` text
Final model without consistency loss
vs
Final model with consistency loss
```

However, this ablation should only be run if it can be obtained without
violating the project's compute constraint.

Otherwise, report the consistency component as part of the final
architecture and avoid inventing an unsupported ablation claim.

------------------------------------------------------------------------

# 38. Success Criteria

The final model should not be declared successful merely because it
beats one previous model.

Primary success criterion:

``` text
Task B subtype Macro-F1 improves over the strongest comparable baseline
with stable fold-level performance.
```

The strongest existing subtype baseline is approximately:

``` text
ViT-B/16 ≈ 0.307
```

DenseNet is approximately:

``` text
0.294
```

Static fusion:

``` text
0.275
```

Gated + CB:

``` text
0.281
```

Therefore a meaningful result should demonstrate:

``` text
Final Macro-F1 > 0.307
```

with evidence that the gain is not caused by a single fold or a single
class.

A stronger result would also improve minority subtype F1 without
severely degrading the dominant classes.

------------------------------------------------------------------------

# 39. Primary Task Success

Task A should be compared against the existing strong ViT result.

The final model should not be considered successful if subtype
performance increases only by destroying primary classification.

Report the trade-off explicitly.

The objective is:

``` text
strong primary classification
+
improved balanced subtype classification
```

not subtype improvement at any cost.

------------------------------------------------------------------------

# 40. Failure Interpretation

If the final model fails to exceed the best baseline, do not continue
adding architectures automatically.

Possible interpretation:

``` text
Dataset limitation
+
rare patient-level subtype support
+
label ambiguity
+
limited visual diversity
```

may dominate architectural effects.

The paper should then frame the result as a controlled investigation
showing that:

``` text
CNN
→ Transformer
→ feature fusion
→ adaptive fusion
→ patient-aware hierarchy
```

does not produce a reliable improvement under the available BreakHis
200X patient-level setting.

This is scientifically preferable to repeated uncontrolled model
modification.

------------------------------------------------------------------------

# 41. Reproducibility

Every run must save:

``` text
CONFIG
random seed
dataset version/reference
split_task_a.csv
folds_task_b.csv
training history
validation metrics
test metrics
per-class metrics
confusion matrices
model checkpoint
attention weights
patient-level predictions
```

Recommended output directory:

``` text
/content/drive/MyDrive/output_final/
```

------------------------------------------------------------------------

# 42. Required Prediction Artifact

Save a prediction table containing:

``` text
patient_id
true_primary
pred_primary
primary_probability
true_subtype
pred_subtype
subtype_probability
fold
```

For the attention model also save:

``` text
image_path
attention_weight
```

This allows later analysis without retraining.

------------------------------------------------------------------------

# 43. Attention Visualization

For selected patients, generate:

``` text
original image
attention weight
predicted primary
predicted subtype
true subtype
```

Sort images by attention weight.

Use this only for qualitative analysis.

Do not claim that attention identifies cancerous regions unless a
separate localization validation is performed.

------------------------------------------------------------------------

# 44. Error Analysis

After the final test/evaluation, identify:

1.  Most confused subtype pairs.
2.  Minority classes with near-zero F1.
3.  Dominant-class behavior.
4.  Primary/subtype disagreement.
5.  Patients where attention is highly concentrated.
6.  Patients where attention is distributed across many images.
7.  Cases where primary classification is correct but subtype is
    incorrect.
8.  Cases where primary classification is incorrect and subtype is
    consequently affected.

This analysis should be performed from saved predictions rather than
requiring another training run.

------------------------------------------------------------------------

# 45. Implementation Modules

The implementation should be divided into clean modules even if
delivered as one Colab notebook.

Recommended logical components:

``` text
1. Environment setup
2. Configuration
3. Reproducibility
4. Dataset discovery
5. Patient metadata construction
6. Patient-level split validation
7. PatientBagDataset
8. Variable-length collate function
9. DenseNet feature extractor
10. Image projection
11. Gated attention MIL
12. Patient representation
13. Primary head
14. Benign subtype expert
15. Malignant subtype expert
16. Soft hierarchical probability construction
17. Consistency loss
18. Training loop
19. Validation loop
20. Cross-validation
21. Final held-out evaluation
22. Attention export
23. Error analysis
24. Artifact export
```

------------------------------------------------------------------------

# 46. Model Class Design

Recommended conceptual API:

``` python
class PatientHierarchicalModel(nn.Module):

    def __init__(...):
        ...

    def extract_image_features(self, images):
        ...

    def project_features(self, features):
        ...

    def attention_pool(self, features, mask):
        ...

    def forward(self, images, mask):
        ...

    def hierarchical_probabilities(
        self,
        primary_logits,
        benign_logits,
        malignant_logits
    ):
        ...

    def extract_attention_weights(self, images, mask):
        ...
```

The implementation should keep these operations separate so that
debugging and analysis remain possible.

------------------------------------------------------------------------

# 47. Dataset Class Requirements

Conceptually:

``` python
class PatientBagDataset(Dataset):

    def __getitem__(self, index):
        return {
            "patient_id": ...,
            "images": ...,
            "primary_label": ...,
            "subtype_label": ...
        }
```

The DataLoader collate function must return:

``` python
images
mask
primary_labels
subtype_labels
patient_ids
image_paths
```

The mask indicates which entries correspond to real images.

------------------------------------------------------------------------

# 48. Hierarchical Label Mapping

Use a fixed mapping.

``` python
BENIGN_CLASSES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]

MALIGNANT_CLASSES = [
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]
```

The global eight-class order must remain fixed across all artifacts.

Never allow alphabetical ordering to silently change between
experiments.

------------------------------------------------------------------------

# 49. Primary/Subtype Consistency

The model must enforce the logical relationship:

``` text
adenosis
fibroadenoma
phyllodes_tumor
tubular_adenoma
    -> benign

ductal_carcinoma
lobular_carcinoma
mucinous_carcinoma
papillary_carcinoma
    -> malignant
```

Therefore:

``` text
P(Benign | subtype)
=
sum benign subtype probabilities

P(Malignant | subtype)
=
sum malignant subtype probabilities
```

This provides the consistency target.

------------------------------------------------------------------------

# 50. Final Model Configuration

Initial locked configuration:

``` python
CONFIG = {
    "data": {
        "magnification": "200X",
        "input_size": 224,
        "patient_batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
    },

    "backbone": {
        "name": "densenet201",
        "pretrained": True,
        "embedding_dim": 1920,
    },

    "projection": {
        "dim": 384,
        "dropout": 0.3,
    },

    "attention": {
        "attention_dim": 128,
    },

    "hierarchy": {
        "benign_classes": 4,
        "malignant_classes": 4,
        "primary_classes": 2,
    },

    "loss": {
        "lambda_primary": 1.0,
        "lambda_subtype": 1.0,
        "lambda_consistency": 0.1,
        "use_mild_class_weights": True,
    },

    "optimizer": {
        "name": "AdamW",
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
    },

    "training": {
        "phase1_epochs": 10,
        "phase2_epochs": 15,
        "early_stopping_patience": 4,
        "mixed_precision": True,
        "gradient_accumulation_steps": 4,
    },

    "cv": {
        "n_folds": 5,
    },

    "seed": 42,
}
```

The agent may reduce batch size if GPU memory requires it, but should
not change the architecture casually.

------------------------------------------------------------------------

# 51. Final Notebook Execution Order

The final notebook should execute in this order:

``` text
1. Environment setup
2. Drive mount
3. Configuration
4. Seed initialization
5. Dataset discovery
6. Dataset audit
7. Patient grouping
8. Patient split verification
9. Class distribution analysis
10. PatientBagDataset creation
11. DataLoader creation
12. DenseNet model initialization
13. Model architecture validation
14. Phase 1 training
15. Phase 1 validation
16. Best checkpoint selection
17. Partial DenseNet unfreezing
18. Phase 2 fine-tuning
19. Best checkpoint selection
20. Task A evaluation
21. Task B 5-fold evaluation
22. Prediction export
23. Attention export
24. Confusion matrices
25. Per-class F1
26. Error analysis
27. Comparison with previous baselines
28. Final experiment summary
```

------------------------------------------------------------------------

# 52. Validation Before Training

Before consuming the main compute budget, run a smoke test.

The smoke test must verify:

``` text
one patient bag loads
multiple images load
mask is correct
DenseNet output = 1920 dimensions
projection output = 384 dimensions
attention weights sum to 1
patient representation = 384 dimensions
primary output = 2
benign output = 4
malignant output = 4
final subtype output = 8
loss is finite
backward pass succeeds
```

Run only a few batches.

Do not begin the full training run until all assertions pass.

------------------------------------------------------------------------

# 53. GPU Safety

The implementation must:

``` text
use CUDA when available
use mixed precision
avoid unnecessary image duplication
avoid multiprocessing DataLoader workers
clear unused CUDA memory where appropriate
save checkpoints frequently
```

Use:

``` python
num_workers = 0
```

because stability is more important than a small data-loading speed
improvement on Colab.

------------------------------------------------------------------------

# 54. Checkpointing

Save:

``` text
best_final_model.pth
last_checkpoint.pth
```

Best model selection:

``` text
validation subtype Macro-F1
```

Do not select based on test metrics.

Checkpoint should contain:

``` text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
best_metric
CONFIG
random_seed
```

------------------------------------------------------------------------

# 55. Final Paper Positioning

The paper should not claim:

``` text
"we invented hierarchical classification"
```

or:

``` text
"we invented MIL"
```

The contribution must be framed around the complete experimental
investigation and the specific patient-aware hierarchical formulation,
subject to verification against the literature.

A defensible framing is:

``` text
Existing image-level CNN and Transformer approaches
        ↓
CNN-ViT fusion
        ↓
Adaptive fusion
        ↓
Observed failure to improve subtype Macro-F1
        ↓
Patient-aware hierarchical formulation
        ↓
Explicit exploitation of:
    - patient-level image grouping
    - attention-based aggregation
    - benign/malignant hierarchy
    - multi-task consistency
```

The exact novelty claim must be checked against the papers already
reviewed before manuscript submission.

------------------------------------------------------------------------

# 56. Final Research Comparison

The final paper should contain a table similar to:

  Model                                Primary Accuracy   Primary F1   Subtype Macro-F1   Std
  ---------------------------------- ------------------ ------------ ------------------ -----
  DenseNet-201                                      ...          ...                ...   ...
  ViT-B/16                                          ...          ...                ...   ...
  Static CNN+ViT                                    ...          ...                ...   ...
  Gated CNN+ViT + CB                                ...          ...                ...   ...
  Patient-Aware Hierarchical Model                  ...          ...                ...   ...

Do not omit unsuccessful models.

The unsuccessful fusion models are important evidence supporting the
final architectural decision.

------------------------------------------------------------------------

# 57. Final Decision Rule

After the final experiment:

## Case A: Final model clearly improves subtype Macro-F1

Use it as the proposed model.

Perform:

``` text
error analysis
ablation using existing experiments
attention analysis
statistical comparison
literature comparison
paper writing
```

Do not start another architecture.

## Case B: Final model improves minority classes but not overall Macro-F1

Analyze the trade-off.

If the result provides a meaningful balanced improvement but not a
higher aggregate score, report the finding honestly.

Do not cherry-pick minority classes.

## Case C: Final model does not improve over ViT/DenseNet

Stop training.

The conclusion becomes that the available BreakHis 200X data and
patient-level evaluation constrain the achievable improvement, and
increasingly complex architectures did not provide a reliable advantage.

At that point the project should move to:

``` text
analysis
ablation
literature comparison
limitations
paper writing
```

not another uncontrolled model.

------------------------------------------------------------------------

# 58. What Must Not Change

The final implementation must not silently change:

``` text
dataset
magnification
patient-level separation
test set
subtype definitions
evaluation metrics
random seed
```

unless the change is explicitly documented as a new experiment.

Do not alter the test split after seeing results.

Do not tune against the test set.

Do not select the architecture using test performance.

------------------------------------------------------------------------

# 59. Final Implementation Checklist

Before training:

``` text
[ ] BreakHis 200X verified
[ ] Patient IDs correctly parsed
[ ] Patient-level split verified
[ ] No patient leakage
[ ] Rare subtype patient counts documented
[ ] PatientBagDataset verified
[ ] Variable-length collate verified
[ ] Attention mask verified
[ ] DenseNet output verified
[ ] Projection output verified
[ ] Attention weights sum to 1
[ ] Primary head verified
[ ] Benign expert verified
[ ] Malignant expert verified
[ ] Hierarchical probability calculation verified
[ ] Consistency loss verified
[ ] Loss is finite
[ ] Mixed precision verified
[ ] Checkpointing verified
```

During training:

``` text
[ ] Train loss logged
[ ] Validation loss logged
[ ] Primary Macro-F1 logged
[ ] Subtype Macro-F1 logged
[ ] Per-class F1 logged
[ ] Best checkpoint selected only from validation
[ ] No test evaluation used for model selection
```

After training:

``` text
[ ] Held-out Task A evaluation completed
[ ] Task B 5-fold evaluation completed
[ ] Mean ± std reported
[ ] Per-class metrics exported
[ ] Confusion matrices exported
[ ] Patient predictions exported
[ ] Attention weights exported
[ ] Error analysis completed
[ ] Baseline comparison generated
[ ] Final conclusion recorded
```

------------------------------------------------------------------------

# 60. Final Instruction to the Implementation Agent

Implement exactly one final architecture:

``` text
Pretrained DenseNet-201
        ↓
Image embeddings
        ↓
384-d projection
        ↓
Gated attention MIL
        ↓
Patient representation
        ↓
        ├── Primary 2-class head
        │
        └── Hierarchical subtype
              ├── Benign 4-class expert
              └── Malignant 4-class expert
        ↓
Soft hierarchical probability
        ↓
Consistency constraint
```

The architecture must exploit patient-level grouping and the
benign/malignant subtype hierarchy.

Do not add ViT.

Do not add an ensemble.

Do not add another fusion mechanism.

Do not add another backbone.

Do not perform a broad hyperparameter search.

Do not create another experimental baseline.

This is the final planned model and the remaining compute should be
dedicated to implementing, validating, training, and analyzing this
architecture rigorously.
