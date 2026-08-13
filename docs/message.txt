## Mock mentor presentation structure

The five examples broadly use the same research narrative—**motivation → literature → gap → methodology → proposed model → experimental setup → results → discussion → conclusion**—but the strongest examples additionally make the **dataset, ablations, comparisons, interpretability, and generalization experiments** explicit.  

For OMNet-V3, I would structure the mentor presentation as **14 slides**. This is more suitable for a project review than copying a paper section-for-section.

---

# Slide 1 — Title

### **OMNet-V3**

**Magnification-Aware Adaptive CNN–ViT Fusion for Breast Histopathology Classification**

**Subtitle:**
A patient-disjoint, hierarchical and stain-robust deep learning framework using BreaKHis

**Team / Members:**
`[Names]`

**Guide / Mentor:**
`[Name]`

**Status:**
**Research proposal / experimental development — results pending**

Do not put accuracy or performance claims here; the original project specification explicitly keeps results and conclusions pending experimentation. 

---

# Slide 2 — Problem Statement

### The problem

Breast histopathology classification requires recognizing both:

* **benign vs malignant tissue**
* **specific pathological subtypes**

The difficulty is that useful information exists at different visual scales:

**Low-level / local**

* nuclei
* cellular morphology
* texture
* fine tissue structures

**High-level / global**

* tissue organization
* spatial relationships
* architectural context

Additionally, BreaKHis images are provided at **40×, 100×, 200× and 400× magnifications**, meaning that the same pathology can present different visual characteristics depending on scale. 

### Core research problem

> **Can we build a lightweight model that combines local and global representations while explicitly adapting their contribution to histological magnification?**

---

# Slide 3 — Why BreaKHis?

### Dataset

**BreaKHis v1**

* **7,909 images**
* **82 patients**
* 24 benign patients
* 58 malignant patients
* 8 pathological subtypes
* 40× / 100× / 200× / 400×
* RGB
* 700 × 460 pixels
* Surgical Open Biopsy (SOB)

The eight classes form a natural hierarchy:

```text
                    Breast Histopathology
                            │
                ┌───────────┴───────────┐
              Benign                 Malignant
                │                         │
       ┌────────┼────────┐        ┌───────┼───────┐
       A        F       PT       DC       LC      MC      PC
```

The project specification explicitly treats this hierarchy as part of the learning problem. 

### Why it is useful for our study

BreaKHis simultaneously provides:

* **class imbalance**
* **multiple magnifications**
* **multiple images per patient**
* **fine-grained subtype classification**
* H&E staining variability

Therefore, it allows us to study more than raw classification accuracy.

---

# Slide 4 — Challenges in BreaKHis

### 1. Multi-scale morphology

Different magnifications reveal different levels of tissue structure.

### 2. Class imbalance

Some subtypes contain substantially more images than others.

### 3. Patient correlation

Multiple images can originate from the same patient.

### 4. Staining variation

H&E appearance can vary between specimens/acquisition conditions.

### 5. Fine-grained classes

Some pathological subtypes can have visually similar characteristics.

### 6. Limited patient count

There are only **82 patients**, despite 7,909 images.

This means that treating every image as an independent observation risks producing an overly optimistic evaluation.

The current OMNet-V3 specification therefore requires patient-disjoint grouping and class-weighted learning. 

---

# Slide 5 — What Existing Approaches Do

Use this slide as the condensed literature review.

### CNN-based approaches

**Strengths**

* excellent local feature extraction
* hierarchical spatial features
* computationally efficient
* strong transfer learning

**Limitation**

* weaker explicit modelling of long-range spatial relationships

### ViT-based approaches

**Strengths**

* self-attention
* global contextual relationships
* long-range interactions

**Limitations**

* data/pretraining sensitivity
* computational cost
* potentially weaker local inductive bias

This local-vs-global distinction also appears repeatedly in the example papers. 

---

# Slide 6 — Existing Hybrid / Related Approaches

The important message is:

> **CNN + Transformer fusion itself is not our novelty.**

The examples demonstrate several types of hybrid approaches:

* CNN + sequence/other learned representations
* ViT + handcrafted local features
* CNN + Transformer dual branches
* hierarchical Transformer systems
* multimodal/domain-adaptive systems

For example, the supplied papers explicitly discuss combining local and global representations and feature fusion.  

### Therefore our question becomes:

> **What remains insufficiently addressed for BreaKHis?**

---

# Slide 7 — Research Gap

### Targeted gaps

| Gap                  | Existing limitation                                                    | OMNet-V3 response            |
| -------------------- | ---------------------------------------------------------------------- | ---------------------------- |
| Local/global balance | CNN and ViT emphasize different information                            | CNN + ViT fusion             |
| Magnification        | Often treated as separate condition rather than contextual information | Magnification embedding      |
| Fusion               | Fixed/naive fusion cannot adapt dynamically                            | Adaptive fusion gate         |
| Hierarchy            | Flat subtype prediction ignores benign/malignant structure             | Binary + subtype heads       |
| Leakage              | Image-level splitting can correlate patients across sets               | Patient-disjoint CV          |
| Stain variability    | Standard augmentation may not explicitly target H&E variation          | Macenko stain perturbation   |
| Interpretability     | Classification score alone does not explain model focus                | Grad-CAM + attention rollout |

**Important:** these are **targeted research gaps**, not yet proven literature claims; the final paper must verify the literature overlap before making a novelty claim. Your original specification already requires this verification. 

---

# Slide 8 — Research Objective

### Primary objective

> **Evaluate whether magnification-conditioned adaptive fusion of local CNN and global ViT representations improves patient-disjoint breast histopathology classification on BreaKHis.**

### Secondary objectives

* Determine whether hierarchical learning improves subtype classification.
* Determine whether magnification information improves fusion.
* Evaluate robustness to staining variation.
* Analyze whether the CNN and ViT branches exhibit complementary behavior.
* Evaluate patient-level rather than only image-level performance.
* Establish reproducible experimental evidence rather than relying on a single accuracy figure.

This follows the project's existing research question but narrows it into a more experimentally testable primary objective. 

---

# Slide 9 — Research Questions / Hypotheses

### RQ1

Does CNN–ViT fusion outperform either branch individually?

### RQ2

Does **magnification-aware adaptive fusion** outperform fixed or magnification-agnostic fusion?

### RQ3

Does hierarchical binary + subtype learning improve 8-class classification?

### RQ4

Does stain augmentation reduce performance degradation under stain variation?

### RQ5

Does the learned fusion behavior vary systematically across magnifications?

### RQ6

Does the improvement remain at the **patient level**?

### Hypotheses

* **H1:** local-global fusion improves 8-class performance.
* **H2:** magnification conditioning improves fusion.
* **H3:** hierarchical learning improves subtype recognition/consistency.
* **H4:** stain augmentation improves robustness under stain-shift evaluation.
* **H5:** fusion weights differ meaningfully across magnifications.

Patient-disjoint validation itself is treated as an **evaluation protocol**, rather than a model hypothesis.

---

# Slide 10 — Proposed Solution: OMNet-V3

### Architecture

```text
                  BreaKHis Image
                        +
                  Magnification
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       EfficientNet-B0         ViT-Tiny
          Local              Global Context
             │                   │
          1280-D                192-D
             │                   │
             ▼                   ▼
           256-D               256-D
              └─────────┬─────────┘
                        │
               Magnification
                  Embedding
                        │
                        ▼
        Magnification-Aware Adaptive
                  Fusion
                        │
                        ▼
             Residual Enrichment
                        │
                ┌───────┴───────┐
                ▼               ▼
           Binary Head      8-Class Head
           Benign/Malignant   Subtypes
```

The locked architecture specifies EfficientNet-B0 and ViT-Tiny, each projected to 256 dimensions, plus a 4×64 magnification embedding. 

---

# Slide 11 — The Core Innovation: MAF

### Magnification-Aware Adaptive Fusion

Inputs:

[
f_{cnn} \in \mathbb{R}^{256}
]

[
f_{vit} \in \mathbb{R}^{256}
]

[
e_m \in \mathbb{R}^{64}
]

Concatenate:

[
[f_{cnn};f_{vit};e_m]
]

Then:

[
\alpha=\sigma(W[f_{cnn};f_{vit};e_m]+b)
]

Fusion:

[
f_{fused}=
\alpha f_{cnn}+(1-\alpha)f_{vit}
]

### Intended behavior

Instead of:

> CNN + ViT → fixed combination

we use:

> **CNN + ViT + magnification → learned adaptive combination**

The architecture and dimensions are explicitly locked in the project specification. 

---

# Slide 12 — Hierarchical Learning

### Two simultaneous predictions

**Head 1**

[
256 \rightarrow 2
]

Benign / Malignant

**Head 2**

[
256 \rightarrow 8
]

Eight pathological subtypes

### Total objective

[
L_{total}
=========

0.3L_{binary}
+
0.6L_{subtype}
+
0.1L_{consistency}
]

The current design gives the subtype objective the largest weight while using the binary task as hierarchical regularization. 

### What we need to verify experimentally

* Does hierarchy improve macro-F1?
* Does it reduce biologically inconsistent predictions?
* Does binary performance improve or remain stable?
* Does the consistency term contribute independently?

---

# Slide 13 — Data Pipeline & Leakage Prevention

### Training

```text
700×460 RGB
      ↓
Resize 256×256
      ↓
Random resized crop 224×224
      ↓
Horizontal flip
      ↓
Vertical flip
      ↓
90° rotation
      ↓
Macenko stain perturbation
      ↓
ImageNet normalization
```

### Validation / Test

```text
Resize 256×256
      ↓
Center crop 224×224
      ↓
ImageNet normalization
```

The locked specification explicitly requires all images belonging to one patient to remain within one partition. 

### Split

**5-fold StratifiedGroupKFold**

* group = patient ID
* stratification = subtype
* ~10% of training patients → validation
* test contains no images from training patients

### Why?

Because thousands of images do **not** represent thousands of independent patients.

---

# Slide 14 — Experimental Plan

This slide is especially important for the mentor.

### Phase 1 — Baselines

| Experiment              | Purpose              |
| ----------------------- | -------------------- |
| EfficientNet-B0         | Local-only baseline  |
| ViT-Tiny                | Global-only baseline |
| CNN + ViT concatenation | Naive hybrid         |
| Fixed fusion            | Fusion baseline      |
| OMNet-V3                | Proposed model       |

### Phase 2 — Ablation

| Variant                       | Removed component              |
| ----------------------------- | ------------------------------ |
| OMNet-V3 − magnification      | Magnification embedding        |
| OMNet-V3 − hierarchy          | Binary/subtype joint objective |
| OMNet-V3 − stain augmentation | Macenko                        |
| OMNet-V3 − adaptive gate      | Fixed fusion                   |

### Phase 3 — Analysis

* magnification-wise performance
* patient-level performance
* fusion α distributions
* hierarchical consistency
* interpretability
* error cases

The supplied papers particularly reinforce the value of **ablation and comparative studies** as explicit experimental sections. 

---

# Slide 15 — Evaluation Metrics

### Primary

**8-class Macro-F1**

Why:

* strong class imbalance
* prevents majority classes dominating the headline result

### Secondary

* Accuracy
* Weighted-F1
* Balanced Accuracy
* Macro-AUC
* MCC
* Precision
* Recall
* Per-class F1

### Additional

* Binary performance
* Patient-level performance
* Per-magnification performance
* confusion matrices

The original OMNet-V3 specification already defines this measurement set. 

---

# Slide 16 — What Results Will We Produce?

### Main result table

| Model           | Binary F1 | 8-Class Macro-F1 | Balanced Acc. | Macro-AUC |     MCC |
| --------------- | --------: | ---------------: | ------------: | --------: | ------: |
| EfficientNet-B0 |       TBD |              TBD |           TBD |       TBD |     TBD |
| ViT-Tiny        |       TBD |              TBD |           TBD |       TBD |     TBD |
| CNN+ViT         |       TBD |              TBD |           TBD |       TBD |     TBD |
| Fixed Fusion    |       TBD |              TBD |           TBD |       TBD |     TBD |
| **OMNet-V3**    |   **TBD** |          **TBD** |       **TBD** |   **TBD** | **TBD** |

### Statistical evidence

Report:

* five-fold mean
* fold standard deviation
* 95% CI
* statistical comparison where appropriate

The existing manuscript deliberately leaves these values empty until actual experiments are completed. 

---

# Slide 17 — Magnification Analysis

### Four conditions

* 40×
* 100×
* 200×
* 400×

### We will measure

* Accuracy
* Macro-F1
* Balanced Accuracy
* Macro-AUC
* patient-level performance
* mean fusion α

### Key question

> **Does the model's local/global representation balance change with magnification?**

This is one of the central analyses already specified in OMNet-V3. 

### Planned figure

```text
Fusion α
  │
1 │                ●
  │        ●
  │   ●
0 │________________________
     40  100  200  400×
```

The actual pattern is **TBD**.

---

# Slide 18 — Interpretability

### CNN branch

**Grad-CAM**

Shows regions associated with CNN feature activation.

### ViT branch

**Attention rollout**

Visualizes Transformer attention across image patches.

### Fused representation

**t-SNE / equivalent embedding**

Color by:

* subtype
* magnification

### Why?

We want to investigate whether the branches appear to provide complementary information rather than simply producing redundant features.

These exact analyses are already part of the intended OMNet-V3 evaluation. 

---

# Slide 19 — Error Analysis

We will inspect representative:

* false positives
* false negatives
* subtype confusions
* low-confidence predictions
* unusual α/fusion behavior
* minority-class failures

### Questions

* Are errors concentrated in minority classes?
* Are particular subtype pairs consistently confused?
* Does magnification influence failure?
* Does the CNN or ViT dominate problematic cases?
* Does stain perturbation expose specific weaknesses?

The final paper should connect these observations back to the quantitative results rather than presenting isolated examples. 

---

# Slide 20 — Robustness / Generalization

### Internal robustness

Train with:

**Macenko stain perturbation**

Then compare:

* no stain augmentation
* stain augmentation

### Stronger test

Introduce controlled stain perturbation during **evaluation** and compare performance degradation.

### External validation — optional

Potential future/extended experiment:

> Train on BreaKHis → evaluate on another compatible histopathology dataset.

### Important distinction

The project should not claim domain generalization merely because stain augmentation is used.

Actual cross-dataset generalization must come from **external evaluation**.

---

# Slide 21 — Computational Efficiency

Because “lightweight” is part of the intended motivation, measure:

| Measurement     | EfficientNet | ViT | OMNet-V3 |
| --------------- | -----------: | --: | -------: |
| Parameters      |          TBD | TBD |      TBD |
| FLOPs           |          TBD | TBD |      TBD |
| Peak VRAM       |          TBD | TBD |      TBD |
| Training time   |          TBD | TBD |      TBD |
| Inference/image |          TBD | TBD |      TBD |

The objective is not simply:

> “highest accuracy wins.”

It is:

> **performance + robustness + patient-level reliability + interpretability + reasonable computational cost.**

---

# Slide 22 — Expected Outcome

### What would constitute success?

Not merely:

> “OMNet-V3 gets high accuracy.”

Instead, we want evidence that:

1. **Fusion improves over individual branches.**
2. **Magnification conditioning improves over equivalent fusion without magnification.**
3. **Hierarchical learning contributes to subtype classification or consistency.**
4. **Stain augmentation reduces degradation under stain variation.**
5. **Patient-level performance remains strong under patient-disjoint evaluation.**
6. **Fusion behavior shows interpretable variation across magnifications.**
7. Improvements are supported by **statistical evidence**, not just one favorable fold.

These are **success criteria**, not current results.

---

# Slide 23 — Limitations We Already Recognize

### Dataset

* only **82 patients**
* class imbalance
* limited rare-class statistical power

### Metadata

* magnification is supplied as metadata
* model does not independently infer magnification

### Evaluation

* primary evaluation is still one dataset
* external validation may be unavailable

### Computation

* student-scale computational resources
* potentially limited number of random seeds

### Clinical translation

**BreaKHis alone cannot establish clinical readiness.**

The existing project specification explicitly identifies these limitations and prohibits clinical-deployment claims from BreaKHis alone. 

---

# Slide 24 — Reproducibility

### Environment

* Google Colab
* BreaKHis on Google Drive
* fixed seed: **42**
* AMP / FP16

### Training

* AdamW
* backbone LR = **1e-4**
* head LR = **5e-4**
* weight decay = **1e-4**
* cosine scheduler
* 5-epoch warmup
* batch size = **32**, subject to VRAM
* maximum epochs = **50**
* early stopping patience = **10**
* gradient clipping = **1.0**

### Saved artifacts

* per-fold checkpoints
* configuration
* metrics
* figures
* self-contained notebook

These are already part of the locked experiment configuration. 

---

# Slide 25 — Current Status

Use this as the final mentor-review slide.

| Component                   | Status                 |
| --------------------------- | ---------------------- |
| Problem definition          | **Complete**           |
| Dataset selection           | **Complete**           |
| Architecture design         | **Complete**           |
| Training configuration      | **Locked**             |
| Leakage-prevention strategy | **Locked**             |
| Research hypotheses         | **Defined**            |
| Literature-gap validation   | **In progress**        |
| Baseline implementation     | **Pending**            |
| OMNet-V3 implementation     | **Pending / ongoing**  |
| Five-fold experiments       | **Pending**            |
| Ablations                   | **Pending**            |
| Statistical analysis        | **Pending**            |
| Interpretability            | **Pending**            |
| Error analysis              | **Pending**            |
| External validation         | **Optional / pending** |
| Final conclusions           | **Not yet available**  |

This is consistent with the original document's explicit rule that all empirical sections remain incomplete until experiments are finished. 

---

# Slide 26 — Closing: Research Pipeline

```text
Literature
   ↓
Research Gap
   ↓
Research Questions
   ↓
Hypotheses
   ↓
BreaKHis + Patient-Disjoint Split
   ↓
Baselines
   ↓
OMNet-V3
   ↓
Ablation Experiments
   ↓
Five-Fold Evaluation
   ↓
Statistical Analysis
   ↓
Magnification / Fusion Analysis
   ↓
Interpretability
   ↓
Error Analysis
   ↓
Optional External Validation
   ↓
Evidence-Based Discussion
   ↓
Conclusion
```

This follows the evidence flow already defined for OMNet-V3. 

---

## How the example papers influenced this structure

| Pattern from examples                                          | OMNet-V3 adaptation                                |
| -------------------------------------------------------------- | -------------------------------------------------- |
| **Introduction → Methods → Results → Discussion → Conclusion** | Core presentation structure                        |
| Dataset/challenge analysis                                     | Dedicated BreaKHis slide                           |
| Explicit research gaps                                         | Dedicated gap slide                                |
| Proposed-method architecture                                   | Dedicated OMNet-V3 architecture slides             |
| Mathematical formulation                                       | Dedicated MAF/loss slide                           |
| Experimental setup                                             | Dedicated protocol slide                           |
| Comparative experiments                                        | Baseline slide                                     |
| **Ablation studies**                                           | Dedicated ablation plan                            |
| Per-class analysis                                             | Metrics/results                                    |
| Visualization / interpretability                               | Grad-CAM + attention rollout                       |
| Domain adaptation/generalization                               | Stain robustness + optional external validation    |
| Results + Discussion                                           | Evidence → interpretation rather than raw accuracy |

The strongest example for **presentation flow** is essentially the first paper's structure: it explicitly moves from materials/methods to quantitative and qualitative results, then discussion and conclusion.  The third example is particularly useful for us because it explicitly separates **experimental setup, comparative studies, and ablation studies**. 

### One important adjustment for our presentation

I would **not present OMNet-V3 as a proven superior model**. Present it as:

> **“Our proposed architecture and experimental agenda.”**

That distinction is consistent with the current project document and will make the mentor discussion substantially more credible. 

**Confidence: 96%.**

**Improved follow-up prompt:**

> “Turn this mentor-review structure into a 12–15 slide professional presentation, using the visual hierarchy and section style of the supplied papers, with speaker notes and explicit TBD placeholders for every experimental result.”
