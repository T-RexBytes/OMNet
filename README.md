# Breast Cancer Research Project

> **This project focuses on developing a reproducible and reliable AI-based breast cancer image classification system. The work includes dataset verification, data preparation, baseline development, proposed model testing, leakage-safe evaluation, ablation studies, explainability, and publication-quality result analysis.**

---

## Table of Contents

1. [Project Mission & Research Agenda](#-project-mission--research-agenda)
2. [Key Highlights & Findings](#-key-highlights--findings)
3. [Repository File Structure](#-repository-file-structure)
4. [Model Evolution (V1 → V2 → V3 → V4)](#-model-evolution-v1--v2--v3--v4)
5. [Dataset Overview: BreakHis](#-dataset-overview-breakhis)
6. [Core Architecture: OMNet-V3](#-core-architecture-omnet-v3)
7. [Benchmark Results Summary](#-benchmark-results-summary)
8. [Interpretability & Diagnostic Explainability](#-interpretability--diagnostic-explainability)
9. [Installation & How to Run](#-installation--how-to-run)
10. [Research Protocols & Anti-Leakage Guardrails](#-research-protocols--anti-leakage-guardrails)
11. [License & Citation](#-license--citation)

---

## 🔬 Project Mission & Research Agenda

Histopathological examination of breast tissue biopsy slides is the gold standard for clinical breast cancer diagnosis. However, manual examination is labor-intensive and subject to inter-observer variability. 

While deep learning has shown immense promise in medical imaging, standard literature benchmarks on datasets like **BreakHis** often suffer from **patient-level data leakage** (mixing images from the same patient across training and testing sets), leading to artificially inflated accuracy numbers (>98%) that fail in real-world clinical practice.

### Core Objectives of This Research
- **Zero-Leakage Evaluation:** Ensure all patient tissue slices are strictly partitioned by patient identity (`patient_id`), guaranteeing true clinical generalization.
- **Hybrid Local-Global Representations:** Unify localized cellular/nuclear morphological feature extraction (via CNNs) with wide-field tissue architecture analysis (via Vision Transformers).
- **Scale-Conditioned Adaptive Fusion:** Learn dynamic, magnification-aware fusion gates ($\alpha(m)$) that adapt feature blending across different optical zoom levels ($40\text{X}, 100\text{X}, 200\text{X}, 400\text{X}$).
- **Hierarchical Pathology Learning:** Jointly optimize coarse binary malignancy detection (Benign vs. Malignant) and fine-grained 8-class histological subtyping with Kullback-Leibler (KL) probability consistency regularization.
- **Traceable, Publication-Ready Evidence:** Deliver 100% verifiable metrics, diagnostic ROC/PR curves, confusion matrices, Grad-CAM attention maps, and t-SNE latent manifold embeddings.

---

## 🌟 Key Highlights & Findings

- **Leak-Free Benchmark Calibration:** When patient leakage is eliminated, true state-of-the-art multi-scale accuracy on BreakHis is **84.56% ± 2.48%**, establishing an honest and reproducible baseline for future research.
- **High Sensitivity for Cancer Screening:** The Vision Transformer architecture achieves **97.45% sensitivity** on unseen test patients, ensuring critical malignant cases are not missed.
- **Ensemble Specificity Gains:** Combining CNN and Transformer posteriors boosts specificity to **55.95%** and Precision-Recall AUC to **0.9537**, mitigating false positives on complex benign lesions.
- **Autonomous Scale Adaptation:** The Magnification-Aware Adaptive Fusion (MAF) module automatically shifts reliance from global transformer context at $40\text{X}$ ($\alpha = 0.685$) to localized cellular morphology at $400\text{X}$ ($\alpha = 0.470$).

---

## 📂 Repository File Structure

The workspace is organized into modular components separating model notebooks, documentation packs, raw outputs, and the unified conference-ready research package:

```text
OMNet/
│
├── README.md                               # Project documentation & master guide
├── LICENSE                                 # Open-source license (MIT)
├── requirements.txt                        # Python package dependencies
├── work.md                                 # Research audit guidelines & operating principles
├── knowledge.md                            # High-level research knowledge base
├── notebook_guidelines.md                  # Notebook implementation guardrails & locked decisions
│
├── docs/                                   # Process flow diagrams & reference checklists
│   ├── OMNet-V3_Final_Paper_Master_Checklist.md
│   ├── process_flow_v2.md
│   ├── process_flow_v3.md
│   ├── message.md
│   └── message.txt
│
├── models/                                 # Jupyter notebooks & model documentation
│   ├── OMNet_BreakHis_V1.ipynb             # V1: 4-block custom CNN baseline from scratch
│   ├── OMNet_BreakHis_V2.ipynb             # V2: Transfer learning (EfficientNet-B0 + ViT-B/16)
│   ├── OMNet_BreakHis_V3.ipynb             # V3: Multi-task dual-branch fusion (Cloud/Kaggle)
│   ├── OMNet_BreakHis_V3_local.ipynb       # V3: Multi-task dual-branch fusion (Local ROCm/CUDA)
│   ├── OMNet_V4.ipynb                      # V4: Multi-dataset cross-domain pipeline
│   ├── OMNet_V4_Local(breakhis_only).ipynb # V4: Cleaned BreakHis-only local pipeline
│   │
│   ├── baseline/                           # Diagnostic baseline notebooks & plans
│   │   ├── baseline_1_densenet201.ipynb
│   │   ├── baseline_2_vit_b16.ipynb
│   │   ├── baseline_3_fusion.ipynb
│   │   └── plan/
│   │
│   ├── info_V2/                            # V2 Specific Documentation Pack
│   │   ├── architecture_V2.md              # End-to-end V2 architecture specification
│   │   ├── dataset_V2.md                   # 200X dataset breakdown & split manifest
│   │   ├── knowledge_V2.md                 # Research framing & literature comparisons
│   │   └── output_V2.md                    # Experimental results & figure catalog
│   │
│   ├── info_V3/                            # V3 Cloud Specific Documentation Pack
│   │   ├── architecture_V3.md              # Dual-branch MAF architecture specification
│   │   ├── dataset_V3.md                   # Full multi-scale dataset documentation
│   │   ├── knowledge_V3.md                 # Scientific hypotheses & paper guide
│   │   └── output_V3.md                    # 5-Fold cross-validation metrics & logs
│   │
│   ├── info_V3_local/                      # V3 Local Run Specific Documentation Pack
│   │   ├── architecture_V3.md              # Local execution architecture details
│   │   ├── dataset_V3.md                   # Metadata distribution & split specs
│   │   ├── knowledge_V3.md                 # Local benchmark & theoretical context
│   │   └── output_V3.md                    # Per-fold logs from ROCm execution
│   │
│   └── info_V4/                            # V4 Specific Documentation Pack
│       ├── architecture_V4.md              # V4 pipeline specification
│       ├── dataset_V4.md                   # BreakHis data documentation
│       └── knowledge_V4.md                 # Research reference document
│
├── result_OMNet/                           # Raw experimental output folders
│   ├── v2_out/                             # OMNet-V2 raw CSVs, JSONs, and 10 PNG plots
│   │   ├── cv_fold_results.csv             # Per-fold cross-validation logs
│   │   ├── cv_summary.csv                  # CV mean and standard deviation
│   │   ├── experiment_summary.json         # Complete configuration and test metrics
│   │   ├── test_metrics_all_models.csv     # Metrics for EfficientNet, ViT, & Ensemble
│   │   ├── test_predictions_detailed.csv   # Image-level predictions with probabilities
│   │   └── *.png                           # ROC, PR, confusion matrices, t-SNE, error plots
│   │
│   ├── V3_out/                             # OMNet-V3 cloud run raw text logs and plots
│   │   ├── result.txt                      # 5-Fold binary and subtype metrics
│   │   ├── summary.txt                     # Execution parameters and summary
│   │   ├── fusion.txt                      # Scale gating alpha progression across 40X-400X
│   │   └── *.png                           # Binary & subtype ROC, confusion matrices, t-SNE
│   │
│   └── V3_local_out/                       # OMNet-V3 local workstation run outputs
│       ├── aggregated_results.json         # Mean ± std across 5 validation folds
│       ├── metrics_summary.csv             # Per-fold validation metrics table
│       ├── experiment_config.json          # Complete JSON run configuration
│       ├── dataset_metadata.csv            # Full metadata for all 7,909 images
│       └── *.png                           # Local confusion matrices, ROC, and t-SNE
│
└── research/                               # Unified Conference-Ready Research Package
    ├── 01_model_analysis/                  # Model cards, inventory, architecture comparisons
    ├── 02_dataset_analysis/                # Multi-scale statistics, distributions, stain analysis
    ├── 03_training_analysis/               # Training histories, convergence curves, stability
    ├── 04_performance_results/             # Master metrics dictionary and comparison tables
    ├── 05_figures/                         # 21 Organized publication-quality figure assets
    ├── 06_tables/                          # Main results table, CV summary, and supplementary tables
    ├── 07_paper_content/                   # Results & Discussion draft, captions, showcase
    ├── 08_final_summary/                   # Executive summary, figure ranking, limitations
    ├── final_output_manifest.md            # Complete catalog of all 67 research files
    └── traceability_matrix.md              # 100% Verifiable claim-to-file traceability table
```

---

## 🔄 Model Evolution (V1 → V2 → V3 → V4)

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    OMNet-V1     │       │    OMNet-V2     │       │    OMNet-V3     │       │    OMNet-V4     │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ • Custom 4-block│       │ • Pretrained    │       │ • End-to-End    │       │ • Multi-Dataset │
│   CNN (Scratch) │ ────► │   EfficientNet  │ ────► │   Dual-Branch   │ ────► │   Cross-Domain  │
│ • 400X Scale    │       │   & ViT-B/16    │       │   (CNN + ViT)   │       │   Generalization│
│ • Binary Task   │       │ • 200X Scale    │       │ • 40X-400X Multi│       │ • IDC & BreakHis│
│ • Baseline test │       │ • Soft Ensemble │       │ • Multi-Task    │       │ • Robustness    │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Detailed Generation Breakdown

| Generation | Architecture & Backbones | Dataset Scope | Key Innovations | Status & Role |
|---|---|---|---|---|
| **OMNet-V1** | 4-block custom CNN trained from scratch ($32 \rightarrow 256$ ch) | BreakHis 400X ($N=1,820$) | Established initial reproducible data loading and training pipeline. | Baseline diagnostic predecessor. |
| **OMNet-V2** | ImageNet-pretrained **EfficientNet-B0** (1280-d) & **ViT-B/16** (768-d) | BreakHis 200X ($N=2,013$, 82 patients) | 2-Phase progressive fine-tuning, 2-layer bottleneck projection ($D=256$), and soft-voting ensemble. | Comprehensive transfer-learning benchmark. |
| **OMNet-V3** | **EfficientNet-B0 + ViT-Tiny/16** Dual-Branch with **MAF Module** | BreakHis Multi-Scale ($40\text{X}-400\text{X}$, $N=7,909$) | Scale embedding ($e_m \in \mathbb{R}^{64}$), dynamic gating ($\alpha \in (0, 1)$), multi-task binary+subtype heads, and hierarchical KL-consistency loss. | **Primary Proposed Flagship Architecture.** |
| **OMNet-V4** | Multi-Task Hybrid Architecture with Cross-Domain Evaluation | BreakHis + External Cohorts (IDC) | Evaluates out-of-distribution stain and cross-dataset domain robustness. | Research extension pipeline. |

---

## 📊 Dataset Overview: BreakHis

The **Breast Cancer Histopathological Image Database (BreakHis)** consists of clinical biopsy samples collected at the P&D Laboratory in Brazil:

```
BreakHis Multi-Scale Database (7,909 Total Images / 82 Clinical Patients)
│
├── Benign (Class 0): 2,480 images (31.36%) / 24 patients
│   ├── Adenosis (A)            : 444 images (5.61%),   4 patients
│   ├── Fibroadenoma (F)        : 1,014 images (12.82%), 10 patients
│   ├── Phyllodes Tumor (PT)    : 453 images (5.73%),   3 patients
│   └── Tubular Adenoma (TA)    : 569 images (7.19%),   7 patients
│
└── Malignant (Class 1): 5,429 images (68.64%) / 58 patients
    ├── Ductal Carcinoma (DC)   : 3,451 images (43.63%), 38 patients
    ├── Lobular Carcinoma (LC)  : 626 images (7.91%),   5 patients
    ├── Mucinous Carcinoma (MC) : 792 images (10.01%),  9 patients
    └── Papillary Carcinoma (PC): 560 images (7.08%),   6 patients
```

### Optical Magnification Distribution

| Histological Subtype | Super-Class | 40X Count | 100X Count | 200X Count | 400X Count | Total Images | Patient Count |
|---|---|---|---|---|---|---|---|
| **Adenosis (A)** | Benign | 114 | 113 | 111 | 106 | 444 | 4 |
| **Fibroadenoma (F)** | Benign | 253 | 260 | 253 | 248 | 1,014 | 10 |
| **Phyllodes Tumor (PT)** | Benign | 109 | 121 | 121 | 102 | 453 | 3 |
| **Tubular Adenoma (TA)** | Benign | 149 | 150 | 138 | 132 | 569 | 7 |
| **Ductal Carcinoma (DC)** | Malignant | 864 | 903 | 883 | 801 | 3,451 | 38 |
| **Lobular Carcinoma (LC)** | Malignant | 156 | 170 | 156 | 144 | 626 | 5 |
| **Mucinous Carcinoma (MC)**| Malignant | 205 | 222 | 211 | 154 | 792 | 9 |
| **Papillary Carcinoma (PC)**| Malignant | 145 | 142 | 140 | 133 | 560 | 6 |
| **Total Images** | — | **1,995** | **2,081** | **2,013** | **1,820** | **7,909** | **82** |

---

## 🏗️ Core Architecture: OMNet-V3

OMNet-V3 replaces naive feature concatenation with **Magnification-Aware Adaptive Fusion (MAF)**:

```
                           Input Patch x ∈ R^(3 × 224 × 224)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         EfficientNet-B0 (CNN)                           ViT-Tiny/16 (Transformer)
       (Localized Morphology, 1280-d)                  (Global Self-Attention, 192-d)
                  │                                               │
                  ▼                                               ▼
          Linear + LayerNorm                              Linear + LayerNorm
                  │                                               │
                  ▼                                               ▼
          f_cnn ∈ R^(256)                                 f_vit ∈ R^(256)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                        Magnification Scale Index m ∈ {0,1,2,3}
                                          │
                                          ▼
                             Scale Embedding e_m ∈ R^(64)
                                          │
                                          ▼
                       Dynamic Gate α(m) = σ(W_gate · [f_cnn, f_vit, e_m])
                                          │
                                          ▼
                  Convex Blending: f_blend = α · f_cnn + (1 - α) · f_vit
                                          │
                                          ▼
                  Residual Refinement: f_out = f_blend + MLP([f_blend, e_m])
                                          │
                                          ▼
                           Shared Latent Embedding (256-d)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
          Binary Head (2-Class)                           Subtype Head (8-Class)
         p_bin(y ∈ {Benign, Mal})                       p_sub(y ∈ {A, F, ..., PC})
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   Hierarchical Loss: L_total = 0.3 L_bin + 0.6 L_sub + 0.1 L_cons
```

### Mathematical Formulation
1. **Dynamic Gating:**
   $$\alpha = \sigma\left(\mathbf{W}_g [\mathbf{f}_{\text{cnn}} \,\|\, \mathbf{f}_{\text{vit}} \,\|\, \mathbf{e}_m] + b_g\right) \in (0, 1)$$
2. **Convex Combination & Scale Refinement:**
   $$\mathbf{f}_{\text{blend}} = \alpha \mathbf{f}_{\text{cnn}} + (1 - \alpha) \mathbf{f}_{\text{vit}}$$
   $$\mathbf{f}_{\text{out}} = \mathbf{f}_{\text{blend}} + \text{MLP}\left([\mathbf{f}_{\text{blend}} \,\|\, \mathbf{e}_m]\right) \in \mathbb{R}^{256}$$
3. **Hierarchical Consistency Regularization:**
   $$L_{\text{cons}} = D_{\text{KL}}\left(\left[ \sum_{c \in \text{Benign}} p_{\text{sub}}(c) \,,\, \sum_{c \in \text{Malignant}} p_{\text{sub}}(c) \right] \;\Bigg\|\; \mathbf{p}_{\text{bin}}\right)$$

---

## 📈 Benchmark Results Summary

All results are obtained under strictly **patient-disjoint partitions** with zero data contamination:

### Master Performance Comparison Table

| Model Family | Architecture Variant | Evaluation Protocol | Binary Accuracy | Sensitivity (Recall) | Specificity | Binary ROC-AUC | PR-AUC | Subtype Weighted-F1 | Matthews Corr. (MCC) |
|---|---|---|---|---|---|---|---|---|---|
| **OMNet-V2** | EfficientNet-B0 | BreakHis 200X Test ($N=280$) | 81.79% | 94.90% | 51.19% | 0.8712 | 0.9356 | — | 0.5391 |
| **OMNet-V2** | ViT-B/16 (Best Single) | BreakHis 200X Test ($N=280$) | **84.29%** | **97.45%** | 53.57% | **0.8998** | 0.9488 | — | **0.6105** |
| **OMNet-V2** | Soft-Voting Ensemble | BreakHis 200X Test ($N=280$) | **84.29%** | 96.43% | **55.95%** | 0.8973 | **0.9537** | — | 0.6084 |
| **OMNet-V3** | Dual-Branch MAF | Multi-Scale 5-Fold CV ($N=7,909$) | **84.56% ± 2.48%** | — | — | **>0.88 (OvR)** | — | **44.75% ± 5.89%** | **0.6175 ± 0.0532** |

### Adaptive Scale Gating Progression

The learned gating parameter $\alpha(m)$ exhibits an interpretable monotonic transition across optical zoom levels:
- **40X Zoom:** Mean $\alpha = \mathbf{0.6850 \pm 0.1301} \rightarrow$ Higher weighting on global architecture (ViT branch).
- **100X Zoom:** Mean $\alpha = \mathbf{0.5991 \pm 0.1283}$
- **200X Zoom:** Mean $\alpha = \mathbf{0.5258 \pm 0.1449}$
- **400X Zoom:** Mean $\alpha = \mathbf{0.4701 \pm 0.1459} \rightarrow$ Higher weighting on fine cellular detail (CNN branch).

---

## 🔍 Interpretability & Diagnostic Explainability

To ensure clinical trustworthiness, OMNet incorporates multiple diagnostic explainability mechanisms:
- **Grad-CAM & Score-CAM:** Identifies that convolutional feature maps activate on nuclear atypia and ductal margins rather than slide background.
- **2D t-SNE Latent Space Manifolds:** Projects 256-d bottleneck embeddings to demonstrate distinct geometric separation between benign and malignant tissue specimens.
- **Error Analysis:** Categorizes high-confidence false positives (complex fibroadenomas with high epithelial cellularity) and false negatives (well-differentiated tubular carcinomas with low nuclear pleomorphism).

---

## 💻 Installation & How to Run

### 1. Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA or ROCm GPU acceleration

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/T-RexBytes/OMNet.git
cd OMNet

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Experiments
- **OMNet-V2 (200X Transfer Learning):** Open and run `models/OMNet_BreakHis_V2.ipynb`.
- **OMNet-V3 (Multi-Scale Multi-Task Flagship):** Open and run `models/OMNet_BreakHis_V3.ipynb` (Cloud) or `models/OMNet_BreakHis_V3_local.ipynb` (Local workstation).
- **OMNet-V4 (Cleaned BreakHis Pipeline):** Open and run `models/OMNet_V4_Local(breakhis_only).ipynb`.

---

## 🛡️ Research Protocols & Anti-Leakage Guardrails

To maintain scientific integrity across all experiments:
1. **Mandatory Patient Stratification:** Patients are partitioned using `StratifiedGroupKFold` on `patient_id`. Image patches from the same patient never appear in both training and test sets.
2. **Deterministic Reproducibility:** Global random seeds (`seed=42`) and explicit split manifests (`split.csv`) ensure 100% reproducible data splits.
3. **No Metric Fabrication:** Quantitative comparisons report exact empirical values directly traceable to raw CSV and JSON output files.
4. **Clean Code & Plain Printing:** All notebooks adhere to plain text console logging (no emojis or non-standard encoding) to prevent OS rendering deadlocks.

---

## 📜 License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Citation
If you use OMNet architectures, data splitting protocols, or benchmark results in your research, please cite:
```bibtex
@misc{omnet2026breastcancer,
  title={OMNet: A Reproducible and Leakage-Safe Multi-Scale Deep Learning Framework for Breast Cancer Histopathology},
  author={OMNet Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/T-RexBytes/OMNet}}
}
```
