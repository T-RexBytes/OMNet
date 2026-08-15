# OMNet: Breast Histopathology Deep Learning Research

> **A rigorous, publication-oriented deep learning framework for multi-magnification breast cancer histopathology classification using adaptive CNN–ViT feature fusion, patient-disjoint evaluation, and hierarchical pathology learning.**

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Objective & Gap](#-research-objective--gap)
3. [Version History](#-version-history)
4. [Current OMNet-V3 Architecture](#-current-omnet-v3-architecture)
5. [Dataset: BreakHis](#-dataset-breakhis)
6. [Baseline Models](#-baseline-models)
7. [Evaluation Protocol](#-evaluation-protocol)
8. [Project Structure](#-project-structure)
9. [Setup & Usage](#-setup--usage)
10. [Research Decision Log](#-research-decision-log)
11. [Future Roadmap](#-future-roadmap)
12. [Key Methodological Constraints](#-key-methodological-constraints)

---

## Project Overview

**OMNet** is a breast histopathology deep learning research project designed to:

- Develop a publication-ready model for breast cancer histopathology classification
- Address patient/specimen-level evaluation without data leakage
- Combine local morphological + global contextual feature learning
- Investigate adaptive CNN/ViT feature fusion conditioned by magnification
- Support hierarchical pathology classification (benign/malignant → 8 subtypes)
- Provide rigorous statistical and interpretability evidence
- Enable cross-dataset generalization research  

**NOT** a black-box accuracy maximization project. Every design decision is documented and justified.

---

## Research Objective & Gap

### The Core Problem

Breast histopathology image classification is challenging because:

1. **Multi-scale visual information**: Diagnostically relevant features appear at multiple magnifications (40×, 100×, 200×, 400×)
2. **Complementary representations**: Local morphological details (cell/nuclear structure) and global tissue architecture both matter
3. **Patient/specimen leakage**: Simple image-level splitting compromises generalization estimates
4. **Class imbalance & rare subtypes**: Some histopathological subtypes have very few patient examples
5. **Stain/domain variation**: H&E staining procedures produce variable image appearance

### The Gap

The literature contains CNN approaches, ViT approaches, ensemble methods, and feature fusion—but the specific combination of:

- **Lightweight CNN + lightweight ViT** (not heavy ResNets or massive ViT-Base)
- **Explicit magnification-aware adaptive fusion** (not simple concatenation or voting)
- **Hierarchical benign/malignant + subtype objectives** (not independent classifiers)
- **Rigorous patient/specimen-disjoint evaluation** (not image-level random splits)
- **Controlled stain/domain robustness analysis** (not just accuracy numbers)

...is insufficiently studied on BreakHis at 200× magnification.

### The Proposed Solution

**OMNet-V3**: EfficientNet-B0 + ViT-Tiny + Magnification-Aware Adaptive Local–Global Fusion + Hierarchical Classification

---

## Version History

### OMNet-V1 [COMPLETED]

| Aspect | Details |
|--------|---------|
| **Status** | Baseline; diagnostic results only |
| **Purpose** | Custom CNN baseline to establish reproducible training pipeline |
| **Architecture** | 4-block custom CNN (32→64→128→256 channels) trained from scratch |
| **Dataset** | BreakHis 400× magnification |
| **Tasks** | Binary classification (benign/malignant) |
| **Evaluation** | Standard train/val/test split; per-epoch metrics |
| **Key Output** | Checkpoint `best_model.pth`; baseline accuracy and loss curves |
| **Limitations** | No pretrained weights; 400× only; single task; no multi-magnification study |
| **Why Superseded** | Need for stronger baselines, pretrained models, subtype classification, and multi-magnification support |

### OMNet-V2 [COMPLETED]

| Aspect | Details |
|--------|---------|
| **Status** | Historical baseline; diagnostic results from old protocol |
| **Purpose** | Establish the expanded repository and experiment structure |
| **Architecture** | Multiple models included in baseline structure |
| **Dataset** | BreakHis (200× magnification) |
| **Tasks** | Task A (benign/malignant) + Task B (8-class subtype classification) |
| **Evaluation Protocol** | Single patient-level 70/15/15 split (later found to be flawed) |
| **Key Limitation** | **CRITICAL**: Old subtype evaluation had zero samples for phyllodes tumor in test set, making 8-class evaluation incomplete |
| **Why Superseded** | Evaluation protocol was methodologically flawed; needed K-fold CV for stable subtype metrics; no hybrid model yet |

**Important**: V1 and V2 results should **NOT** be used as final paper evidence because:
- V2's subtype test set was incomplete (phyllodes support = 0)
- Both used old splits/protocols since invalidated
- No patient-leakage verification on old splits
- Focus was diagnostic, not publication-ready

### OMNet-V3 [CURRENT / LOCKED]

| Aspect | Details |
|--------|---------|
| **Status** | Current, locked design; implementation in progress |
| **Purpose** | Publication-ready hybrid model with corrected evaluation protocol |
| **Architecture** | **EfficientNet-B0** + **ViT-Tiny** + **Magnification-Aware Adaptive Fusion** + **Hierarchical heads** |
| **Dataset** | BreakHis 200× magnification (primary), future: multi-magnification analysis |
| **Evaluation** | **Task A**: Single patient-level 70/15/15 split **Task B**: 5-fold patient-level stratified CV |
| **Key Methodological Fixes** | Patient/specimen-disjoint splits; K-fold for stability; explicit handling of rare classes; per-class metrics |
| **Expected Outputs** | `split_task_a.csv`, `folds_task_b.csv`, per-fold metrics, confusion matrices, interpretability figures, final checkpoints |

---

## Current OMNet-V3 Architecture

### Design Philosophy

Simple, interpretable, and evidence-driven. All models are loadable from standard libraries (timm, torchvision).

### Model Components (Loadable)

```python
# Model loading specifications (correct, no errors)
import timm
import torchvision.models as tv_models

# CNN Backbone - Local morphological features
cnn_backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
cnn_output_dim = 1280  # EfficientNet-B0 feature dimension

# Vision Transformer - Global contextual features
vit_backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
vit_output_dim = 192  # ViT-Tiny embedding dimension

# Optional: Alternative lightweight models (if EfficientNet not available)
# cnn_backbone = tv_models.mobilenet_v3_small(pretrained=True)
# cnn_output_dim = 576  # MobileNetV3-Small feature dimension

# Optional: ViT alternative
# vit_backbone = timm.create_model('vit_tiny_patch8_224', pretrained=True, num_classes=0)
```

### Fusion Architecture

```
Input Image (224×224 RGB)
    |
    +-------- CNN Branch --------+
    |                            |
    v                            v
  EfficientNet-B0           ViT-Tiny
   1280-d features          192-d features
    |                            |
    v                            v
  Linear Projection          Linear Projection
  1280 -> 256               192 -> 256
    |                            |
    +------------ Concat --------+
               |
               v
            512-d
               |
       Magnification Gate (Learned)
               |
               v
          256-d (gated)
               |
               v
         Fusion MLP
       256 -> 128 -> 64
               |
      +--------+--------+
      |                 |
      v                 v
  Primary Head      Subtype Head
   64 -> 2           64 -> 8
  (Benign/          (8 Subtypes:
   Malignant)       A, F, PT, TA,
                    DC, LC, MC, PC)
```

### Architecture Rationale

| Component | Why | Loading Method |
|-----------|-----|-----------------|
| **EfficientNet-B0** | Compact CNN; efficient local morphology extraction; good accuracy/compute for Colab | `timm.create_model('efficientnet_b0', pretrained=True)` |
| **ViT-Tiny** | Lightweight transformer; global context via self-attention; efficient on GPU | `timm.create_model('vit_tiny_patch16_224', pretrained=True)` |
| **256-D projection** | Standardizes dimensions; reduces overfitting; enables modular fusion | Linear layer: `nn.Linear(1280, 256)` + `nn.Linear(192, 256)` |
| **Magnification conditioning** | Learned gate weights CNN/ViT by magnification; adapts fusion to imaging scale | Embedding layer: `nn.Embedding(4, 256)` for 4 magnifications |
| **Adaptive fusion** | Learned gates (not fixed concat); allows scale-dependent representation balance | Gate: `sigmoid(mag_embedding @ weight)` element-wise multiplication |
| **Hierarchical heads** | Benign/malignant easier; subtype harder; shared fusion + multi-task loss | Two `nn.Linear` heads from fused representation |

### NOT in V3 (Intentional Constraints)

- No attention/cross-attention between CNN and ViT branches
- No gating mechanisms beyond magnification-conditioned weighting
- No additional backbones (HOG, LBP, etc.)
- No LSTM/GRU/RNN
- No massive ensemble
- No black-box architecture search  

**Why**: First, prove that simple magnification-aware fusion + hierarchical learning improves over baselines. Only if V3 results reveal a specific limitation do we add complexity.

---

## Dataset: BreakHis

### Structure

```
BreakHis/SOB/
├── benign/
│   ├── adenosis/           (A)
│   ├── fibroadenoma/       (F)
│   ├── phyllodes_tumor/    (PT)
│   └── tubular_adenoma/    (TA)
└── malignant/
    ├── ductal_carcinoma/   (DC)
    ├── lobular_carcinoma/  (LC)
    ├── mucinous_carcinoma/ (MC)
    └── papillary_carcinoma/ (PC)
```

### Key Facts

| Property | Value |
|----------|-------|
| **Total Patients** | 82 (24 benign, 58 malignant) |
| **Total Images (200×)** | ~1,700+ RGB PNG files |
| **Image Size** | Typically 700×460 pixels |
| **Image Format** | RGB PNG; H&E staining |
| **Magnifications Available** | 40×, 100×, 200×, 400× |
| **Current Scope** | 200× magnification only |
| **Access** | Private Kaggle dataset; `kagglehub` library + Kaggle API |
| **Stain Variability** | Yes; slides from different labs/protocols |

### Critical Constraint: Patient-Level Splitting

**MANDATORY**: Images from the same patient must **never** cross train/val/test partitions.

This is not a stylistic choice:
- Prevents data leakage
- Produces realistic generalization estimates
- Required for publication-quality research

Images from patient `SOB_B_A-14-22549AB` must stay together.

### Known Dataset Limitation

Some subtypes (especially phyllodes tumor, mucinous carcinoma, papillary carcinoma) have **very few patients** (≤5). A single 3-way split may leave a subtype with zero test samples. **Solution**: Use 5-fold patient-level stratified CV for Task B to ensure all folds have fair subtype representation.

---

## Baseline Models

### Baseline 1: DenseNet-201 [COMPLETED]

| Aspect | Details |
|--------|---------|
| **Model** | ImageNet-pretrained DenseNet-201 |
| **Tasks** | Binary (Task A) + 8-class subtype (Task B) |
| **Dataset Protocol** | BreakHis 200×; old split (later invalidated) |
| **Old Results** | Primary F1: 0.6975 / Subtype F1: 0.2556 (diagnostic only; not final) |
| **Architecture** | Single backbone → two classification heads |
| **Key Finding** | Severe difficulty with minority subtypes (lobular carcinoma, mucinous, papillary F1 very low; phyllodes had zero test support) |
| **Notebook** | `models/baseline/baseline_1_densenet201.ipynb` |
| **Status** | Diagnostic baseline; do NOT use old results as final paper evidence |

### Baseline 2: ViT-B/16 [COMPLETED]

| Aspect | Details |
|--------|---------|
| **Model** | Pretrained ViT-B/16 (timm library) |
| **Architecture** | Transformer backbone → two classification heads |
| **Embedding Dim** | 768-dimensional |
| **Dataset Protocol** | BreakHis 200×; old split |
| **Old Results** | Primary F1: 0.8544 / Subtype F1: 0.3070 (diagnostic; old protocol) |
| **Key Finding** | ViT substantially stronger than DenseNet for primary task; moderately better for subtype Macro-F1; still weak on minority subtypes |
| **Notebook** | `models/baseline/baseline_2_vit_b16.ipynb` |
| **Status** | Diagnostic baseline; old evaluation protocol invalidated |

### Baseline Comparison Summary

```
Task A (Benign/Malignant):
  DenseNet-201 Macro-F1: 0.6975
  ViT-B/16    Macro-F1: 0.8544  ← ViT significantly stronger
  
Task B (8-class Subtype):
  DenseNet-201 Macro-F1: 0.2556
  ViT-B/16    Macro-F1: 0.3070  ← ViT better, but both weak
  
Conclusion: ViT > DenseNet overall, but neither solves minority subtype problem.
            → Motivation for hybrid model.
```

**Critical Note**: These numbers are from the old evaluation protocol. They are diagnostic only. Baseline 3 must re-evaluate both models under the corrected protocol before publishing final comparisons.

### Baseline 3: CNN-ViT Fusion [CURRENT]

| Aspect | Details |
|--------|---------|
| **Model** | DenseNet-201 + ViT-B/16 feature-level fusion (not voting/ensemble) |
| **Architecture** | Frozen backbones → projections → fusion MLP → two heads |
| **Key Difference** | **NOT** just average predictions; **learned feature representation fusion** |
| **Fusion Strategy** | Concatenate projected embeddings → Fusion MLP → shared heads |
| **Dataset Protocol** | BreakHis 200×; **corrected split_task_a.csv + folds_task_b.csv** |
| **Phase 1** | Frozen backbones (cheap; cached embeddings) |
| **Phase 2** | Partial unfreeze (if Phase 1 promising) |
| **Preregistered Success** | Subtype Macro-F1 ≥ Better-Baseline Macro-F1 + 1.0 × Fold Std |
| **Notebook** | `models/baseline/baseline_3_fusion.ipynb` (to be implemented) |
| **Status** | Implementation in progress |

---

## Evaluation Protocol

### Task A: Primary Classification (Benign vs Malignant)

**Split**: Single patient-level stratified 70/15/15 split  
**Rationale**: Benign/malignant has sufficient patients per class; single split is stable

**Metrics**:
- Accuracy
- Precision (per-class)
- Recall (per-class)
- Macro-F1
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Confusion Matrix
- Classification Report

**Output**: `split_task_a.csv` (patient IDs and assigned fold)

### Task B: Subtype Classification (8-class)

**Split**: 5-fold patient-level stratified cross-validation  
**Rationale**: Some subtypes have very few patients; K-fold ensures all folds contain all subtypes (or explicitly flag if insufficient)

**Metrics Per Fold**:
- Accuracy
- Macro Precision / Recall / F1
- Per-class Precision / Recall / F1
- Balanced Accuracy
- MCC
- Confusion Matrix

**Aggregation**:
- Mean ± standard deviation (across 5 folds)
- Per-class fold stability analysis

**Special Handling**:
- If a subtype has zero or only one patient in a held-out fold:
  - **DO NOT** silently report 0.0
  - **Explicitly flag**: "Insufficient patient count for this fold; support = N"
  - Report how many of 5 folds actually contained that subtype
  - This is a methodological limitation, not a model failure

**Output**: `folds_task_b.csv` (patient IDs and fold assignments)

### Cross-Validation Details

- **Stratification**: Patient-level; all patients from a subtype spread across folds proportionally
- **Patient Leakage Prevention**: Verify no patient appears in both training and test sets
- **Per-Fold Seed**: Use a fixed random seed for reproducibility
- **Best Checkpoint Selection**: Save best checkpoint per fold (by validation metric)

---

## Project Structure

```
OMNet/
├── README.md                              # This file
├── LICENSE
├── requirements.txt                       # Python dependencies
│
├── docs/
│   ├── OMNet-V3_Final_Paper_Master_Checklist.md    # Research evidence checklist
│   ├── process_flow_v2.md
│   ├── process_flow_v3.md
│   ├── message.md
│   └── message.txt
│
├── models/
│   ├── OMNet_BreakHis_V1.ipynb            # V1: Custom CNN baseline
│   ├── OMNet_BreakHis_V3.ipynb            # V3: Hybrid CNN-ViT (latest)
│   ├── OMNet_BreakHis_V3 (1).ipynb        # V3 variant/checkpoint
│   │
│   └── baseline/
│       ├── baseline_1_densenet201.ipynb   # DenseNet baseline
│       ├── baseline_2_vit_b16.ipynb       # ViT baseline
│       ├── baseline_3_fusion.ipynb        # CNN-ViT fusion (planned)
│       │
│       └── plan/
│           ├── baseline_1_densenet201_plan.md
│           ├── baseline_2_vit_b16_plan.md
│           └── baseline_3_fusion_plan.md
│
├── pdf/                                   # Research papers & references
│   ├── 1-s2.0-S1746809426009857-main.pdf
│   ├── 21-s2.0-S1746809425018920-main.pdf
│   ├── 31-s2.0-S1746809425008092-main.pdf
│   └── s00521-025-10984-2.pdf
│
└── notebook_guidelines.md                 # Notebook best practices
```

### Key Directories (Not in Repo, Stored on Google Drive)

```
/content/drive/MyDrive/BreakHis/           # Dataset (private Kaggle)
/content/drive/MyDrive/output_v3/          # Experiment outputs
    ├── split_task_a.csv
    ├── folds_task_b.csv
    ├── fold_0_metrics.json
    ├── fold_1_metrics.json
    ├── ...
    ├── checkpoints/
    │   ├── best_fold_0.pth
    │   ├── best_fold_1.pth
    │   └── ...
    └── figures/
        ├── confusion_matrices/
        ├── fold_curves/
        └── interpretability/
```

---

## Setup & Usage

### Prerequisites

- **Python** 3.8+
- **GPU** (strongly recommended): NVIDIA T4 or better
- **PyTorch** 2.0+
- **Google Colab** (primary development environment)

### Dependencies (from requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0          # For EfficientNet, ViT models
kagglehub>=0.3.0
grad-cam>=1.5.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Model Initialization (No Errors - Tested)

This code reliably loads the V3 models:

```python
import torch
import torch.nn as nn
import timm
from torch import Tensor

# Initialize backbones (will auto-download pretrained weights)
def load_backbone_models():
    """Load CNN and ViT backbones with correct output dimensions"""
    
    # CNN backbone - returns features only (no classification head)
    cnn_model = timm.create_model('efficientnet_b0', 
                                   pretrained=True, 
                                   num_classes=0)  # Remove classification head
    cnn_feature_dim = cnn_model.num_features  # Returns 1280 for EfficientNet-B0
    
    # ViT backbone - returns embeddings only
    vit_model = timm.create_model('vit_tiny_patch16_224',
                                   pretrained=True,
                                   num_classes=0)  # Remove classification head
    vit_feature_dim = vit_model.embed_dim  # Returns 192 for ViT-Tiny
    
    return cnn_model, cnn_feature_dim, vit_model, vit_feature_dim

# Fusion model class (example)
class OMNetV3(nn.Module):
    """OMNet-V3: EfficientNet-B0 + ViT-Tiny with Magnification-Aware Fusion"""
    
    def __init__(self, projection_dim=256, num_magnifications=4):
        super().__init__()
        
        # Load backbones
        self.cnn, self.cnn_dim, self.vit, self.vit_dim = load_backbone_models()
        
        # Freeze backbones for Phase 1
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Projections
        self.cnn_projection = nn.Sequential(
            nn.Linear(self.cnn_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.vit_projection = nn.Sequential(
            nn.Linear(self.vit_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Magnification embedding for adaptive fusion
        self.mag_embedding = nn.Embedding(num_magnifications, projection_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(projection_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Classification heads
        self.primary_head = nn.Linear(128, 2)  # Benign/Malignant
        self.subtype_head = nn.Linear(128, 8)  # 8 subtypes
    
    def forward(self, x: Tensor, mag_idx: Tensor):
        """
        Args:
            x: Input image tensor (B, 3, 224, 224)
            mag_idx: Magnification indices (B,) with values in [0, 1, 2, 3]
        
        Returns:
            primary_logits: (B, 2)
            subtype_logits: (B, 8)
        """
        # Extract features
        cnn_features = self.cnn(x)  # (B, 1280)
        vit_features = self.vit(x)  # (B, 192)
        
        # Project to common dimension
        cnn_proj = self.cnn_projection(cnn_features)  # (B, 256)
        vit_proj = self.vit_projection(vit_features)  # (B, 256)
        
        # Magnification-aware gating
        mag_gate = torch.sigmoid(self.mag_embedding(mag_idx))  # (B, 256)
        
        # Adaptive fusion with magnification weighting
        fused = torch.cat([cnn_proj * mag_gate, vit_proj * (1 - mag_gate)], dim=1)  # (B, 512)
        
        # Fusion MLP
        fused_features = self.fusion_mlp(fused)  # (B, 128)
        
        # Classification
        primary_logits = self.primary_head(fused_features)  # (B, 2)
        subtype_logits = self.subtype_head(fused_features)  # (B, 8)
        
        return primary_logits, subtype_logits

# Test initialization (run this to verify no errors)
if __name__ == '__main__':
    print("Testing model initialization...")
    model = OMNetV3(projection_dim=256, num_magnifications=4)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy input
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    mag_idx = torch.randint(0, 4, (4,))  # Random magnifications
    
    with torch.no_grad():
        primary, subtype = model(x, mag_idx)
    
    print(f"Model initialized successfully!")
    print(f"Primary head output shape: {primary.shape}")  # Should be (4, 2)
    print(f"Subtype head output shape: {subtype.shape}")  # Should be (4, 8)
```

### Local Setup

```bash
# Clone repository
git clone <repo_url>
cd OMNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify model loading (optional, test in Python)
python -c "
import timm
import torchvision.models as tv_models
cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
print('Models loaded successfully!')
print(f'CNN output dim: {cnn.num_features}')
print(f'ViT output dim: {vit.embed_dim}')
"
```

### Magnification Mapping

BreakHis images are available at 4 magnification levels. V3 uses an embedding layer to condition fusion:

```python
# Magnification to index mapping (used in data loading)
MAGNIFICATION_MAPPING = {
    '40x': 0,
    '100x': 1,
    '200x': 2,
    '400x': 3,
}

# Extract magnification from filename
# Example filename: SOB_B_A-14-22549AB-40-001.png
def extract_magnification_from_filename(filename: str) -> int:
    """Extract magnification index from BreakHis filename"""
    # Filename format: SOB_[B|M]_[SUBTYPE]-[PATIENT]-[MAG]-[IMAGE].png
    parts = filename.split('-')
    mag_str = parts[-2]  # '-40-' or '-100-' etc
    mag_idx = MAGNIFICATION_MAPPING.get(f'{mag_str}x', 2)  # Default to 200x
    return mag_idx
```

### Data Loading Example

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class BreakHisDataset(Dataset):
    """BreakHis dataset with magnification conditioning"""
    
    def __init__(self, image_paths, labels, primary_labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Subtype class labels (0-7)
            primary_labels: Binary labels (0-1 for benign/malignant)
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.primary_labels = primary_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Extract magnification from filename
        filename = os.path.basename(img_path)
        mag_idx = extract_magnification_from_filename(filename)
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'magnification_idx': torch.tensor(mag_idx, dtype=torch.long),
            'subtype_label': torch.tensor(self.labels[idx], dtype=torch.long),
            'primary_label': torch.tensor(self.primary_labels[idx], dtype=torch.long),
            'filename': filename,
        }

# Usage example
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create dataset and loader
train_dataset = BreakHisDataset(train_paths, train_labels, train_primary, train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
```

### Running in Google Colab

1. Open notebook in Colab:
   - https://colab.research.google.com
   - Upload `models/baseline/baseline_3_fusion.ipynb`
   
2. Set runtime to **GPU** (T4 or better recommended)

3. Configure Kaggle API:
   - Upload `kaggle.json` from your Kaggle account settings
   - Or use Colab Secrets: add `KAGGLE_USERNAME` and `KAGGLE_KEY`

4. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. Run cells top-to-bottom

6. Model and results are saved to `/content/drive/MyDrive/output_v3/`

---

## Research Decision Log

### Decision 1: Why CNN + ViT?

**Question**: Why combine CNN and ViT instead of using a single large model?

**Answer**:
- CNNs excel at local morphological patterns (cell/nuclear structure)
- ViTs capture global tissue architecture via self-attention
- Complementary representations are likely to improve over either alone
- Lightweight models (EfficientNet-B0, ViT-Tiny) fit within Colab GPU constraints
- Feature fusion is more interpretable than black-box ensemble voting

**Decision**: Proceed with CNN-ViT fusion hypothesis.

### Decision 2: Why EfficientNet-B0 + ViT-Tiny for V3?

**Question**: Why not use DenseNet-201 and ViT-B/16 (from Baselines 1 & 2)?

**Answer**:
- Baseline 1/2 were diagnostic with old evaluation protocol
- Need to re-evaluate baselines under corrected protocol first
- EfficientNet-B0 and ViT-Tiny are lighter, better for Colab, and proven effective
- Smaller models reduce overfitting on limited patient data
- Easier interpretability; cleaner ablations

**Decision**: V3 uses EfficientNet-B0 + ViT-Tiny as primary models. Baseline 1/2 will be re-evaluated under corrected protocol for final comparison.

### Decision 3: Why Magnification Conditioning?

**Question**: Should magnification be a model input or handled during preprocessing?

**Answer**:
- BreakHis images are at 40×, 100×, 200×, 400×
- Different magnifications reveal different diagnostic features
- Magnification is known metadata; not using it is wasteful
- Learned magnification embedding allows scale-aware fusion
- Enables future multi-magnification training without explicit dataset mixing

**Decision**: Magnification-aware adaptive fusion is central to V3.

### Decision 4: Patient-Level Evaluation & K-Fold for Subtype

**Question**: Why use 5-fold CV for subtype classification instead of a single split?

**Answer**:
- BreakHis has only 82 patients total
- Some subtypes have <5 patients (e.g., phyllodes tumor)
- Single 70/15/15 split can leave a subtype with zero test support (discovered in Baseline 1/2)
- 5-fold CV is more stable for rare classes
- Explicit handling of insufficient-support folds preserves methodological integrity

**Decision**: Task A uses single split; Task B uses 5-fold CV. This asymmetry is intentional and justified.

### Decision 5: Why NOT Extensive Architecture Search?

**Question**: Should we try ResNet, DenseNet, Swin, multiple ViT variants, etc.?

**Answer**:
- Limited Colab GPU budget and researcher time
- Premature architecture search before core fusion is validated is poor research practice
- One well-designed model + controlled ablations > 10 models + no systematic comparison
- If V3 results reveal a specific limitation, then propose targeted improvement
- This follows "correct evaluation → strong baselines → simple hybrid → evidence → targeted final model"

**Decision**: V3 locks EfficientNet-B0 + ViT-Tiny. Future alternatives only if evidence justifies it.

### Decision 6: Frozen Backbones in Phase 1

**Question**: Should backbones be trainable from the start?

**Answer**:
- Frozen backbones → cached embeddings → fast Phase 1 experiments
- Cheaper to validate fusion architecture quickly
- Feature space is fixed; focus on fusion layer quality
- Phase 2 (if promising) unfreezes carefully with discriminative learning rates
- Avoids brute-force fine-tuning when the core hypothesis hasn't been tested

**Decision**: Phase 1 uses frozen backbones. Phase 2 (if justified) uses partial unfreezing.

---

## Future Roadmap

### Phase 1: V3 Pipeline Verification & Fold-0 Validation [IN PROGRESS]

**Objective**: Ensure evaluation protocol is correct; validate on Fold 0.

**Prerequisite**: 
- Patient/subtype distribution audited
- `split_task_a.csv` and `folds_task_b.csv` generated
- Patient leakage verified
- Data loaders tested

**Work**:
1. Load pretrained EfficientNet-B0 + ViT-Tiny backbones
2. Extract and cache frozen features on Fold 0 training set
3. Train fusion MLP + heads on Fold 0 training
4. Validate on Fold 0 validation set
5. Test on Fold 0 test set
6. Verify metrics calculation
7. Verify rare-class handling (flag insufficient support where applicable)

**Expected Output**:
- Fold 0 Task A metrics (Accuracy, F1, AUC, etc.)
- Fold 0 Task B per-class metrics
- Fold 0 confusion matrices
- Validation that protocol is leakage-free

**Why It Matters**: Catches evaluation bugs before investing in 5 folds.

### Phase 2: Complete V3 Patient-Level Evaluation [PLANNED]

**Objective**: Run all 5 folds; compute mean ± std.

**Prerequisite**: Phase 1 is validated and bug-free.

**Work**:
1. Run Folds 1–4 using identical protocol as Fold 0
2. Extract per-fold metrics
3. Aggregate: mean, std, min, max across folds
4. Verify fold stability
5. Explicitly report rare-class support per fold
6. Generate aggregated confusion matrices

**Expected Output**:
- Task A: Mean ± std metrics
- Task B: Mean ± std metrics per subtype
- Per-fold breakdown
- Confidence intervals (if time permits)
- Rare-class limitations documented

**Why It Matters**: Produces stable, publishable baseline numbers.

### Phase 3: Final V3 Analysis, Interpretability & Paper Evidence [PLANNED]

**Objective**: Generate figures, tables, and interpretability analyses for publication.

**Prerequisite**: Phase 2 complete and validated.

**Work**:
1. Generate publication-quality confusion matrices (per fold + aggregated)
2. Compute per-subtype F1, precision, recall
3. Analyze which subtypes improved vs. DenseNet / ViT baselines
4. Grad-CAM or attention visualizations (CNN and ViT separately, then fusion)
5. Error analysis: which samples did fusion help? Which remain hard?
6. Magnification analysis: does adaptive fusion behave differently at 200× vs. other scales (if multi-mag data available)?
7. Compute confidence intervals and statistical significance tests (if applicable)
8. Ablation: frozen vs. partial unfreeze performance comparison

**Expected Output**:
- Figures: confusion matrices, per-class metrics bar charts, learning curves, attention maps
- Tables: Task A/B metrics with uncertainty estimates
- Error analysis document
- Interpretability report
- Code and random seed for reproducibility

**Why It Matters**: Produces manuscript-ready evidence.

### Phase 4: Re-evaluate Baseline 1 & 2 Under Corrected Protocol [PLANNED]

**Objective**: Generate final comparison numbers for DenseNet-201 and ViT-B/16 under new protocol.

**Prerequisite**: Phase 2 complete (so we have the corrected splits).

**Work**:
1. Re-train/evaluate DenseNet-201 on `split_task_a.csv` and `folds_task_b.csv`
2. Re-train/evaluate ViT-B/16 on same splits
3. Compute final metrics using corrected protocol
4. Compare against V3 results
5. Verify that V3 improvement exceeds preregistered threshold

**Expected Output**:
- Final DenseNet-201 metrics (Task A, Task B)
- Final ViT-B/16 metrics (Task A, Task B)
- Final V3 metrics
- Comparison table: Baseline 1 vs. Baseline 2 vs. V3
- Evidence for/against preregistered success criterion

**Why It Matters**: Ensures fair comparison; invalidates old numbers; supports final paper claims.

### Phase 5: External Cross-Dataset Evaluation [FUTURE / OPTIONAL]

**Objective**: Test V3 generalization on a different breast histopathology dataset.

**Prerequisite**: 
- Phase 3 complete (V3 is finalized)
- Compatible external dataset identified and licensed
- Candidate datasets: e.g., invasive cancer detection dataset, other H&E breast pathology sources (if available)

**Work**:
1. Identify compatible external dataset
2. Verify licensing and accessibility
3. Preprocess using same protocol as BreakHis
4. Load V3 checkpoint trained on BreakHis
5. Evaluate zero-shot or fine-tuned performance
6. Analyze domain shift artifacts
7. Propose stain normalization or domain adaptation if needed

**Expected Output**:
- Cross-dataset performance metrics
- Domain shift analysis
- Generalization limitations documented

**Why It Matters**: Supports claim of generalizable approach; identifies domain robustness gaps.

### Phase 6: Potential Architectural Improvements (Evidence-Driven) [FUTURE / OPTIONAL]

**Objective**: Only if V3 Phase 2-3 reveal a specific, well-documented limitation.

**Examples**:

**If minority subtypes remain weak despite fusion**:
- Investigate class-balanced loss (focal loss, weighted cross-entropy)
- Balanced sampling strategies
- Subtype-aware regularization

**If CNN and ViT errors are not complementary**:
- Analyze error correlation
- Experiment with gated/attention fusion (not forced; only if evidence motivates)
- Cross-branch regularization

**If generalization gap is large**:
- Stain normalization preprocessing
- Domain adaptation techniques
- Multi-magnification training

**Constraint**: Do NOT combine multiple improvements simultaneously. Each major change gets a dedicated, controlled experiment with ablation.

**Why It Matters**: Ensures final model is driven by evidence, not premature optimization.

---

## Key Methodological Constraints

### DO's

- **Patient-level splitting**: All images from a patient stay together
- **5-fold CV for subtype**: Use stratified K-fold for rare-class stability
- **Verify leakage**: Explicitly check no patient crosses train/test
- **Document rare-class handling**: Flag if a subtype has zero support in a fold
- **Save splits**: Preserve `split_task_a.csv` and `folds_task_b.csv` for reproducibility
- **Hierarchical loss**: Combine Task A + Task B objectives
- **Per-class metrics**: Report F1, precision, recall for every subtype
- **Reproducibility**: Use fixed random seeds; save configs; document hyperparameters
- **Ablation on major changes**: If Phase 2 results motivate an improvement, test it systematically

### DON'Ts

- **Image-level splitting**: Never use random shuffled image-level splits
- **Patient leakage**: Do not let the same patient appear in train and test
- **Silently hide rare-class zeros**: If a subtype has zero support in a fold, explicitly flag it
- **Reuse old V1/V2 splits**: Those are invalidated; start fresh with corrected protocol
- **Combine multiple architectural changes at once**: Test one idea per experiment
- **Claim novelty without literature review**: Verify what already exists in the literature
- **Report results without reproducibility artifacts**: Save seeds, configs, checkpoints, splits
- **Ignore minority class failure**: If some subtypes remain weak, analyze why; don't hide it in aggregate F1
- **Use unvalidated checkpoints**: Always verify Fold 0 before running all 5 folds

---

## References & Related Work

Papers are stored in `pdf/`:
- Recent breast histopathology classification methods
- CNN and ViT baseline techniques
- Multi-magnification histopathology approaches
- Patient-level evaluation methodologies

See `docs/OMNet-V3_Final_Paper_Master_Checklist.md` for detailed literature matrix and gap analysis.

---

## License

See [LICENSE](LICENSE) file.

---

## Project Context & Reproducibility

For full research context, see:
- `docs/OMNet-V3_Final_Paper_Master_Checklist.md` — Evidence checklist and requirements
- `docs/process_flow_v3.md` — Pipeline diagrams and data flow
- `models/baseline/plan/*.md` — Detailed baseline specifications
- `notebook_guidelines.md` — Notebook best practices

**Golden Rule**: This project prioritizes:

1. **Correctness** over speed
2. **Reproducibility** over convenience
3. **Evidence** over claims
4. **Simplicity** over premature complexity

Every major decision is documented in this README and traced to experimental evidence or research necessity. Future contributors should:
- Read this README completely
- Consult the evidence checklist before making design changes
- Preserve all historical experiment records
- Extend the research roadmap, not replace it
- Document decisions in the Research Decision Log

---

**Last Updated**: 2025-08-15  
**Status**: V3 Locked; Phase 1 In Progress  
**Maintained By**: OMNet Research Team
