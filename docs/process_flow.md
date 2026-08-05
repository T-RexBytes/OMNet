# OMNet Process Flow & Pipeline Documentation

This document provides a comprehensive, simplified, yet production-accurate process flow for the **OMNet** research framework on breast cancer histopathology image classification (BreakHis 400X).

---

## High-Level System Architecture

The following diagram illustrates the complete end-to-end pipeline from environment initialization to metric export and visual interpretability.

```mermaid
flowchart TD
    subgraph Env["1. Setup & Config"]
        A[Environment Init & Seed] --> B[GPU Detection]
        B --> C[Google Drive Mount]
        C --> D[Centralized Config]
    end

    subgraph Data["2. Data Pipeline"]
        D --> E[KaggleHub Download]
        E --> F[Directory & Split Audit]
        F --> G[Dataset-Wide Statistics\nCompute RGB Mean & Std]
        G --> H[Train Transforms\nRandom Crop/Flip/Jitter]
        G --> I[Eval Transforms\nResize 256 -> Crop 224]
        H --> J[PyTorch DataLoaders]
        I --> J
    end

    subgraph Model["3. OMNet-V1 Architecture"]
        J --> K[Input Image Tensor\n3 x 224 x 224]
        K --> L[4 Conv Blocks\n32->64->128->256]
        L --> M[Global Avg Pooling]
        M --> N[Embedding Layer\n128-dim]
        N --> O[Classifier Head\n2-class Logits]
    end

    subgraph Train["4. Training & Validation Engine"]
        O --> P[Loss Function\nWeighted CE / Focal Loss]
        P --> Q[Adam Optimizer + Cosine LR]
        Q --> R[Mixed Precision FP16 AMP]
        R --> S{Val ROC-AUC Improved?}
        S -- Yes --> T[Save best_model.pth]
        S -- No --> U[Increment Patience]
    end

    subgraph Eval["5. Evaluation & Interpretability"]
        T --> V[Test Set Inference]
        V --> W[Compute 13 Metrics\nAccuracy, F1, AUC, MCC...]
        V --> X[Grad-CAM Heatmaps]
        V --> Y[t-SNE Embeddings]
        V --> Z[Export Assets\nCSV, JSON, PNGs]
    end
```

---

## 1. Setup & Centralized Configuration

### Process Flow
1. **Dependency Installation**: Install dynamic dependencies (`kagglehub`, `pytorch-grad-cam`).
2. **Deterministic Seed Allocation**: Seed `random`, `numpy`, `torch`, and CUDA backends (`cudnn.deterministic = True`).
3. **Hardware Assessment**: Queries `torch.cuda.get_device_properties(0)` for compute capability and memory capacity.
4. **Drive Mount & Workspace Pathing**: Mounts Google Drive (`/content/drive/MyDrive/OMNet/outputs`) with an automatic local fallback if non-Colab.
5. **Centralized Configuration (`CONFIG`)**: Central dictionary defining batch size, learning rates, loss functions, normalization mode, seed, and `smoke_test` toggle.

---

## 2. Dataset Ingestion & Validation

The dataset used is **BreakHis 400X** (`pankaj4321/breakhis-400x`), containing 1,693 histopathology microscopy images at 400× magnification.

```mermaid
flowchart LR
    A[KaggleHub Download] --> B[Locate Split Root]
    B --> C[Verify Folders:\ntrain/, validation/, test/]
    C --> D[Audit Subfolders:\nbenign/, malignant/]
    D --> E[Check File Integrity & Resolution]
    E --> F[Class Balance Analysis:\nBenign ~32.3%, Malignant ~67.7%]
```

### Key Verification Rules:
- **Zero Data Leakage / No Re-splitting**: The pre-split directory structure (`train/`, `validation/`, `test/`) is strictly preserved.
- **Corrupt File Interception**: Each image is opened with PIL to verify integrity prior to DataLoader creation.

---

## 3. Dataset-Specific Normalization Computation

Rather than using generic ImageNet statistics ($[0.485, 0.456, 0.406]$), OMNet computes the exact RGB mean and standard deviation across all training images to match the staining characteristics of histopathology images.

```mermaid
sequenceDiagram
    autonumber
    participant Loop as DataLoader Loop
    participant Image as Training Image
    participant Stat as Running Accumulator
    participant Output as Normalization Stats

    Loop->>Image: Read RGB Image (PIL)
    Image->>Stat: Convert to Tensor in [0, 1]
    Stat->>Stat: Accumulate Channel Pixel Sum
    Stat->>Stat: Accumulate Channel Pixel Squared Sum
    Stat->>Stat: Accumulate Total Pixel Count
    Loop->>Output: Calculate Final Mean & Std
```

$$\text{Mean}_c = \frac{\sum P_c}{N_{pixels}}, \quad \text{Std}_c = \sqrt{\frac{\sum P_c^2}{N_{pixels}} - (\text{Mean}_c)^2}$$

---

## 4. Data Processing & Augmentation Pipeline

```mermaid
flowchart TD
    Raw[Raw Image: 700x460 RGB] --> Split{Split Type?}
    
    Split -- Train Split --> Crop[RandomResizedCrop 224x224]
    Crop --> FlipH[Random Horizontal Flip p=0.5]
    FlipH --> FlipV[Random Vertical Flip p=0.5]
    FlipV --> Rot[Random Rotation 15°]
    Rot --> Jitter[ColorJitter brightness/contrast/hue]
    Jitter --> Norm1[Normalize dataset mean/std]
    Norm1 --> TensorTrain[Train Batch Tensor: B x 3 x 224 x 224]

    Split -- Val/Test Split --> Resize[Resize to 256x256]
    Resize --> CCrop[CenterCrop 224x224]
    CCrop --> Norm2[Normalize dataset mean/std]
    Norm2 --> TensorEval[Eval Batch Tensor: B x 3 x 224 x 224]
```

---

## 5. OMNet-V1 Architecture & Modular Interfaces

OMNet-V1 is a custom 4-block CNN built from scratch. It decouples feature extraction from classification to ensure seamless future migration to **Vision Transformers**, **Deep Metric Learning**, and **Proxy Anchor Loss**.

```mermaid
graph TD
    Input[Input Tensor: B x 3 x 224 x 224] --> B1[ConvBlock 1: 3 -> 32\nConv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool -> Drop(0.25)]
    B1 --> B2[ConvBlock 2: 32 -> 64\nConv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool -> Drop(0.25)]
    B2 --> B3[ConvBlock 3: 64 -> 128\nConv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool -> Drop(0.25)]
    B3 --> B4[Final Conv Block: 128 -> 256\nConv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU]
    B4 --> GAP[Global Average Pooling\nAdaptiveAvgPool2d(1)]
    GAP --> Flatten[Flatten -> B x 256]
    
    subgraph ModularInterface["Modular DML & Backbone Interface"]
        Flatten --> Embed[Linear(256 -> 128) + ReLU + Dropout(0.5)]
        Embed --> FeatureOut["extract_features(x)\nReturns B x 128 Embeddings"]
    end
    
    FeatureOut --> Head[Classifier: Linear(128 -> 2)]
    Head --> LogitsOut["forward(x)\nReturns B x 2 Class Logits"]
```

---

## 6. Training & Validation Execution Loop

The training engine (`Trainer`) integrates Automatic Mixed Precision (AMP), inverse-frequency class weighting, cosine learning rate decay, and validation-AUC early stopping.

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Trainer Engine
    participant Model as OMNet-V1 Model
    participant Loss as Criterion (CE / Focal)
    participant AMP as GradScaler (FP16)
    participant Opt as Adam Optimizer
    participant Sched as Cosine Scheduler

    loop Every Epoch
        Engine->>Model: Set train() mode
        loop Every Batch
            Engine->>AMP: autocast(device='cuda')
            AMP->>Model: Forward pass -> Logits
            AMP->>Loss: Compute Weighted Loss
            Engine->>AMP: scale(loss).backward()
            Engine->>AMP: step(optimizer) & update()
        end
        
        Engine->>Model: Set eval() mode
        loop Validation Batch
            Engine->>Model: Forward pass -> Logits
            Engine->>Loss: Calculate Val Loss & Metrics
        end
        
        Engine->>Sched: Step LR Scheduler
        Engine->>Engine: Compare Val ROC-AUC with Best
        alt ROC-AUC Improved
            Engine->>Engine: Save best_model.pth & reset patience
        else ROC-AUC Not Improved
            Engine->>Engine: Increment patience counter
        end
    end
```

---

## 7. Comprehensive Evaluation Metrics Suite

OMNet evaluates test performance across 13 holistic metrics:

```mermaid
mindmap
  root((OMNet Metrics Suite))
    Primary Classification
      Accuracy
      Balanced Accuracy
    Sensitivity & Specificity
      Recall / Sensitivity (TP / TP + FN)
      Specificity (TN / TN + FP)
      Precision (TP / TP + FP)
    Composite & Imbalance Resilient
      F1-Score
      Matthews Correlation Coefficient (MCC)
    Probability & Threshold Curves
      ROC-AUC (Receiver Operating Characteristic)
      PR-AUC (Precision-Recall Average Precision)
    Detailed Diagnostics
      Confusion Matrix (Counts & Normalized)
      Per-Class Classification Report
```

---

## 8. Interpretability & Error Analysis Pipeline

To prevent "black-box" predictions, OMNet generates three visual diagnostic artifacts:

```mermaid
flowchart TD
    ModelEval[Trained OMNet-V1 Model] --> TestInfer[Test Set Inference]
    
    TestInfer --> GradCAM["1. Grad-CAM Activation Maps\nTargets: model.final_conv[3]\nGenerates heatmaps over pathologically critical regions"]
    
    TestInfer --> ErrorAnalysis["2. Error & Confidence Analysis\nSeparates Correct vs Misclassified\nPlots confidence histograms & saves false negative/positive grids"]
    
    TestInfer --> TSNE["3. t-SNE Embedding Projection\nCalls extract_features(x)\nProjects 128-dim vectors to 2D scatter plot colored by class"]
```

---

## 9. Output Artifact Structure

Every experiment run automatically exports structured assets into the `outputs/` directory:

```
outputs/
├── best_model.pth           # Full checkpoint of peak validation AUC epoch
├── last_model.pth           # Final epoch checkpoint
├── training_history.csv     # Epoch-by-epoch loss, metrics, LR, and time
├── metrics.csv              # Final test set evaluation metrics
├── experiment_summary.json  # Complete environment, config, and metric dump
├── confusion_matrix.png     # Counts and normalized confusion heatmaps
├── roc_curve.png            # ROC curve with AUC annotation
├── precision_recall.png     # Precision-Recall curve with AP annotation
├── training_curve.png       # 2x2 grid (Loss, Acc, LR, AUC vs Epochs)
├── tsne_embeddings.png      # 2D t-SNE plot of learned feature space
├── gradcam/                 # Grad-CAM overlay images
├── misclassified/           # Misclassified test samples with confidence
└── correct_predictions/     # Top high-confidence correct predictions
```

---

## 10. Future Extensibility Architecture

```mermaid
flowchart LR
    subgraph Current["OMNet-V1"]
        A[OMNet-V1 Custom CNN] --> B[extract_features]
        B --> C[CrossEntropy / Focal Loss]
    end

    subgraph FutureExt["Future Modular Extensions"]
        B -. Replace Backbone .-> D[Vision Transformer / ViT]
        B -. Replace Loss .-> E[Proxy Anchor Loss]
        B -. Training Strategy .-> F[Deep Metric Learning / DML]
    end
```

By leveraging `extract_features()` and `get_embedding_dim()`, future iterations (OMNet-V2, ViT, Proxy Anchor Loss) plug directly into the existing data loader, trainer, and evaluation suite without modifying the surrounding infrastructure.
