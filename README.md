# 🔬 OMNet

**OMNet** is a research-oriented deep learning framework for breast cancer histopathology image classification using custom CNN architectures trained from scratch on the BreakHis dataset, with a modular design for future integration of Vision Transformers, Deep Metric Learning, and Proxy Anchor Loss.

---

## Quick Start

1. Open `OMNet_BreakHis_V1.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (T4 recommended)
3. Run all cells top-to-bottom

> **Smoke Test**: The notebook ships with `CONFIG['smoke_test'] = True` (2 epochs). Set to `False` for full training.

## Project Structure

```
OMNet/
├── OMNet_BreakHis_V1.ipynb    # Complete research notebook
├── docs/
│   └── process_flow.md        # Detailed process flow & Mermaid pipeline diagrams
├── requirements.txt           # Dependencies (for local runs)
├── README.md                  # This file
└── LICENSE
```

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Environment Setup | Install packages, import libraries, set seeds |
| 2 | GPU Detection | CUDA device configuration |
| 3 | Mount Google Drive | Persistent storage setup |
| — | Configuration | Centralized hyperparameter cell |
| 4 | Dataset Loading | KaggleHub download + verification |
| 5 | Dataset Analysis | Image counts, class ratios, integrity checks |
| 6 | Dataset Visualization | Sample images, distribution charts |
| 7 | Dataset Statistics | Compute dataset-specific mean/std |
| 8 | DataLoader | BreakHisDataset class, transforms, loaders |
| 9 | Data Augmentation | Augmentation visualization |
| 10 | CNN Architecture | OMNet-V1 (4-block custom CNN) |
| 11 | Model Summary | Parameter counts, forward pass tests |
| 12 | Training Loop | Loss functions, Trainer class |
| 13 | Validation | Run training with per-epoch validation |
| 14 | Evaluation | Full test set evaluation (13 metrics) |
| 15 | Error Analysis | Misclassified samples, confidence analysis |
| 16 | Grad-CAM | Gradient-weighted class activation maps |
| 17 | ROC | Receiver Operating Characteristic curve |
| 18 | Precision-Recall | PR curve with Average Precision |
| 19 | Confusion Matrix | Absolute + normalized confusion matrices |
| 20 | Training Curves | Loss, accuracy, LR, AUC over epochs |
| — | t-SNE | Feature embedding visualization |
| 21 | Save Best Model | Checkpoint verification |
| 22 | Export Metrics | CSV + JSON export for experiment tracking |
| 23 | Final Summary | Experiment summary card |

## Architecture

**OMNet-V1** — Custom 4-block CNN trained from scratch:

```
Block 1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Drop
Block 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Drop
Block 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Drop
Block 4: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → GAP
Embedding: Linear(256→128) → ReLU → Dropout
Classifier: Linear(128→2)
```

### Future-Compatible Interface

```python
logits = model(x)                    # Standard classification
embeddings = model.extract_features(x)  # For metric learning / DML
dim = model.get_embedding_dim()      # Embedding dimensionality
```

Swap the backbone to ViT or integrate Proxy Anchor Loss without changing the training pipeline.

## Dataset

**BreakHis 400X** — Breast Cancer Histopathological Image Classification

- **Source**: `kagglehub.dataset_download("pankaj4321/breakhis-400x")`
- **Images**: ~1,693 at 400× magnification (700×460 px, RGB PNG)
- **Classes**: Benign (~32.3%) / Malignant (~67.7%)
- **Splits**: Pre-split into train / validation / test

## Metrics

The notebook computes: Accuracy, Precision, Recall, Specificity, Sensitivity, F1-Score, ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Confusion Matrix, Classification Report, and per-class metrics.

## Outputs

All results are saved to `outputs/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (by val AUC) |
| `last_model.pth` | Final epoch checkpoint |
| `training_history.csv` | Per-epoch metrics for experiment comparison |
| `metrics.csv` | Final test set metrics |
| `experiment_summary.json` | Full experiment configuration + results |
| `roc_curve.png` | ROC curve |
| `precision_recall.png` | Precision-Recall curve |
| `confusion_matrix.png` | Confusion matrix heatmaps |
| `training_curve.png` | Training dynamics (loss, acc, LR, AUC) |
| `gradcam/` | Grad-CAM heatmaps |
| `misclassified/` | Misclassified test samples |
| `correct_predictions/` | High-confidence correct predictions |

## Roadmap

- [x] OMNet-V1: Baseline custom CNN
- [ ] OMNet-V2: Multi-scale CNN with spatial attention
- [ ] Vision Transformer backbone
- [ ] Deep Metric Learning integration
- [ ] Proxy Anchor Loss training

## License

MIT
