# OMNet — Model Architecture and Data Processing (v2)

This document describes the assumed model architecture used in the OMNet experiments, the learning objective (loss function), and the end-to-end data preparation and feeding pipeline. It is written for research purposes and is intended to be detailed and reproducible.

## Overview

- **Model name:** OMNet (research classifier for BreakHis histopathology images)
- **Task:** Patch-level classification of histopathology images (benign vs malignant or multi-class tumour subtypes)
- **Input:** RGB image patches (typical size 224×224 or 256×256) extracted from whole-slide images
- **Output:** Softmax probabilities over K classes; predicted class is argmax of probabilities

## 1. Architecture

This section describes a recommended architecture for OMNet used in the repository. The design is modular and follows common, well-tested CNN design patterns suitable for histopathology images.

1. Stem

- Input layer accepts a 3-channel RGB patch of size H×W (e.g., 224×224×3).
- Initial convolution: Conv(7×7, stride=2, filters=64) → BatchNorm → ReLU → MaxPool(3×3, stride=2).

2. Encoder (Residual blocks)

- Use a ResNet-style encoder (e.g., ResNet-34 or ResNet-50) with residual bottleneck blocks.
- Each block: conv → BN → ReLU → conv → BN (+ conv for shortcut when changing dimensions) → add → ReLU.
- Typical filter progression: 64 → 128 → 256 → 512 (with downsampling by stride=2 between stages).

3. Attention / Context module (optional but recommended)

- A lightweight channel/spatial attention module (e.g., SE block or CBAM) after the final encoder stage improves representational power for histology textures.

4. Pooling and Head

- Global average pooling over spatial dimensions produces a 1D feature vector per patch.
- Optional dropout (p=0.3–0.5) for regularization.
- Fully-connected head: Dense(512) → ReLU → BatchNorm → Dropout → Dense(K) → Softmax.

5. Output

- Softmax gives per-class probabilities p_i for classes i=1..K.

Design notes:

- Use Batch Normalization to stabilize training and support larger learning rates.
- Use small dropout in the head to reduce overfitting on small datasets.
- When using transfer learning, initialize encoder weights from ImageNet and train either the head only (linear probe) or fine-tune deeper layers with a lower learning rate.

## 2. Loss functions

The choice of loss depends on class balance, label noise, and whether you want to emphasize hard examples.

1. Categorical Cross-Entropy (standard for multi-class classification)

The cross-entropy loss for a single sample with one-hot target y (y_i ∈ {0,1}) and predicted probabilities p_i is:

$$
L_{CE} = -\\sum_{i=1}^{K} y_i \\log(p_i)
$$

When training with mini-batches, average the loss across the batch.

2. Class-weighted Cross-Entropy

If classes are imbalanced, apply class weights w_i (inverse proportional to class frequency):

$$
L_{WCE} = -\\sum_{i=1}^{K} w_i\\, y_i \\log(p_i)
$$

Weights can be computed using $w_i = \\frac{N}{K \\cdot N_i}$ where $N_i$ is examples per class and $N$ is total samples.

3. Focal Loss (to emphasize hard examples)

Focal loss down-weights easy examples and focuses learning on difficult, misclassified examples. For a single sample:

$$
L_{FL} = -\\sum_{i=1}^{K} (1 - p_i)^{\\gamma} y_i \\log(p_i)
$$

Where $\\gamma \\ge 0$ (typical values 1–3). Combine with class weights if needed.

4. Regularization losses

- L2 weight decay on model weights is applied through the optimizer (commonly 1e-4 to 1e-2).

Recommendation:

- Start with class-weighted cross-entropy; if the model is still dominated by easy negatives, switch to focal loss or hybrid schemes (weighted focal loss).

## 3. Data preprocessing

High-quality preprocessing for histopathology images has a large impact. Key steps:

1. Patch extraction

- Extract fixed-size patches (e.g., 224×224) from whole-slide images (WSIs) using a sliding window or region-of-interest sampling.
- Use a stride smaller than the patch size to cover tissue adequately or use non-overlapping patches depending on dataset size and redundancy.
- Exclude patches with low tissue content by thresholding on Otsu grayscale or simple saturation/v brightness heuristics (e.g., require > X% of pixels above a tissue mask threshold).

2. Stain normalization / color augmentation

- Histology images vary with staining protocols. Apply stain normalization (e.g., Macenko or Reinhard methods) to reduce slide-to-slide stain variance.
- During training, use color jitter (brightness, contrast, saturation, hue) conservatively to increase robustness.

3. Resize / Rescale

- Resize patches to the model input size if extracted at different resolutions.
- Normalize pixel intensities to the range expected by the backbone (commonly zero-mean, unit-variance using ImageNet mean/std or dataset-specific statistics):

$$
\\hat{x} = \\frac{x - \\mu}{\\sigma}
$$

Where $\\mu,\\sigma$ are per-channel mean and standard deviation.

4. Augmentation

- Geometric: random horizontal/vertical flips, rotations (multiples of 90° or small angles), random cropping, scaling.
- Photometric: color jitter, gamma transform, Gaussian blur (small sigma), stain augmentation.
- Advanced: MixUp, CutMix, or RandAugment can be used carefully for histology.

5. Labeling and patient-wise splitting

- When constructing train/validation/test splits ensure patient-level separation to avoid data leakage — slides from the same patient must not appear across splits.
- Use stratified splits to preserve class proportions across splits.

## 4. Data preparation and feeding pipeline

Below are recommendations for efficient, reproducible pipelines in PyTorch and TensorFlow. The core principles apply regardless of framework.

1. On-disk layout

- Store extracted patches in a structured layout: `/data/{split}/{class}/{slide_id}/{patch_id}.png` or as an HDF5/LMDB/TFRecord file for faster sequential reads.

2. Dataset and DataLoader (PyTorch)

- Implement a custom `Dataset` that yields (image_tensor, label, metadata) where metadata may include slide_id or patch coordinates.
- Use `DataLoader` with `num_workers >= 4`, `pin_memory=True`, `prefetch_factor` (PyTorch 1.7+).
- Shuffle training data each epoch. Use `DistributedSampler` for multi-GPU training.

Example sketch (PyTorch):

```python
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.files = file_list
        self.transforms = transforms
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = read_image(self.files[idx]['path'])
        if self.transforms:
            img = self.transforms(img)
        label = self.files[idx]['label']
        return img, label

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
```

3. Pipeline optimizations

- Cache precomputed transforms for validation/test sets to remove augmentation nondeterminism.
- Use image decoding libraries with SIMD support (e.g., Pillow-SIMD) or store images in a fast binary container (TFRecords, LMDB).
- Use mixed-precision training (AMP) to accelerate training and reduce memory usage.

4. Sampling strategies for imbalance

- Class-balanced sampling: oversample minority classes or use weighted random sampler.
- Hard-negative mining: periodically mine false-positive patches and add them to the training set.

5. Batch composition and aggregation

- For patch-level labels: each batch is K-way labeled patches.
- For slide-level inference: aggregate patch predictions per slide using mean, max, or trainable pooling; convert aggregated logits to slide-level probabilities.

## 5. Training recipe (recommended)

- Optimizer: AdamW with weight decay (1e-4). Alternative: SGD with momentum (0.9) and weight decay.
- Learning rate: cyclical or cosine schedule with linear warmup (e.g., initial LR 1e-3 for AdamW, warmup 1–5 epochs, then cosine annealing).
- Batch size: 16–128 depending on GPU memory; larger batches benefit from LR scaling.
- Epochs: 30–200 depending on dataset size and early stopping on validation loss.
- Gradient clipping (global norm 1.0–5.0) if training unstable.

## 6. Evaluation

- Primary metrics: accuracy, macro-F1 (for imbalanced multi-class), AUC for binary/multi-class (one-vs-rest).
- Report per-class precision/recall and confusion matrix.
- For slide-level reporting: aggregate patch predictions into slide-level scores then compute patient-level metrics.

## 7. Reproducibility and logging

- Log random seeds, dataset splits, augmentation parameters, and exact architecture/weights initialization.
- Use experiment tracking (Weights & Biases, TensorBoard) and save model checkpoints with validation score as the key.

## 8. Appendix — equations summary

Categorical cross-entropy: $L_{CE} = -\\sum_{i} y_i \\log(p_i)$

Focal loss (multi-class): $L_{FL} = -\\sum_{i} (1-p_i)^{\\gamma} y_i \\log(p_i)$

Weighted cross-entropy: $L_{WCE} = -\\sum_{i} w_i y_i \\log(p_i)$

---

If you want, I can adapt this doc to the exact architecture and hyperparameters used in your notebooks (e.g., the model in `OMNet_BreakHis_V2.ipynb`) — point me to the training cell or model definition and I will update the file accordingly.
