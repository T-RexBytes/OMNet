# OMNet Model-Building Specification — AMD RX 9060 XT 16 GB / Linux / ROCm

## Purpose

This document defines the hardware-specific constraints that must be followed when building or modifying an OMNet model intended to run on the friend's machine.

The target environment is:

- GPU: AMD Radeon RX 9060 XT 16 GB
- GPU architecture: RDNA 4
- LLVM/HIP target: `gfx1200`
- OS: Linux
- GPU software stack: ROCm + HIP
- PyTorch device string: `cuda:0` under ROCm-compatible PyTorch builds
- Usable VRAM reported by the existing notebook: approximately 15.92 GiB

AMD's current ROCm documentation officially lists the RX 9060 XT as supported on Linux and identifies it as `gfx1200`. ROCm's Radeon compatibility matrix lists PyTorch 2.9.1 with ROCm 7.2 as an official production-supported combination for Radeon GPUs. The exact installed stack should still be verified on the machine before a full experiment.

## 1. Most important rule: design for ROCm, not CUDA

The model must be written against portable PyTorch APIs wherever possible. Do not assume NVIDIA-specific CUDA kernels, CUDA-only libraries, or CUDA-only optimization paths.

Important distinction:

- ROCm PyTorch intentionally exposes the CUDA-style PyTorch device API (`torch.cuda`) for compatibility.
- Therefore `torch.cuda.is_available()`, `torch.device("cuda:0")`, `torch.cuda.empty_cache()`, and related APIs can still be valid on AMD under ROCm.
- The backend is nevertheless HIP/ROCm, not NVIDIA CUDA.

The existing OMNet notebook already follows this pattern: it reports ROCm/HIP, selects `torch.device("cuda:0")`, and reads GPU properties through `torch.cuda`.

## 2. Target hardware characteristics that affect model design

| Property | Target |
|---|---|
| GPU | AMD Radeon RX 9060 XT |
| VRAM | 16 GB nominal; ~15.92 GiB reported by current notebook |
| Architecture | RDNA 4 |
| LLVM target | `gfx1200` |
| ROCm backend | HIP / ROCm |
| Primary deep-learning precision | FP32 for stable full training |
| BF16 | Supported by the installed stack for diagnostic tests, but **not approved for repeated full training on the current environment** |
| Image size in OMNet-V3 | 224 × 224 |
| Current batch size | 16 |
| Current model | EfficientNet-B0 + ViT-Tiny/16 + MAF fusion |

The RX 9060 XT has 16 GiB VRAM and 32 compute units according to AMD's GPU specification table. For OMNet, VRAM capacity is more important than raw GPU specification because the model performs two pretrained backbone forward/backward passes for each batch.

## 3. OMNet architecture implications

The current OMNet-V3 design is already appropriate for this GPU because it uses comparatively lightweight backbones:

1. EfficientNet-B0 for local morphology features.
2. ViT-Tiny/16 for global context.
3. Projection of both branches into a 256-dimensional fusion space.
4. Magnification embedding of 64 dimensions.
5. Magnification-Aware Adaptive Fusion (MAF).
6. Binary classification head.
7. Eight-class subtype classification head.
8. Hierarchical multi-task loss.

The current notebook reports the exact feature dimensions used by the model:

- EfficientNet-B0: 1280-dimensional backbone output.
- ViT-Tiny/16: 192-dimensional backbone output.
- Fusion dimension: 256.
- Magnification embedding: 64.

Do not replace these backbones with much larger CNN/ViT variants merely because the GPU has 16 GB VRAM. The dual-backbone training graph, optimizer state, activations, and cross-validation workload all consume memory together.

## 4. Precision policy — critical ROCm specialization

### Full training default: FP32

For this exact machine/software combination, use FP32 as the default training precision unless a later environment validation proves another mode stable.

The current notebook contains a very important empirical finding:

- A single-batch BF16 autocast forward/backward test passed.
- A BF16 stability test also reported finite gradients and a successful optimizer step.
- However, the full repeated training path subsequently observed a reproducible AMD GPUVM page fault under BF16.
- Repeated FP32 dual-backbone training was stable.
- The production training engine therefore deliberately sets `use_amp = False` and trains in FP32.

This means:

> Do not enable `amp_enabled=True` simply because BF16 reports as supported.

Backend support and workload stability are not the same thing.

### Recommended precision policy

```python
CONFIG["amp_enabled"] = False
TRAIN_DTYPE = torch.float32
```

Use BF16/FP16 only as an explicitly controlled experiment after validating the exact installed PyTorch/ROCm/kernel stack.

## 5. AMP rules

Avoid hard-coding a blanket assumption that AMP is safe just because the GPU reports BF16 support.

The existing notebook tested:

```python
with torch.amp.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    ...
```

and separately checked:

```python
torch.cuda.is_bf16_supported()
```

Those tests are useful diagnostics, not sufficient evidence for full-run stability.

For the production model:

```python
use_amp = False
```

Do not introduce `GradScaler` unless the selected precision mode actually requires it and has been validated on the exact ROCm stack. The current BF16 diagnostic path intentionally did not use `GradScaler`.

## 6. VRAM-aware batch-size policy

The current OMNet-V3 configuration uses:

```python
batch_size = 16
image_size = 224
```

Keep `batch_size=16` as the first-choice baseline.

Do not increase the batch size solely because the card has 16 GB VRAM. OMNet simultaneously trains EfficientNet-B0 and ViT-Tiny/16, so peak memory occurs during backward rather than during the simple input transfer test.

If memory pressure appears:

1. Keep image size at 224.
2. Reduce batch size from 16 to 8.
3. If necessary reduce it further to 4.
4. Prefer gradient accumulation over increasing per-device batch size.
5. Only change model architecture after confirming that memory, not an ROCm kernel issue, is the bottleneck.

Recommended fallback order:

```text
16 -> 8 -> 4
```

Do not jump directly to a much smaller image size because the histopathology task is sensitive to fine morphological detail.

## 7. DataLoader rules for Linux + AMD

The current notebook intentionally uses:

```python
num_workers = 0
pin_memory = True
```

This is a conservative configuration for the target machine.

When increasing `num_workers`, validate the complete training pipeline rather than assuming more workers automatically improve throughput. Linux multiprocessing can increase CPU RAM usage and data-pipeline pressure, while ROCm GPU failures should not be confused with DataLoader failures.

The notebook already implements worker-specific settings only when workers are enabled:

```python
persistent_workers = True
prefetch_factor = 2
```

Recommended starting configuration:

```python
CONFIG["num_workers"] = 0
CONFIG["batch_size"] = 16
```

Then benchmark `num_workers=2` and `num_workers=4` only if the CPU and storage subsystem benefit.

## 8. Pinned memory and non-blocking transfers

The current notebook uses:

```python
pin_memory = True
```

and transfers tensors using:

```python
batch["image"].to(device, non_blocking=True)
```

This pattern is acceptable and should be retained when it gives measurable throughput improvement.

Do not assume that `non_blocking=True` itself solves performance problems. It mainly enables asynchronous transfer when the underlying memory path supports it.

## 9. Device abstraction requirement

Even though ROCm PyTorch uses the CUDA compatibility namespace, avoid scattering literal device logic across the model.

Preferred pattern:

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Then pass `DEVICE` into the training/evaluation functions.

Avoid model code such as:

```python
x = x.cuda()
```

Use:

```python
x = x.to(DEVICE)
```

This keeps the model portable between the friend's AMD Linux system and another machine.

## 10. ROCm-safe coding rules

### Prefer

- PyTorch core operators.
- `torch.nn`, `torch.optim`, `torch.nn.functional`.
- `timm` models that use standard PyTorch operators.
- Standard torchvision transforms.
- Standard tensor indexing, reductions, matrix operations, convolutions, attention, normalization, and activations.
- Explicit device management with `.to(DEVICE)`.
- Runtime capability detection.
- Numerical finite-value checks during development.

### Avoid unless specifically validated

- NVIDIA-only extensions.
- CUDA custom kernels.
- Libraries whose documentation explicitly requires NVIDIA CUDA.
- Hand-written `.cu` extensions.
- CUDA-only fused optimizers.
- Assumptions about cuDNN behavior.
- NVIDIA-specific benchmarking conclusions applied to AMD.

The project should not depend on CUDA source compilation to create the final model.

## 11. cuDNN-specific logic must remain conditional

The notebook already correctly treats cuDNN settings as NVIDIA-specific:

```python
if torch.version.hip is None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Keep this logic conditional. Do not add unconditional cuDNN configuration to the AMD version.

## 12. Memory management

The current notebook uses:

```python
torch.cuda.empty_cache()
```

This is valid under ROCm's CUDA-compatible PyTorch API and can be used between large diagnostic/model lifecycle stages.

However, `empty_cache()` is not a substitute for memory-efficient architecture or batch sizing. Do not insert it into every training batch; that can hurt performance without reducing the underlying live tensor footprint.

Use it mainly when:

- deleting an entire model before constructing another model;
- moving between evaluation and training experiments;
- running multiple diagnostic models sequentially;
- recovering cached allocator memory between large independent stages.

## 13. Model lifecycle and cross-validation memory

OMNet uses 5-fold patient-disjoint cross-validation. This is computationally heavier than a single train/validation split because models, optimizers, metrics, and intermediate artifacts are repeatedly created.

For each fold:

1. Release the previous model before creating the next one.
2. Delete optimizer, scheduler, criterion, and temporary tensors when they are no longer needed.
3. Run garbage collection where necessary.
4. Call `torch.cuda.empty_cache()` after large objects are deleted.
5. Do not keep all fold models resident in VRAM.

A safe lifecycle is:

```text
build fold model
      -> train
      -> evaluate
      -> save checkpoint/metrics
      -> delete model + optimizer + temporary tensors
      -> gc.collect()
      -> torch.cuda.empty_cache()
      -> next fold
```

## 14. Gradient safety is especially important on this system

The current notebook checks for non-finite losses, logits, and gradients and clips the gradient norm at 1.0.

Retain these checks during development because ROCm runtime faults and numerical instability can otherwise appear as generic training failures.

Recommended configuration:

```python
CONFIG["gradient_clip_norm"] = 1.0
```

During new model development, validate at least:

```python
torch.isfinite(loss)
torch.isfinite(outputs["binary_logits"]).all()
torch.isfinite(outputs["subtype_logits"]).all()
```

and verify gradients before the optimizer step.

## 15. Do not confuse a successful smoke test with a stable training run

A GPU smoke test can prove that:

- tensors transfer to the GPU;
- the model can execute a forward pass;
- a small backward pass completes;
- the selected dtype is accepted by the runtime.

It does **not** prove that hundreds or thousands of repeated optimizer steps will be stable.

For this RX 9060 XT environment, this distinction is already demonstrated by the BF16 issue recorded in the notebook.

Therefore every newly introduced optimization must pass:

1. import/startup test;
2. one-batch forward test;
3. one-batch backward test;
4. short multi-batch training test;
5. at least one complete training epoch;
6. only then the full 5-fold experiment.

## 16. Linux-specific filesystem rules

Do not copy Colab/Google Drive assumptions into the friend's model file.

The existing notebook uses local Linux paths such as:

```text
/home/saviour/00_Workspace/Projects/gnit/OMNet/OMNet/output_v3
~/.kaggle/access_token
~/.cache/kagglehub/...
```

The final notebook should construct paths with `pathlib.Path` instead of embedding Windows or Colab paths.

Recommended:

```python
PROJECT_ROOT = Path.cwd().resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output_v3"
```

Avoid hard-coded paths such as:

```text
/content/drive/...
D:\...
C:\Users\...
```

## 17. Kaggle and dataset storage

Because the machine is Linux, use the local filesystem and Kaggle cache directly.

The existing notebook already downloads the dataset using `kagglehub` and discovers the local cache automatically.

Keep dataset extraction off the system root and use a project-local data directory or a fast SSD.

The current BreakHis workload discovered approximately 7,909 images across 82 patients. The dataset is relatively small enough that the main bottleneck should be model training and I/O behavior rather than the raw dataset size.

## 18. Macenko stain augmentation considerations

The current pipeline performs Macenko-style stain perturbation on CPU before converting the image to a tensor.

This is intentionally outside the GPU model graph. Do not move CPU-side NumPy/SVD stain processing into the model unless there is a clear performance reason and ROCm support has been validated.

The current probability is:

```python
stain_aug_prob = 0.5
```

This augmentation is computationally relevant because it can become part of the CPU input bottleneck. If GPU utilization is low, profile the input pipeline before changing the model.

## 19. EfficientNet + ViT dual-branch constraint

The dual-backbone design means both branches should normally receive the same normalized image tensor.

The current input pipeline uses:

```python
Resize -> crop -> flips/rotation -> stain perturbation -> ToTensor -> ImageNet normalization
```

with 224 × 224 final input.

Do not introduce branch-specific preprocessing without experimental justification, because it changes the behavior of the fusion mechanism.

## 20. MAF fusion must remain lightweight

The Magnification-Aware Adaptive Fusion module is not the memory bottleneck. Its inputs are only the projected 256-dimensional CNN and ViT features plus a 64-dimensional magnification embedding.

Keep the fusion layer compact.

Avoid adding large token-level cross-attention between the full CNN feature map and the full ViT token sequence unless you deliberately want a substantially larger model.

Such a modification could increase VRAM usage and may create new ROCm kernel/performance issues.

## 21. Recommended configuration for this computer

Use this as the default starting point:

```python
CONFIG = {
    # Data
    "image_size": 224,
    "train_resize": 256,
    "train_crop": 224,
    "val_resize": 256,
    "val_crop": 224,
    "batch_size": 16,
    "num_workers": 0,

    # Model
    "cnn_backbone": "efficientnet_b0",
    "vit_backbone": "vit_tiny_patch16_224",
    "fusion_dim": 256,
    "magnification_embedding_dim": 64,
    "dropout": 0.3,

    # Training
    "max_epochs": 30,
    "warmup_epochs": 5,
    "early_stopping_patience": 8,
    "base_lr": 1e-5,
    "head_lr": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,

    # AMD / ROCm stability
    "amp_enabled": False,
    "preferred_dtype": "float32",
}
```

## 22. Recommended runtime verification cell

Every AMD-specific notebook should begin by verifying the actual environment instead of assuming the machine matches the design document.

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU is not available to PyTorch.")

DEVICE = torch.device("cuda:0")

props = torch.cuda.get_device_properties(0)

print("Backend : ROCm/HIP")
print("Device  :", torch.cuda.get_device_name(0))
print("Arch    :", getattr(props, "gcnArchName", "unknown"))
print("VRAM    :", props.total_memory / 1024**3, "GiB")
print("PyTorch :", torch.__version__)
print("HIP     :", torch.version.hip)
print("BF16    :", torch.cuda.is_bf16_supported())
```

A model file should use the detected architecture to verify that the machine really is the intended RX 9060 XT environment.

## 23. What must NOT be changed casually

The following are treated as stability-sensitive settings on this machine:

- `batch_size=16`
- `image_size=224`
- EfficientNet-B0 + ViT-Tiny/16 dual branch
- FP32 full training
- `gradient_clip_norm=1.0`
- `num_workers=0` baseline
- patient-disjoint 5-fold evaluation
- ImageNet normalization
- Macenko augmentation probability 0.5

Any change should be benchmarked against the stable baseline.

## 24. Safe optimization order

When optimizing performance, use this order:

1. Verify the ROCm/PyTorch installation.
2. Verify GPU utilization and VRAM usage.
3. Profile DataLoader/CPU preprocessing.
4. Benchmark `num_workers`.
5. Benchmark batch size 16 vs 8 only if memory or throughput requires it.
6. Benchmark model compilation or kernel-level optimizations only after baseline stability is established.
7. Experiment with mixed precision separately from the main production configuration.
8. Only then consider architectural enlargement.

Do not optimize several variables simultaneously; otherwise a new ROCm failure becomes difficult to attribute.

## 25. Compatibility and reproducibility policy

The model repository should record these values alongside every serious experiment:

```text
GPU model
GPU architecture / gfx target
VRAM
Linux distribution and kernel
ROCm version
PyTorch version
Python version
timm version
torchvision version
batch size
image size
precision / dtype
num_workers
random seed
commit hash
```

The existing notebook already prints several of these values. Keep that behavior.

## 26. Current verified baseline from the notebook

The uploaded OMNet-V3 notebook already contains an AMD/Linux-oriented implementation and reports:

- PyTorch: `2.13.0+rocm7.2`
- HIP / ROCm: `7.2.53211`
- GPU: `AMD Radeon RX 9060 XT`
- Architecture: `gfx1200`
- GPU memory: `15.92 GiB`
- Batch size: `16`
- Image size: `224`
- Num workers: `0`
- Production AMP setting: `False`

The notebook's production training engine explicitly documents that repeated BF16 training triggered a reproducible GPUVM page fault on this stack, while repeated FP32 dual-backbone training was stable. Therefore the current model should treat FP32 as the trusted baseline, not BF16.

## 27. Final engineering rule

The model should be designed around **ROCm compatibility + numerical stability + 16 GB VRAM**, not around the assumption that an AMD GPU behaves identically to an NVIDIA CUDA GPU.

The strongest baseline for this specific machine is:

```text
RX 9060 XT 16 GB
        |
      ROCm/HIP
        |
 PyTorch ROCm build
        |
   FP32 training
        |
EfficientNet-B0 + ViT-Tiny/16
        |
      MAF Fusion
        |
 Binary + 8-class heads
        |
 5-fold patient-disjoint CV
```

The model architecture itself does not need to be redesigned merely because the GPU is AMD. The important specialization is in the **runtime, precision policy, memory management, device abstraction, data pipeline, and validation strategy**.

## Sources

- AMD ROCm Linux system requirements and supported GPU table: ROCm documentation, including RX 9060 XT (`gfx1200`) support.
- AMD ROCm Radeon Linux compatibility matrix for ROCm 7.2 and PyTorch production support.
- AMD ROCm GPU specification table for RX 9060 XT hardware characteristics.
- Project source: `OMNet_BreakHis_V3.ipynb`, especially the environment, GPU detection, data loader, AMP diagnostics, and training-engine sections.

Official documentation domains:

- `rocm.docs.amd.com`
- `pytorch.org`

## Source notes from the current OMNet implementation

The uploaded notebook identifies OMNet-V3 as a dual-branch EfficientNet-B0 + ViT-Tiny/16 architecture with Magnification-Aware Adaptive Fusion and hierarchical multi-task loss. It uses 224 × 224 inputs, batch size 16, five patient-disjoint folds, and 256-dimensional fusion with a 64-dimensional magnification embedding. fileciteturn1file0L20-L38 fileciteturn1file0L403-L443

The same notebook verifies the RX 9060 XT, `gfx1200`, and approximately 15.92 GiB VRAM, and reports a ROCm/HIP runtime. fileciteturn1file0L211-L247

The current training engine explicitly records the BF16 GPUVM page-fault issue and intentionally uses FP32 for repeated training. fileciteturn3file0L1556-L1594

The notebook also confirms that the DataLoader baseline is batch size 16 with `num_workers=0` and pinned memory, while GPU transfers use `non_blocking=True`. fileciteturn3file0L38-L99
