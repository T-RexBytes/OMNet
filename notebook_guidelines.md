# Guidelines for Baseline Notebook Implementation (Experiment 2 & 3)

Use this document as a reference when implementing Experiment 2 (ViT-B/16) and subsequent notebooks to ensure parity, compatibility, and error-free execution.

---

## 1. Locked Decisions & Dataset Continuity
* **Same Assets & Split:** Reuse the generated `/content/drive/MyDrive/output_CNN/split.csv` to ensure identical patient-level splits across all experiments. Do not regenerate a new split for Experiment 2 or 3.
* **Magnification:** Retain 200X magnification lock.
* **Outputs:** Store all checkpoints and metric files under `/content/drive/MyDrive/output_CNN/` (for CNN baselines) or equivalent directories (e.g., `output_ViT` for ViT baselines) so they are kept separate but accessible.

---

## 2. Mandatory Rules & Guardrails
* **No Emojis in Notebook Code/Markdown Cells:** Emojis and non-standard Unicode characters cause encoding and layout rendering failures in certain environments (e.g., standard Windows consoles and some Jupyter/Javscript engines). Keep print statements and markdown cells plain text.
* **Free GPU Constraints:** Keep batches small (default `batch_size=16`) and clear PyTorch's GPU cache `torch.cuda.empty_cache()` between folds and phases to avoid Out-Of-Memory (OOM) exceptions.

---

## 3. Errors Fixed (Do Not Repeat)

### A. CUDA Device Properties (AttributeError)
* **Problem:** Accessing `.total_mem` on `torch.cuda.get_device_properties(device)` throws:
  `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`.
* **Fix:** Always use `.total_memory` instead of `.total_mem`:
  ```python
  torch.cuda.get_device_properties(0).total_memory
  ```

### B. DataLoader Multiprocessing (AssertionError)
* **Problem:** Creating and deleting temporary PyTorch `DataLoader` objects in loops (like 5-fold cross-validation) with `num_workers > 0` causes background worker processes to crash during garbage collection on Colab/Jupyter:
  `AssertionError: can only test a child process`.
* **Fix:** Set `num_workers = 0` for all DataLoaders created inside loops or iterations (such as cross-validation folds). Set the default dataset `num_workers` in `CONFIG` to `0` to guarantee stability in Colab.

### C. PyTorch Autocast & GradScaler Deprecations (FutureWarning)
* **Problem:** `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` are deprecated in newer versions of PyTorch (2.4+), causing verbose `FutureWarning` logs during training.
* **Fix:** Use the modern, device-specified API endpoints under `torch.amp`:
  ```python
  # Autocast
  with torch.amp.autocast('cuda', enabled=mixed_precision):
      logits = model(images)

  # GradScaler
  scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)
  ```

### D. NumPy Reduction Overflow with Autocast float16 Tensors (RuntimeWarning)
* **Problem:** Tensors computed inside `torch.amp.autocast('cuda')` (such as gate activation vectors, intermediate logits, or embeddings) are returned in half-precision (`torch.float16`). When appended and converted to NumPy via `.numpy()` without casting, reductions like `np.mean()`, `np.std()`, or `np.sum()` on large arrays perform summation in `float16` by default. Once the intermediate sum exceeds `65,504` (the maximum finite value for IEEE 754 float16), NumPy triggers:
  `RuntimeWarning: overflow encountered in reduce: arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)`.
* **Fix:** Always explicitly cast tensors to full precision (`.float().cpu()`) before converting to NumPy, and pass `dtype=np.float64` to `np.mean` and `np.std` reductions:
  ```python
  # Ensure float32 before numpy conversion
  gate_values.append(g.float().cpu())
  g_all = torch.cat(gate_values, dim=0).float().numpy()

  # Reductions with float64 accumulator
  gate_mean = float(np.mean(g_all, dtype=np.float64))
  gate_std  = float(np.std(g_all, dtype=np.float64))
  ```

