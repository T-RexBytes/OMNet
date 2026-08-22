# Research Limitations & Future Directions

1. **Discrete Patient Scarcity in Rare Subtypes:** Subtypes with fewer patients than cross-validation folds ($N=3$ for Phyllodes Tumor, $N=4$ for Adenosis) are mathematically absent in some test folds, depressing unweighted Macro-F1.
2. **Single-Dataset Domain:** Benchmarked on BreakHis; external multi-center dataset validation (e.g., TCGA, CAMELYON) is recommended to evaluate cross-institutional stain domain shift.
3. **Patch vs. Whole Slide Image (WSI):** Models operate on pre-extracted $224 \times 224$ tiles; scaling to WSI requires integration with gigapixel slide aggregation methods (e.g., CLAM, ABMIL).
4. **Hardware Precision Constraints:** Execution on AMD RDNA 4 hardware under ROCm 7.2 required FP32 precision to avoid BF16 page faults, which slightly increases training time compared to hardware-native FP16/BF16.
