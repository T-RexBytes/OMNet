# Key Research Findings & Takeaways

1. **Zero-Leakage Benchmark Integrity:** Models evaluated with strict patient-disjoint grouping achieve $\approx 84.56\%$ accuracy on BreakHis, revealing that published claims of $>98\%$ accuracy are predominantly artefacts of patch-level data contamination across identical patients.
2. **Vision Transformer Sensitivity Superiority:** Pretrained Vision Transformers (ViT-B/16) demonstrated exceptional sensitivity ($97.45\%$) on unseen test patients, outperforming pure CNNs in clinical malignancy screening.
3. **Soft-Voting Ensemble Specificity Gain:** Ensembling CNN and Transformer predictions maximized Specificity ($55.95\%$) and Precision-Recall AUC ($0.9537$), reducing false-positive rates on complex benign proliferations.
4. **Interpretable Scale Gating:** The MAF module learns an autonomous, scale-conditioned gating schedule that shifts feature reliance from global transformer context at $40\text{X}$ to localized convolutional detail at $400\text{X}$.
5. **Hierarchical Consistency Regularization:** Multi-task training with KL consistency prevents contradictory predictions between coarse binary malignancy and fine-grained subtyping.
