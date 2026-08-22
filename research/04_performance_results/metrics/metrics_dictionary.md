# Unified Metrics Dictionary & Evaluation Protocol

| Metric Name | Mathematical Definition / Formula | Diagnostic Focus | Evaluation Split |
|---|---|---|---|
| **Accuracy** | $rac{TP + TN}{TP + TN + FP + FN}$ | Overall classification correctness | Held-out Test & CV Folds |
| **Precision (PPV)** | $rac{TP}{TP + FP}$ | Fraction of predicted malignancies that are true malignancies | Held-out Test |
| **Recall / Sensitivity (TPR)** | $rac{TP}{TP + FN}$ | Clinical cancer screening sensitivity (minimizing false negatives) | Held-out Test |
| **Specificity (TNR)** | $rac{TN}{TN + FP}$ | True benign identification rate (avoiding overtreatment) | Held-out Test |
| **F1-Score (Binary)** | $rac{2 \cdot 	ext{Precision} \cdot 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$ | Harmonic mean of precision and recall | Held-out Test |
| **Macro-F1 (Multiclass)** | $rac{1}{C} \sum_{c=1}^C F1_c$ | Unweighted mean F1 across 8 subtypes (equal class importance) | 5-Fold Stratified Group CV |
| **Weighted-F1 (Multiclass)** | $\sum_{c=1}^C rac{N_c}{N_{	ext{total}}} F1_c$ | Frequency-weighted subtype F1 score | 5-Fold Stratified Group CV |
| **Balanced Accuracy** | $rac{1}{C} \sum_{c=1}^C 	ext{Recall}_c$ | Macro-average recall (invariant to class imbalance) | Held-out Test & CV Folds |
| **Matthews Correlation Coefficient (MCC)** | $rac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Chance-corrected correlation coefficient ($\in [-1, 1]$) | Held-out Test & CV Folds |
| **ROC-AUC** | $\int 	ext{TPR}(t) \, d(	ext{FPR}(t))$ | Threshold-independent discrimination power | Held-out Test & CV Folds |
| **PR-AUC** | $\int 	ext{Precision}(r) \, dr$ | Performance across high-recall operating points | Held-out Test |
