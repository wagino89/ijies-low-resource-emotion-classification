# Reconciliation summary (2026-11-09 14:40)

Reference: git 2c997704beeb8b1c86e1539146864071754ef3bd. Splits identical to both submissions: True.

- SemEval, procedure `first_submission_folds` vs first submission: 5/30 model×seed runs with identical per-language scores; max |Δp| = 5.86e-02; texts with a different label set = 99.
- SemEval, procedure `first_submission_folds` vs revision: 0/30 model×seed runs with identical per-language scores; max |Δp| = 0.00e+00; texts with a different label set = 1381.
- SemEval, procedure `revision_folds` vs first submission: 0/30 model×seed runs with identical per-language scores; max |Δp| = 5.86e-02; texts with a different label set = 1382.
- SemEval, procedure `revision_folds` vs revision: 30/30 model×seed runs with identical per-language scores; max |Δp| = 0.00e+00; texts with a different label set = 0.
- Tokopedia vs first submission: 12/30 model×seed runs identical; max |ΔF1| = 0.0062; causes of the remaining differences: same coefficients; test probabilities differ at the decision boundary (16); class-prior search: both coefficient vectors lie within the tie tolerance on the out-of-fold data (2)
- Tokopedia vs revision: 30/30 model×seed runs identical; max |ΔF1| = 0.0000; causes of the remaining differences: 
