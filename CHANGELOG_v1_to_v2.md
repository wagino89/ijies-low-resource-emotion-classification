# Version-to-version change log (first submission → revision)

IJIES Paper ID 20266676. This file answers Reviewer 3, points 2 and 3, of the second review round.

* **v1, first submission**: notebooks `01_main_experiment.ipynb` and `02_ablation_lr_regularization.ipynb`; outputs in `results/main/`, `results/ablation_lr_regularization/`, `results/post_hoc/`.
* **v2, revision**: notebooks `03`–`06b`; outputs in `results/rerun_*` and `results/rerun_tables/`.
* **Second-round checks**: notebooks `07_reconcile_lr_cpu.ipynb` (`results/reconcile_lr/`) and `08_crossover_decomposition_gpu.ipynb` (`results/crossover_decomp/`).

Every revised principal result is linked to its raw per-seed results, predictions, selection records, and analysis script in [`results/table_to_file.csv`](results/table_to_file.csv), with SHA-256 digests of the result files. Version 2 and the second-round checks are frozen in git tag `ijies-20266676-r2`. The commit of the first revision is `2c997704beeb8b1c86e1539146864071754ef3bd`.

## 1. What changed

| Item | v1 (first submission) | v2 (revision) | Effect on reported numbers |
|---|---|---|---|
| Train / validation / test indices | Two-stage split per language and per seed (65/15/20), `random_state = seed` | **Unchanged.** `split_manifest_all.csv` is byte-identical in v1, v2 and the reconciliation run (SHA-256 `d83e011b…`) | None |
| Text preprocessing and representation | Prefix `query: ` (E5), max. length 128, masked mean pooling, L2 normalisation, StandardScaler, PCA (95% variance) re-fitted in every fold | **Unchanged** | None |
| Embedding matrices | Encoded in the v1 Kaggle GPU session or read from a cache; which one was used and their fingerprints were not recorded | Archived cache `embedding-cache-me-xbal`, with fingerprints in `environment_cpu.json` (Tokopedia) and `reconcile_lr/embedding_fingerprints.csv` (all matrices) | Small differences: see §3 |
| Inner cross-validation folds (SemEval) | `MultilabelStratifiedKFold` on the emotion labels only | Stratified on the emotion labels **and** the language (`FOLD_STRATIFY_BY_LANG = True`) | **Main cause of the changed SemEval scores, including the LR rows of Table 3** (see §3) |
| Ensemble-weight objective, threshold and class-prior rules, grids, tie rules | Pooled OOF Macro F1; τ 0.15–0.85, precision ≥ 0.20, positive-rate cap; α 0.5–3.0, 3 passes | **Unchanged** (functions identical; v2 additionally logs fallbacks and constant fits) | None |
| XGBoost runtime | `n_jobs = -1`, raw inputs, XGBoost 3.2.0 | `n_jobs = 4`, inputs rounded to 10⁻⁵, XGBoost 3.4.1 | XGBoost-based scores differ; LR-only models unaffected |
| Configurations (E1) | 11 models | 11 models + 7 factorial controls; all fitted in **one** session (v1 ablation came from a separate session) | New Table 8 |
| Selected hyper-parameters per run | Stored only inside the prediction files | Every selected weight, threshold vector, α vector, fallback, and constant fit in `rerun_frozen/selected_params.csv` | Records only |
| Learning-curve subsets | Independent stratified subsets per size (`random_state = seed + 10000 + n`), 5 seeds, not exported | **Nested** subsets (`lc_subset_order.csv`), 10 seeds | See §4 |
| Label budget of the learning curve | E2b: frozen pipeline fits and selects on *n*; fine-tuning early-stops on the 809 validation texts | Regime A (both see only *n* labels), regime B (both select on the validation partition), regime C (frozen also fits on validation; = v1 E2) | See §4 |
| Fine-tuning comparator | One configuration: first token, unweighted CE, constant lr 2×10⁻⁵, no warm-up, ≤ 5 epochs, patience 2, fp32 | Validation search over 8 full + 2 LoRA configurations; warm-up + decay, class-weighted CE, 10–40 epochs, patience 3, fp16 | See §4 |
| Fine-tuning checkpoints and histories | Not saved | Per-epoch histories (`ft_history.csv`), LoRA adapters of every run, fp16 checkpoints for seed 42 at *n* = 100 / 1,000 / 3,510 (GitHub Releases) | Records only |
| Re-implemented SemEval-2025 systems | None | E4: UoB-NLP-style adapter and Team UBD-style system (notebooks 06, 06b) | New Table 11 |
| SHAP | 59 texts (most confident correct), explained probability before decision adjustment, inconsistent mask token | 191 of 197 prespecified stratified texts (correct/incorrect × confidence terciles), explained decision-adjusted score, `<mask>` in SHAP and AOPC, 10 random rankings, exclusions recorded | New Table 12 |
| Statistics | Wilcoxon over seeds; percentile bootstrap over seed differences; no multiplicity correction | Nadeau–Bengio corrected resampled t-test (ratio 0.25; learning curve also 1081/*n*), Wilcoxon, Holm within each family | p-values and CIs changed |
| Hardware / software | Kaggle session with Tesla T4; Python 3.12.13, PyTorch 2.10, Transformers 5.0.0, scikit-learn 1.6.1, XGBoost 3.2.0, NumPy 2.0.2 (CPU model not recorded) | Frozen pipeline and SHAP: Kaggle CPU, 4 cores Intel Xeon 2.20 GHz; Python 3.13.15, PyTorch 2.11 (CPU), Transformers 5.16.1, scikit-learn 1.6.1, XGBoost 3.4.1, NumPy 2.1.3. Fine-tuning and E4: Tesla T4 | See §3 |

## 2. Newly trained versus re-analysed

| Revised result | Origin |
|---|---|
| Tables 3, 5, 8; Table 9 and Fig. 2; Table 11; Table 12 and Figs. 3–6 | **Newly trained models** in the revision sessions (notebooks 03, 04, 05, 06, 06b). No v1 prediction is reused. |
| Tables 4, 6, 7 | Re-analysis of the stored v2 predictions and per-seed scores (`scripts/analyze_rerun.py`, `scripts/post_hoc_rerun.py`) |
| Table 10 (new) | Newly trained original fine-tuning recipe (called the *default recipe* in the manuscript; notebook 08, 10 seeds), combined with the stored v2 frozen and tuned fine-tuning results. The v1 E2b budget is called regime *U* (unmatched) in the manuscript. The v1 learning curve, re-analysed with the v2 statistics (`decomp_cells.csv`, cell `first submission E2b`), is reported in the response letter and here only, not in the manuscript |
| Section 4.2 reconciliation | Newly trained LR-only models (notebook 07) compared text by text with the stored predictions of v1 and v2 |

## 3. Reconciliation of the LR-only results (Reviewer 3, point 1)

The revision stated that LR results on XLM-R and IndoBERT reproduce the first submission exactly. That statement rested on one check, Tokopedia seed 42 (`determinism_check.csv`), and is withdrawn. Notebook 07 re-ran XLM-R LR, IndoBERT LR and E5 LR on all ten seeds, with the v1 and v2 fold procedures, from the archived embedding cache (`results/reconcile_lr/`):

| Language-averaged Macro F1 (SemEval) | v1, Table 3 | Notebook 07, v1 folds | v2, Table 3 | Notebook 07, v2 folds |
|---|---|---|---|---|
| XLM-R LR | 0.3966 | 0.3963 | 0.3892 | 0.3892 |
| IndoBERT LR | 0.3702 | 0.3707 | 0.3679 | 0.3679 |
| E5 LR | 0.3935 | 0.3926 | 0.3989 | 0.3989 |

* With the v2 folds, notebook 07 reproduces **every** v2 prediction (30/30 SemEval runs and 30/30 Tokopedia runs; maximum probability difference 0).
* With the v1 folds, the means return to v1 within 0.001, so the change of the fold stratification explains the change of Table 3. It also explains most of the E5 LR change, which the revision had attributed to the embeddings.
* The remaining v1-versus-notebook-07 differences are small: 21 (E5), 32 (XLM-R) and 46 (IndoBERT) of about 2,900 SemEval test texts per model change their label set, with at most 0.008 per seed. On Tokopedia, 8, 3 and 1 of 10 seeds are identical (XLM-R, IndoBERT, E5), with |ΔF1| ≤ 0.006. In 16 runs the class-prior coefficients are equal and test probabilities differ at the decision boundary; in 2 runs both coefficient vectors lie within the tie tolerance. Probability differences of 0.002–0.06 are too large for floating-point noise and are consistent with v1 embeddings that are not bit-identical to the archived cache. This cannot be verified because v1 did not store its embedding matrices.
* Re-encoding SemEval with the current libraries on CPU gives embeddings close to the cache for E5 and XLM-R (minimum cosine 0.977 and 0.998), but **not** for IndoBERT (minimum cosine 0.31). Reproduction therefore requires the archived cache; its fingerprints are recorded.

## 4. Why the fine-tuning crossover changed (Reviewer 3, point 4)

Notebook 08 re-ran the **v1 fine-tuning recipe** in regimes A and B on the v2 nested subsets and seeds. The recipe is faithful to v1: replaying v1 runs gives 0.2702 vs. 0.2702 (seed 42, *n* = 100) and 0.2988 vs. 0.2984 (seed 7, *n* = 100). Δ is the frozen pipeline minus fine-tuning (10 seeds; † = Holm-adjusted *p* < 0.05, primary ratio):

| *n* | v1 E2b (5 seeds) | E2b budget, v1 recipe | A, v1 recipe | A, tuned | B, v1 recipe | B, tuned | Tuning effect (B) | Budget effect (E2b→A, v1 recipe) |
|---|---|---|---|---|---|---|---|---|
| 100 | +0.163 | +0.229 † | +0.325 † | +0.108 | +0.246 † | −0.005 | −0.251 † | +0.095 |
| 250 | +0.049 | +0.081 † | +0.143 † | −0.020 | +0.088 † | −0.028 † | −0.116 † | +0.062 |
| 500 | −0.031 | −0.025 | −0.018 | −0.049 | −0.030 | −0.067 † | −0.037 | +0.007 |
| 750 | −0.055 | −0.070 † | −0.051 | −0.067 † | −0.070 † | −0.076 † | −0.006 | +0.018 |
| 1,000 | −0.062 | −0.067 † | −0.065 † | −0.072 † | −0.071 † | −0.077 † | −0.006 | +0.002 |
| 3,510 | −0.063 † | −0.061 † | — | — | −0.055 † | −0.057 † | −0.002 | — |

* The nested design and the 10 seeds reproduce the v1 pattern under the v1 budget and recipe.
* Correcting the label budget does not move the crossover towards fine-tuning; in regime A it slightly favours the frozen pipeline (not significant).
* Tuning the comparator causes the change: −0.251 at 100 texts and −0.116 at 250 texts, < 0.04 from 500 texts. The v1 recipe was unstable on small subsets (1.6 and 2.8 predicted classes on average at *n* = 100 in regimes A and B; epoch cap reached in 70–100% of regime-B runs up to *n* = 1,000).
* LoRA versus tuned full fine-tuning: ≤ 0.016, not significant at any size.

Full tables: `results/crossover_decomp/decomp_cells.csv`, `decomp_effects.csv`, `decomp_crossover_summary.csv`; figure `decomp_figure.png`.

## 5. No test-label access during fitting or selection

**Code level.** In every notebook, the test partition enters only the final evaluation:

* **Frozen pipeline (notebooks 03, 05, 07).** `run_sem` / `run_tok` compute the out-of-fold probabilities, the ensemble weight, the thresholds and the class priors on the development partition (training + validation; regime A: the *n* texts; regime B: the validation partition). Only afterwards do they predict the test indices.
* **Fine-tuning (notebooks 04, 08).** Early stopping uses the validation partition (regime B) or an 80/20 split of the *n* training texts (regime A). The pilot search uses seeds 42 and 7 with the validation partition.
* **E4 (notebooks 06, 06b).** Learning rate, epoch and thresholds are selected on the validation partition.
* **SHAP (notebook 05).** It explains test texts after fitting and selection are complete. The gold labels of the sample define the reporting strata (correct / misclassified) and are not used by any model.

**Data level.** `scripts/audit_test_access.py` (output `results/audit_test_access.csv`, all checks passed) verifies the following:

* the partitions are disjoint for every dataset, language and seed;
* the split manifests of v1, v2 and notebook 07 are identical;
* every learning-curve subset lies in the training partition;
* every regime-A early-stopping split lies inside its *n*-text subset (notebooks 04 and 08);
* the SHAP sample lies in the seed-42 test partition.
