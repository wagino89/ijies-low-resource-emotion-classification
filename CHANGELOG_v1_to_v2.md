# Version-to-version change log (first submission → revision)

IJIES Paper ID 20266676. This file answers Reviewer 3, points 2 and 3, of the second review round.

* **v1, first submission**: notebooks `01_main_experiment.ipynb` and `02_ablation_lr_regularization.ipynb`; outputs in `results/main/`, `results/ablation_lr_regularization/`, `results/post_hoc/`.
* **v2, revision**: notebooks `03`–`06b`; outputs in `results/rerun_*` and `results/rerun_tables/`.
* **Second-round checks**: notebooks `07_reconcile_lr_cpu.ipynb` (`results/reconcile_lr/`) and `08_crossover_decomposition_gpu.ipynb` (`results/crossover_decomp/`).

Every revised principal result is linked to its raw per-seed results, per-seed predictions, split records, selection records, the test-label exclusion checks that cover it, and its analysis script in [`results/table_to_file.csv`](results/table_to_file.csv), with SHA-256 digests of the result files.

## Release history

| Release / commit | Date | Manuscript version | Content |
|---|---|---|---|
| `30ba992` (first commit) | 2026-10-05 | first submission (v1) | notebooks 01–02; `results/main/`, `results/ablation_lr_regularization/`, `results/post_hoc/` |
| `2c99770` | 2026-10-28 | first revision (v2) | notebooks 03–06b; `results/rerun_*`. Its README still used the v2 table numbering of that round (Table 10 = re-implemented systems, Table 11 = SHAP) and contained the withdrawn exact-reproduction statement (§3) |
| tag `ijies-20266676-r2` = `3b80d3d` | 2026-11-10 | second revision | notebooks 07–08, this change log, `table_to_file.csv`, test-label audit; README renumbered (Table 10 = label budget and tuning, Table 11 = re-implemented systems) |
| tag `ijies-20266676-r3` | see GitHub *Releases* | third revision (cited in the manuscript) | documentation only, **no result changed**: §3 extended (identity of the earlier run, cell-by-cell and per-seed decomposition, software and embedding differences, relationship of the embedding caches), §6 (fine-tuning step counts), `results/reconcile_lr/D_table3_LR_decomposition.csv`, `results/rerun_finetune/ft_step_budget.csv`, split-record and test-label-check columns in `table_to_file.csv` |

The manuscript cites tag `ijies-20266676-r3`. The table numbering of the manuscript and of this README is that of r3 (identical to r2).

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

### 3a. Which earlier run the exact-reproduction statement referred to (third round)

The withdrawn statement rested on rows 6–8 of `results/rerun_frozen/determinism_check.csv` (notebook 03, cell §(c) "LR-only Tokopedia results vs the original run"). The "original run" there is the **first-submission run** of `notebooks/01_main_experiment.ipynb`, whose per-seed scores are `results/main/Tokopedia_final_raw.csv` (commit `30ba992`). Only Tokopedia, seed 42, and test Macro F1 at six decimals were compared: XLM-R LR 0.577854 = 0.577854 and IndoBERT LR 0.590687 = 0.590687 (identical), E5 LR 0.620776 vs. 0.622023 (different). Predictions and probabilities were not compared, and SemEval was not checked. The code comment of that cell in notebook 03 ("identical embeddings + splits => identical LR result") is therefore an assumption that does not hold; the notebook is left unchanged so that it matches the run that produced the results, and this paragraph replaces that comment.

### 3b. Cell-by-cell and per-seed decomposition of the Table 3 LR rows

`results/reconcile_lr/D_table3_LR_decomposition.csv` lists, for every model × seed × language (Sundanese, Javanese, Indonesian, language average, pooled): the v1 Table 3 value, the value of notebook 07 with the v1 folds, the v2 Table 3 value (identical to notebook 07 with the v2 folds in every cell), the total change (v2 − v1), the **fold effect** (v2 folds − v1 folds, same embeddings, same code) and the **residual** (notebook 07 with v1 folds − v1). Total = fold effect + residual in every cell.

Per-language means (10 seeds):

| Model | Cell | v1 | v2 | Total | Fold effect | Residual | Seeds with residual = 0 |
|---|---|---|---|---|---|---|---|
| XLM-R LR | Sundanese | 0.3793 | 0.3790 | −0.0004 | +0.0006 | −0.0009 | 1/10 |
| XLM-R LR | Javanese | 0.3903 | 0.3793 | −0.0110 | −0.0106 | −0.0004 | 9/10 |
| XLM-R LR | Indonesian | 0.4203 | 0.4092 | −0.0111 | −0.0113 | +0.0002 | 8/10 |
| XLM-R LR | **Average** | **0.3966** | **0.3892** | **−0.0075** | **−0.0071** | **−0.0003** | 1/10 |
| XLM-R LR | Pooled | 0.3962 | 0.3933 | −0.0029 | −0.0022 | −0.0006 | 1/10 |
| IndoBERT LR | Sundanese | 0.3804 | 0.3780 | −0.0025 | −0.0028 | +0.0004 | 5/10 |
| IndoBERT LR | Javanese | 0.3346 | 0.3403 | +0.0057 | +0.0036 | +0.0021 | 5/10 |
| IndoBERT LR | Indonesian | 0.3956 | 0.3855 | −0.0102 | −0.0090 | −0.0011 | 8/10 |
| IndoBERT LR | **Average** | **0.3702** | **0.3679** | **−0.0023** | **−0.0028** | **+0.0005** | 3/10 |
| IndoBERT LR | Pooled | 0.3850 | 0.3830 | −0.0020 | −0.0027 | +0.0007 | 3/10 |

Per-seed language averages:

| Seed | XLM-R v1 | XLM-R v2 | fold | residual | IndoBERT v1 | IndoBERT v2 | fold | residual |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.3859 | 0.3748 | −0.0106 | −0.0006 | 0.3241 | 0.3285 | +0.0044 | 0.0000 |
| 7 | 0.4117 | 0.3876 | −0.0240 | −0.0001 | 0.4078 | 0.3931 | −0.0109 | −0.0038 |
| 13 | 0.4406 | 0.4401 | −0.0012 | +0.0007 | 0.3537 | 0.3405 | −0.0211 | +0.0078 |
| 77 | 0.3997 | 0.3880 | −0.0086 | −0.0031 | 0.3473 | 0.3414 | +0.0003 | −0.0062 |
| 101 | 0.4093 | 0.4148 | +0.0059 | −0.0004 | 0.3470 | 0.3323 | −0.0165 | +0.0019 |
| 202 | 0.4300 | 0.4031 | −0.0268 | 0.0000 | 0.4047 | 0.4044 | +0.0001 | −0.0003 |
| 303 | 0.3340 | 0.3324 | −0.0042 | +0.0025 | 0.3730 | 0.3789 | +0.0061 | −0.0002 |
| 404 | 0.4158 | 0.3999 | −0.0142 | −0.0017 | 0.3823 | 0.3976 | +0.0153 | 0.0000 |
| 505 | 0.3535 | 0.3778 | +0.0246 | −0.0003 | 0.3981 | 0.3907 | −0.0128 | +0.0053 |
| 606 | 0.3857 | 0.3731 | −0.0122 | −0.0004 | 0.3640 | 0.3716 | +0.0077 | 0.0000 |

A residual of 0.0000 means that the seed is reproduced exactly. The fold effect accounts for 95% (XLM-R) and more than 100% (IndoBERT; the residual has the opposite sign) of the change of the language average.

### 3c. What changed, and what is responsible

| Change between v1 and v2 | Affects LR-only scores? | Evidence |
|---|---|---|
| Inner folds stratified on emotions **and language** (`FOLD_STRATIFY_BY_LANG = True`; code change in notebook 03) | **Yes — the main cause** | fold-effect column above; with the v1 folds the language averages return to v1 within 0.0005 (XLM-R, IndoBERT) and 0.0009 (E5) |
| Embedding matrices (see §3d) | Yes, small | residual column; same splits, folds, code and scikit-learn 1.6.1, yet stored v1 probabilities differ by up to 0.027 (XLM-R), 0.024 (IndoBERT) and 0.059 (E5) on SemEval and by up to 0.017 on Tokopedia seed 42 (XLM-R) with identical predictions |
| NumPy 2.0.2 → 2.1.3 and Python 3.12 → 3.13 (BLAS used by `lbfgs` and PCA) | Possibly, far below the residual | not separable from the embedding difference without the v1 matrices |
| XGBoost 3.2.0 → 3.4.1, `n_jobs`, input rounding | **No** (LR-only) | — |
| Splits, preprocessing, encoding code, PCA, LR settings, threshold and class-prior rules | No (unchanged) | `split_check.csv` (identical SHA-256); the `encode` and `cached_embeddings` functions of notebooks 01, 03 and 07 are identical apart from a log message |

### 3d. Relationship between the earlier and the revised embedding caches

* **v1 (notebook 01, Kaggle T4 GPU, PyTorch 2.10.0+cu128, Transformers 5.0.0).** `cached_embeddings` uses an attached `embedding_cache/*.npz` if one is found under `/kaggle/input`; otherwise it encodes the texts on the GPU (fp32, batch 16) and writes `embedding_cache/*.npz` to the session output. The v1 notebook output was not kept, so it is not known which of the two paths ran, and neither the matrices nor their fingerprints were exported.
* **v2 (notebooks 03, 05, 07, Kaggle CPU).** All frozen-pipeline sessions read the twelve matrices from the attached Kaggle dataset `embedding-cache-me-xbal` (`/kaggle/input/datasets/rezkanorhafizah/embedding-cache-me-xbal/embedding_cache`) and never re-encode (log lines `[cache] …` for all twelve matrices in notebook 03). Their fingerprints are recorded (`results/reconcile_lr/embedding_fingerprints.csv`; Tokopedia also in `results/rerun_frozen/environment_cpu.json`) and are identical in notebooks 03 and 07.
* **Relationship.** Both caches are produced by the same encoding code from the same texts, with the encoders loaded by name and without a pinned model revision. The residuals of §3b show that the matrices used by v1 were **not bit-identical** to the archived v2 cache. The archived cache is therefore the reference for every revised result; v1 results can be reproduced only approximately (within the residuals above).
* **Library sensitivity.** Re-encoding SemEval with Transformers 5.20.2 on CPU (notebook 07) gives minimum cosine similarities to the archive of 0.977 (E5), 0.998 (XLM-R) and **0.31** (IndoBERT) (`embedding_fresh_vs_cache.csv`). Because the encoding code is unchanged, this difference comes from the library and model-file versions; the IndoBERT matrices can only be reproduced from the archive.

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

## 6. Fine-tuning step counts (third round)

Notebook 04 (and the tuned comparators of Table 10) sets the epoch limit of a run to E = min(40, max(10, ⌈300 / s⌉)), where s = ⌈m / 16⌉ is the number of batches of the m fitting texts. Each epoch is one pass over a fresh permutation of the m texts; the last batch is smaller when m is not a multiple of 16; there is no resampling, repetition or `drop_last`. The warm-up (10%) and linear decay are defined over the planned E · s steps. The "≥ 300 steps" of earlier wording is therefore a target that the 40-epoch cap overrides at the smallest size, and early stopping ends most runs earlier. `results/rerun_finetune/ft_step_budget.csv` lists every regime, comparator and size:

| Case | m | s | E | Planned steps | Executed steps (min–max) |
|---|---|---|---|---|---|
| Regime B, n = 100 (full FT / LoRA) | 100 | 7 | 40 | **280** | 91–147 / 77–147 |
| Regime A, n = 100, early-stopping fit | 80 | 5 | 40 | **200** | 20–80 |
| Regime A, n = 100, refit on all n | 100 | 7 | selected epoch (1–13) | = executed | 7–91 |
| Regime B, n = 250 | 250 | 16 | 19 | 304 | 144–240 / 128–304 |
| Regime A, n = 250, early-stopping fit | 200 | 13 | 24 | 312 | 91–260 |
| n ≥ 500 (both regimes) | ≥ 400 | ≥ 25 | 10–12 | ≥ 300 | see file |

In regime A, the refit trains for the selected number of epochs with the schedule spanning that shorter length; it has no 300-step floor. The pilot search used the same rule. The default recipe of Table 10 (≤ 5 epochs, patience 2, constant learning rate) has no step target.
