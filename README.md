# Frozen Multilingual Encoder or Fine-Tuning? — Code and Results

Code, exported results, and figures for the manuscript *"Frozen Multilingual Encoder or Fine-Tuning? A Controlled Empirical Study of Imbalance-Aware Emotion Classification in Low-Resource Languages with Faithfulness-Tested SHAP"* (International Journal of Intelligent Engineering and Systems, IJIES, Paper ID 20266676).

> **Cited release:** tag [`ijies-20266676-r3`](https://github.com/wagino89/ijies-low-resource-emotion-classification/releases/tag/ijies-20266676-r3). Table numbers in this README follow the r3 manuscript (Table 10 = label budget and comparator tuning; Table 11 = re-implemented SemEval-2025 systems; Table 12 = SHAP faithfulness). The release history, including the older README of commit `2c99770` that still used the earlier table numbering, is in [`CHANGELOG_v1_to_v2.md`](CHANGELOG_v1_to_v2.md#release-history).

The study evaluates a frozen **multilingual-E5-base** encoder (prefix `query: `, masked mean pooling, L2 normalization, StandardScaler, PCA with 95% variance) followed by an **imbalance-aware head**: cost-sensitive XGBoost plus class-weighted logistic regression with C = 0.1. The ensemble weight, the per-emotion thresholds (multi-label), and the class-prior coefficients (single-label) are all selected out of fold. The study runs four experiments on SemEval-2025 Task 11 (Sundanese, Javanese, Indonesian) and on the Tokopedia review corpus PRDECT-ID:

| Experiment | Question | Notebook |
|---|---|---|
| **E1** | Frozen heads on multilingual-E5, XLM-R, and IndoBERT; 2 × 2 factorials (class weighting × regularization) for the LR and hybrid heads (RQ1, RQ2) | `03-rerun-frozen-cpu.ipynb` |
| **E2** | Learning curve on Tokopedia against tuned full fine-tuning and LoRA under three label-budget regimes (RQ3) | `03-rerun-frozen-cpu.ipynb` (frozen) + `04-rerun-finetune-gpu.ipynb` (fine-tuning) |
| **E3** | SHAP faithfulness on a prespecified stratified sample (RQ4) | `05_rerun_shap.ipynb` |
| **E4** | Re-implemented SemEval-2025 systems (UoB-NLP-style adapter, Team UBD-style class-balanced training) on the same partitions (RQ3) | `06_rerun_paketD_gpu.ipynb` + `06b_rerun_paketD_adapter_v2_gpu.ipynb` |
| Check R1 | Reconciliation of the LR-only results with the first submission, cell by cell and seed by seed (Section 4.2) | `07_reconcile_lr_cpu.ipynb` |
| Check R2 | Decomposition of the change of the fine-tuning crossover: label budget vs. tuning/LoRA (Table 10) | `08_crossover_decomposition_gpu.ipynb` |

All results in the revised manuscript come from these notebooks. What changed between the first submission and the revision, and which results are newly trained or re-analysed, is listed in [`CHANGELOG_v1_to_v2.md`](CHANGELOG_v1_to_v2.md); every table is linked to its files in [`results/table_to_file.csv`](results/table_to_file.csv). Every frozen-pipeline configuration was fitted in **one** CPU session on the same ten seeds (42, 7, 13, 77, 101, 202, 303, 404, 505, 606).

> **Versions.** The first submission was produced by `notebooks/01_main_experiment.ipynb` and `02_ablation_lr_regularization.ipynb` (outputs in `results/main/`, `results/ablation_lr_regularization/`, `results/post_hoc/`). These files are kept so that both review rounds can be audited. They are superseded by the `rerun_*` folders; the revised manuscript does not use them; notebooks 07 and 08 read them from this repository to reconcile the two versions for the second review round (see `CHANGELOG_v1_to_v2.md`).

## Repository layout

```
notebooks/
  03-rerun-frozen-cpu.ipynb                E1 + frozen side of E2 (Kaggle CPU, ≈ 11.6 h)
  04-rerun-finetune-gpu.ipynb              fine-tuning side of E2: pilot search, full FT, LoRA (Kaggle T4)
  05_rerun_shap.ipynb                      E3, SHAP faithfulness (Kaggle CPU, ≈ 1 h)
  06_rerun_paketD_gpu.ipynb                E4, UoB-NLP-style adapter and Team UBD-style system (Kaggle T4)
  06b_rerun_paketD_adapter_v2_gpu.ipynb    E4, adapter with tuned-threshold early stopping (replaces the adapter of 06)
  07_reconcile_lr_cpu.ipynb                second round: LR-only reconciliation with the first submission (Kaggle CPU, ≈ 70 min)
  08_crossover_decomposition_gpu.ipynb     second round: original fine-tuning recipe in regimes A/B + decomposition (Kaggle T4, ≈ 7 h)
  01_main_experiment.ipynb                 first submission (superseded)
  02_ablation_lr_regularization.ipynb      first submission, separate-session ablation (superseded)
scripts/
  analyze_rerun.py                         E1 and E2 statistics → results/rerun_tables/E1_*, LC_*
  post_hoc_rerun.py                        per-class F1 and SemEval support/CIs → results/rerun_tables/T6_*, T7_*
  analyze_paketD.py                        E4 statistics → results/rerun_tables/paketD_*
  audit_test_access.py                     data-level check that no test text enters fitting or selection → results/audit_test_access.csv
  post_hoc_analysis.py, make_figures.py    first submission (superseded)
results/
  rerun_frozen/        outputs of notebook 03
  rerun_finetune/      outputs of notebook 04
  rerun_shap/          outputs of notebook 05
  rerun_paketD/        outputs of notebook 06
  rerun_paketD_v2/     outputs of notebook 06b
  rerun_tables/        tables computed by the three scripts above
  reconcile_lr/        outputs of notebook 07
  crossover_decomp/    outputs of notebook 08 (Table 10)
  table_to_file.csv    manuscript table → raw per-seed results, per-seed predictions, split records, selection records, test-label checks, scripts, SHA-256
  audit_test_access.csv
CHANGELOG_v1_to_v2.md  release history and version-to-version change log (v1 → v2 → r3)
  main/, ablation_lr_regularization/, post_hoc/   first submission (superseded)
figures/               Figs. 1–6 of the revised manuscript (`*_rerun.png`; Fig. 1 also as editable draw.io file)
requirements.txt
```

## Data

The datasets are **not** redistributed here; download them from their original sources:

- **SemEval-2025 Task 11** (Bridging the Gap in Text-Based Emotion Detection): Track A training and development files for Sundanese and the labeled Track C development files for Javanese and Indonesian. The official test partition is not used.
- **PRDECT-ID** (Indonesian product reviews from Tokopedia, 5 emotions, 5,400 texts).

Each language and the Tokopedia corpus are split per seed in two stages (65% training, then 15% validation and 20% test), stratified on the labels. The split of every text for every seed is recorded in `results/rerun_frozen/split_manifest_all.csv`.

On Kaggle, add both datasets with *Add Input*; the notebooks find the files under `/kaggle/input` automatically. To run elsewhere, set `MEXBAL_DATA` to the folder that holds the data and `MEXBAL_WORK` to an output folder. `MEXBAL_SMOKE=1` runs a short smoke test.

**Embedding cache (required for exact reproduction).** Notebooks 03, 05 and 07 read the twelve frozen-encoder matrices from the Kaggle dataset `embedding-cache-me-xbal` (path `/kaggle/input/datasets/rezkanorhafizah/embedding-cache-me-xbal/embedding_cache`); every revised frozen-pipeline result uses exactly these matrices. Their fingerprints are in `results/reconcile_lr/embedding_fingerprints.csv` (Tokopedia also in `results/rerun_frozen/environment_cpu.json`). Without the cache the notebooks re-encode the texts with the same code. With current library versions on CPU, the re-encoded E5 and XLM-R embeddings are close to the cache (minimum cosine 0.977 and 0.998), but the IndoBERT embeddings are not (minimum cosine 0.31; `results/reconcile_lr/embedding_fresh_vs_cache.csv`), so IndoBERT results can only be reproduced from the cache. The first submission (notebook 01) either read an attached cache or encoded on its own GPU session; this was not logged and its matrices were not exported. Its matrices were not bit-identical to the archived cache (`CHANGELOG_v1_to_v2.md` §3d), which explains the small residual differences of the reconciliation.

## How to reproduce

Run the notebooks on Kaggle with *Save & Run All (Commit)*, Internet on. Long loops write their CSV after every step and skip finished work, so an interrupted session can be resumed by attaching the partial output as an input.

| Step | Notebook / command | Hardware | Inputs besides the two datasets |
|---|---|---|---|
| 1 | `notebooks/03-rerun-frozen-cpu.ipynb` | Kaggle CPU (accelerator **None**) | embedding cache (optional) |
| 2 | `notebooks/04-rerun-finetune-gpu.ipynb` | Kaggle GPU T4 (two sessions) | none (optional: output of step 1, to verify that the nested subsets match `lc_subset_order.csv`) |
| 3 | `notebooks/05_rerun_shap.ipynb` | Kaggle CPU (accelerator **None**) | embedding cache of step 1 |
| 4 | `notebooks/06_rerun_paketD_gpu.ipynb`, then `06b_rerun_paketD_adapter_v2_gpu.ipynb` | Kaggle GPU T4 | output of step 1 (`split_manifest_all.csv`) |
| 5 | `notebooks/07_reconcile_lr_cpu.ipynb` | Kaggle CPU, Internet on | embedding cache; reference results are cloned from this repository at a fixed commit |
| 6 | `notebooks/08_crossover_decomposition_gpu.ipynb` | Kaggle GPU T4, Internet on | none (reference results cloned at a fixed commit) |
| 7 | scripts below | any CPU, seconds | `results/` |

```bash
pip install -r requirements.txt
python scripts/analyze_rerun.py  --frozen results/rerun_frozen --ft results/rerun_finetune --out results/rerun_tables
python scripts/post_hoc_rerun.py --frozen results/rerun_frozen --out results/rerun_tables
python scripts/analyze_paketD.py --frozen results/rerun_frozen --paketd results/rerun_paketD results/rerun_paketD_v2 --out results/rerun_tables
python scripts/audit_test_access.py --out results/audit_test_access.csv
```

`analyze_paketD.py` takes the Team UBD-style results from `rerun_paketD/` and the UoB-NLP-style adapter results from `rerun_paketD_v2/` (a later folder replaces the systems it contains). The adapter in `rerun_paketD/` stopped too early because it monitored Macro F1 at a fixed threshold of 0.5; notebook 06b monitors Macro F1 with tuned per-emotion thresholds and selects the learning rate on the validation partition.

Notebook 05 must run on the same kind of machine as notebook 03: it first refits the seed-42 models and stops if their scores differ from notebook 03 (`check_vs_nb03` in `results/rerun_shap/environment_shap.json`).

### Statistics

Every paired comparison over the ten seeds uses the Nadeau–Bengio corrected resampled t-test (ratio n_test/n_dev = 0.25), with 95% confidence intervals from the same corrected standard error, the two-sided Wilcoxon signed-rank test, and Holm adjustment within each family: the ten comparators of the main comparison on each dataset, the contrasts of each metric in the factorial analysis, the training sizes of each learning-curve comparison or decomposition cell, and the four re-implemented systems. For the learning curve, the conservative ratio 1081/n is reported as a sensitivity analysis.

## Where each manuscript table and figure comes from

| Manuscript | File(s) |
|---|---|
| Table 1 (related studies) | literature; no code |
| Table 2 (configuration) | `results/rerun_finetune/ft_step_budget.csv` (epoch limits and steps), `results/rerun_frozen/environment_cpu.json`, `results/rerun_finetune/environment_gpu.json`, `selected_configs.json`, `results/rerun_paketD_v2/environment_paketD.json`, `results/rerun_shap/environment_shap.json` |
| Table 3 (SemEval results) | `results/rerun_tables/E1_table_performance.csv` (raw: `results/rerun_frozen/E1_SemEval_raw.csv`) |
| Table 4 (SemEval paired comparisons) | `results/rerun_tables/E1_table_comparisons.csv` (metric = SemEval average / pooled) |
| Table 5 (Tokopedia results and comparisons) | `results/rerun_tables/E1_table_performance.csv`, `E1_table_comparisons.csv` (metric = Tokopedia) (raw: `results/rerun_frozen/E1_Tokopedia_raw.csv`) |
| Table 6 (per-class F1) | `results/rerun_tables/T6_per_class_f1.csv` |
| Table 7 (SemEval test support and corrected CIs) | `results/rerun_tables/T7_semeval_support_ci.csv` |
| Table 8 (component and factorial contrasts) | `results/rerun_tables/E1_table_factorial_contrasts.csv` |
| Section 4.2, session variability and sensitivity | `results/rerun_frozen/determinism_check.csv`; `results/rerun_tables/E1_table_sensitivity.csv`, `E1_table_selection_diagnostics.csv` (raw: `results/rerun_frozen/E1_SemEval_sensitivity_raw.csv`) |
| Section 4.2, reconciliation of the LR rows of Table 3 with the first submission | `results/reconcile_lr/D_table3_LR_decomposition.csv` (cell × seed: v1, v1 folds, v2, fold effect, residual), `D_table3_LR_reconciliation.csv`, `D_table3_LR_per_seed.csv`, `B_semeval_vs_reference.csv`, `C_tokopedia_vs_reference.csv` (text level); predictions: `results/reconcile_lr/predictions/`, `results/main/predictions.zip`, `results/rerun_frozen/predictions/` |
| Table 9 (learning curve) | `results/rerun_tables/LC_table_performance.csv`, `LC_table_comparisons.csv`, `LC_table_crossover_summary.csv`, `LC_table_finetune_diagnostics.csv` (raw: `results/rerun_frozen/LC_frozen_raw.csv`, `results/rerun_finetune/ft_runs.csv`) |
| Table 10 (label budget U/A/B × default vs. tuned fine-tuning; *default recipe* = first-submission recipe, *U* = first-submission E2b budget) | `results/crossover_decomp/decomp_cells.csv`, `decomp_effects.csv`, `decomp_crossover_summary.csv` (raw: `ft_original_runs.csv` + the Table 9 raw files; fidelity check: `fidelity_replay.csv`) |
| Table 11 (re-implemented SemEval-2025 systems) | `results/rerun_tables/paketD_performance.csv`, `paketD_comparisons.csv` |
| Table 12 (SHAP faithfulness) | `results/rerun_shap/faithfulness_summary.csv` (per text: `faithfulness_per_text.csv`) |
| Table 13 (notation) | no code |
| Fig. 1 (workflow) | `figures/fig1_pipeline.png`, `figures/fig1_pipeline.drawio` |
| Fig. 2 (learning curve) | `figures/fig2_learning_curve_rerun.png`; data: `results/rerun_tables/LC_table_performance.csv` |
| Figs. 3–5 (SHAP, Sundanese / Javanese / Indonesian) | `figures/fig3_shap_semeval_sun_rerun.png`, `fig4_shap_semeval_jav_rerun.png`, `fig5_shap_semeval_ind_rerun.png`; data: `results/rerun_shap/figures/SHAP_SemEval_{sun,jav,ind}.csv`, texts in `examples_manifest.csv` |
| Fig. 6 (SHAP, Tokopedia) | `figures/fig6_shap_tokopedia_rerun.png`; data: `results/rerun_shap/figures/SHAP_Tokopedia_ind.csv` |

## Audit files

| Item | File(s) |
|---|---|
| Split of every text for every seed | `results/rerun_frozen/split_manifest_all.csv` (identical to `results/main/split_manifest_all.csv` of the first submission and to `results/reconcile_lr/split_manifest_this_run.csv`) |
| Nested learning-curve subsets (subset *n* = first *n* ranks of a seed) | `results/rerun_frozen/lc_subset_order.csv` |
| Regime-A internal 80/20 splits of fine-tuning | `results/rerun_finetune/regimeA_inner_splits.csv` |
| Selected ensemble weights, per-emotion thresholds, class-prior coefficients α, threshold fallbacks, constant classifiers (every run) | `results/rerun_frozen/selected_params.csv` |
| Test labels, predictions, and probabilities of every run | `results/rerun_frozen/predictions/`, `results/rerun_finetune/predictions/`, `results/rerun_paketD*/predictions/` |
| Fine-tuning pilot search and selected configurations | `results/rerun_finetune/pilot_runs.csv`, `pilot_summary.csv`, `selected_configs.json` |
| Fine-tuning histories (every epoch, selected and stopping epoch) | `results/rerun_finetune/ft_history.csv`, `ft_runs.csv` |
| Fine-tuning epoch limits, planned and executed steps | `results/rerun_finetune/ft_step_budget.csv` |
| Adapter and Team UBD-style histories | `results/rerun_paketD/paketD_history.csv`, `results/rerun_paketD_v2/paketD_history.csv` |
| LoRA adapters of every run | `results/rerun_finetune/lora_adapters/` |
| Full fine-tuning checkpoints (fp16, seed 42, n = 100 / 1,000 / 3,510) | GitHub *Releases* (≈ 556 MB each, above the GitHub file limit; not in the git tree) |
| SHAP sample (IDs, strata, candidates), exclusion record, attributions | `results/rerun_shap/sample_manifest.csv`, `candidates_seed42.csv`, `exclusions.csv`, `shap_values.jsonl` |
| Random-ranking seeds and masking variants | `results/rerun_shap/environment_shap.json` (`sample_seed`, `n_random`, `replacements`) and `faithfulness_per_text.csv` |
| Determinism check (same session, thread count, earlier run) | `results/rerun_frozen/determinism_check.csv` |
| Reconciliation with the first submission (all seeds, both fold procedures, text-level) | `results/reconcile_lr/` |
| No test text in fitting or selection | `results/audit_test_access.csv` (`scripts/audit_test_access.py`); code-level description in `CHANGELOG_v1_to_v2.md` §5 |
| Original fine-tuning recipe: runs, histories, predictions | `results/crossover_decomp/ft_original_runs.csv`, `ft_original_history.csv`, `predictions/` |
| Environment | `results/*/environment_*.json`, `results/rerun_frozen/pip_freeze_cpu.txt`, `results/rerun_finetune/pip_freeze_gpu.txt` |

## Environment

- Frozen pipeline and SHAP (notebooks 03, 05): Kaggle CPU session, 4 cores, Intel Xeon @ 2.20 GHz; Python 3.13.15, PyTorch 2.11.0 (CPU), Transformers 5.16.1, scikit-learn 1.6.1, XGBoost 3.4.1, NumPy 2.1.3, pandas 2.3.3, SHAP 0.51.0.
- Fine-tuning and re-implemented systems (notebooks 04, 06, 06b): Kaggle Tesla T4; Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), Transformers 5.16.1, fp16 mixed precision. LoRA is implemented directly in PyTorch (no PEFT dependency).
- Encoders: `intfloat/multilingual-e5-base`, `xlm-roberta-base`, `indobenchmark/indobert-base-p1`.

## Reproducibility

- Repeating a configuration in the same session reproduces every prediction, and changing the XGBoost thread count changes none (`determinism_check.csv`).
- A new session (notebook 07) reproduces every prediction of the LR-only models of the revision (30/30 SemEval and 30/30 Tokopedia runs).
- Fine-tuning epochs: E = min(40, max(10, ⌈300/s⌉)) with s = ⌈m/16⌉ batches of the m fitting texts (no resampling); the planned schedule has ≥ 300 steps except at n = 100, where the 40-epoch cap gives 280 steps (regime B) and 200 steps on the 80-text early-stopping subset (regime A). Planned and executed steps of every run: `results/rerun_finetune/ft_step_budget.csv`; details in `CHANGELOG_v1_to_v2.md` §6.
- XGBoost trees can differ between processor types. XGBoost therefore uses 4 threads and inputs rounded to 1e-5, and all frozen-pipeline results come from one machine type.
- **Correction of an earlier statement.** The README of commit `2c99770` stated that the LR results on XLM-R and IndoBERT are identical to the first submission. That statement rested on rows 6–8 of `results/rerun_frozen/determinism_check.csv`, which compare Tokopedia seed 42 test Macro F1 with the first-submission run (`results/main/Tokopedia_final_raw.csv`, notebook 01, commit `30ba992`): XLM-R and IndoBERT matched, E5 did not, and SemEval was not checked. It does not hold in general and is withdrawn. The SemEval scores changed mainly because the inner folds are now stratified on the emotions and the language: the fold effect is −0.0071 (XLM-R) and −0.0028 (IndoBERT) of the total changes −0.0075 and −0.0023, and the residual is −0.0003 and +0.0005. Every cell and seed is listed in `results/reconcile_lr/D_table3_LR_decomposition.csv` and summarised in `CHANGELOG_v1_to_v2.md` §3.

## Label note

`ME-XBAL` was a working name used while the experiments were running. It remains only as an internal label in file, folder, and dataset names (for example `MEXBAL_DATA` or `embedding-cache-me-xbal`); the manuscript does not use it. In the revision results, the model names follow the manuscript, for example `E5 Hybrid (imbalance-aware)` in the CSV files and `E5_Hybrid_imbalance_aware` in prediction file names.
