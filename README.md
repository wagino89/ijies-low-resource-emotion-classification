# Frozen Multilingual Encoder or Fine-Tuning? — Code and Results

Code, exported results, and figures for the manuscript *"Frozen Multilingual Encoder or Fine-Tuning? A Controlled Empirical Study of Imbalance-Aware Emotion Classification in Low-Resource Languages with Faithfulness-Tested SHAP"*.

The study evaluates a frozen **multilingual-E5-base** encoder (prefix `query: `, masked mean pooling, L2 normalization) followed by StandardScaler + PCA (95% variance) and an **imbalance-aware hybrid head** (cost-sensitive XGBoost + class-weighted logistic regression, C = 0.1). The ensemble weight and the decision thresholds (multi-label) or class priors (single-label) are selected out of fold. The pipeline is compared with XLM-R, IndoBERT, and E5 baselines (LR / XGBoost / Hybrid) that use the same selection procedure, with fine-tuned multilingual-E5 in a controlled learning curve, and its SHAP explanations are tested for faithfulness.

## Label note

`ME-XBAL` was a working name used while the experiments were running. It is kept **only as an internal label** in the code and result files, so that these files stay identical to the run reported in the manuscript. The manuscript does not use this name:

| Label in code / result files | Name in the manuscript |
|---|---|
| `ME-XBAL`, `ME_XBAL` | E5 Hybrid (imbalance-aware, ours) |
| `ME-XBAL (XLM-R)`, `ME_XBAL_XLM_R_` | XLM-R Hybrid (imbalance-aware) |
| `ME-XBAL (train subset only)` (learning curve) | Frozen pipeline (equal data budget) |
| `ME-XBAL` in `learning_curve_ME_XBAL.csv` | Frozen pipeline (+validation data) |
| `/kaggle/working/ME-XBAL_E5` | output folder of the main notebook |

## Repository layout

```
notebooks/
  01_main_experiment.ipynb               E1/E3 main comparison, statistics, E2/E2b learning curve, SHAP
  02_ablation_lr_regularization.ipynb    ablation: E5 Hybrid with C = 0.1 and no class weights
scripts/
  post_hoc_analysis.py                   per-class F1, pooled SemEval Macro F1, test-set support + corrected CIs
  make_figures.py                        Figs. 2–6
results/
  main/                                  outputs of notebook 01 (raw CSVs, tables/, predictions.zip, SHAP/, learning curves)
  ablation_lr_regularization/            outputs of notebook 02
  post_hoc/                              outputs of scripts/post_hoc_analysis.py
figures/                                 Figs. 1–6 (Fig. 1 also as editable draw.io file)
requirements.txt
```

## Data

The datasets are **not** redistributed here; download them from their original sources:

- **SemEval-2025 Task 11** (Bridging the Gap in Text-Based Emotion Detection), Track A Sundanese and Track C Javanese and Indonesian. The public training and development data are pooled and repartitioned per seed (65/15/20, stratified); see `results/main/split_manifest_all.csv`.
- **PRDECT-ID** (Indonesian product reviews from Tokopedia, 5 emotions).

On Kaggle, add both datasets with *Add Input*; the notebooks locate the files under `/kaggle/input` automatically. To run elsewhere, set the environment variable `MEXBAL_DATA` to the folder that holds the data and `MEXBAL_WORK` to an output folder.

## How to reproduce

1. **Main experiment** – run `notebooks/01_main_experiment.ipynb` on Kaggle (GPU T4 or P100, Internet on) with *Save & Run All (Commit)*. Results are written to `/kaggle/working/ME-XBAL_E5` and correspond to `results/main/`. Long blocks write their CSV after every step and skip finished work, so an interrupted session can be resumed.
2. **Ablation** – run `notebooks/02_ablation_lr_regularization.ipynb` the same way (about 40 min). It re-runs the reference models in the same session, so its contrasts are computed within that run; its XGBoost-based scores differ slightly from the main run because it is a separate session. Outputs correspond to `results/ablation_lr_regularization/`.
3. **Post-hoc tables** – from the repository root:
   ```bash
   pip install -r requirements.txt
   python scripts/post_hoc_analysis.py   # writes results/post_hoc/
   python scripts/make_figures.py        # writes figures/
   ```
   These two scripts only read the exported results and run on a CPU in a few seconds.

Seeds: 42, 7, 13, 77, 101, 202, 303, 404, 505, 606 (main comparison); 42, 7, 13, 77, 101 (learning curve). Statistical tests: corrected resampled t-test (Nadeau–Bengio, test/train ratio 0.25) with 95% CIs, Holm adjustment, and Wilcoxon signed-rank tests.

## Where each manuscript table and figure comes from

| Manuscript | File(s) |
|---|---|
| Table 1 (related studies) | literature; no code |
| Table 2 (configuration) | `notebooks/01_main_experiment.ipynb` (Sec. 1, 5, 6), `results/main/environment.json`, `results/main/tables/table_selected_weights.csv` |
| Table 3 (SemEval results) | `results/main/tables/table_semeval.csv`, `table_semeval_per_language.csv`; *Pooled* column: `results/post_hoc/semeval_pooled_macro_f1.csv` |
| Table 4 (SemEval paired comparison) | `results/main/tables/table_semeval_comparison.csv` |
| Table 5 (Tokopedia) | `results/main/tables/table_tokopedia.csv`, `table_tokopedia_comparison.csv` |
| Table 6 (per-class F1) | `results/post_hoc/per_class_f1.csv` |
| Table 7 (test-set support, corrected CIs) | `results/post_hoc/semeval_support_ci.csv` |
| Table 8 (contrasts) | `results/main/tables/table_contrasts.csv`; rows marked ‡: `results/ablation_lr_regularization/table_ablationC.csv` |
| Table 9 (learning curve) | `results/main/tables/table_learning_curve.csv`; raw: `learning_curve_ME_XBAL_matched_budget.csv`, `learning_curve_ME_XBAL.csv`, `learning_curve_E5_finetuned.csv` |
| Table 10 (SHAP faithfulness) | `results/main/SHAP/SemEval_SHAP_faithfulness_summary.csv`, `Tokopedia_SHAP_faithfulness_summary.csv` (per text: `*_per_sample.csv`) |
| Table 11 (notation) | no code |
| Fig. 1 (workflow) | `figures/fig1_pipeline.png`, `figures/fig1_pipeline.drawio` |
| Fig. 2 (learning curve) | `figures/fig2_learning_curve.png` |
| Figs. 3–5 (SHAP, Sundanese / Javanese / Indonesian) | `figures/fig3_shap_semeval_sun.png`, `fig4_shap_semeval_jav.png`, `fig5_shap_semeval_ind.png`; data: `results/main/SHAP/SemEval_SHAP_{sun,jav,ind}.csv`, texts in `SemEval_SHAP_manifest.csv` |
| Fig. 6 (SHAP, Tokopedia) | `figures/fig6_shap_tokopedia.png`; data: `results/main/SHAP/Tokopedia_SHAP_example.csv` |

Other files in `results/main/`: `SemEval_final_raw.csv` and `Tokopedia_final_raw.csv` (one row per model × seed × language), `predictions.zip` (220 joblib files with the test-set labels and predictions per dataset, model, and seed; `scripts/post_hoc_analysis.py` extracts it to `predictions/` automatically), `tables/table_primary_seed_bootstrap.csv` (prediction-level bootstrap), and `split_manifest_all.csv` (split membership of every text).

## Environment

Recorded in `results/main/environment.json`: Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), Transformers 5.0.0, scikit-learn 1.6.1, XGBoost 3.2.0, NumPy 2.0.2, pandas 2.3.3, SHAP 0.51.0, GPU Tesla T4 (Kaggle). Encoders: `intfloat/multilingual-e5-base`, `xlm-roberta-base`, `indobenchmark/indobert-base-p1`.

Exact scores can differ slightly on other hardware or library builds (XGBoost and BLAS are not bit-identical across CPUs); the reported numbers are those in `results/`.
