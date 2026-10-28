"""Statistics and manuscript tables for the second-round rerun (notebooks 03 and 04).

Usage (from the repository root, CPU only, a few seconds):
    python scripts/analyze_rerun.py --frozen results/rerun_frozen --ft results/rerun_finetune --out results/rerun_tables

--ft is optional (E1 tables only). All paired comparisons use the Nadeau-Bengio corrected resampled
t-test over seeds (95% CI from the same standard error), the two-sided Wilcoxon signed-rank test, and
Holm adjustment within each stated family.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats

REF = "E5 Hybrid (imbalance-aware)"
ORIGINAL_11 = ["XLM-R LR", "XLM-R XGB", "XLM-R Hybrid", "IndoBERT LR", "IndoBERT XGB", "IndoBERT Hybrid",
               "E5 LR", "E5 XGB", "E5 Hybrid", "XLM-R Hybrid (imbalance-aware)", REF]
NEW_CONTROLS = ["E5 LR (C=0.1)", "E5 LR (weighted)", "E5 LR (imbalance-aware)", "E5 XGB (cost-sensitive)",
                "E5 Hybrid (C=0.1)", "E5 Hybrid (weighted)", "XLM-R LR (imbalance-aware)"]
RATIO_SEM = 0.20 / 0.80                   # test / development (each seed re-partitions every language 65/15/20)
N_TOK_TEST, N_TOK_DEV = 1081, 4319        # Tokopedia test / (train + validation)
RATIO_TOK = N_TOK_TEST / N_TOK_DEV


def corrected_ttest(d, ratio):
    d = np.asarray(d, float); J = len(d); m = d.mean()
    if J < 2: return m, np.nan, np.nan, np.nan
    se = np.sqrt((1.0 / J + ratio) * d.var(ddof=1))
    if se == 0: return m, m, m, 1.0
    t = m / se; h = stats.t.ppf(0.975, J - 1) * se
    return m, m - h, m + h, 2 * stats.t.sf(abs(t), J - 1)


def wilcoxon_p(d):
    d = np.asarray(d, float)
    return float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 1 else 1.0


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    q = p[ok]; o = np.argsort(q); adj = np.empty_like(q); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(q) - r) * q[i]); adj[i] = min(1.0, run)
    out[ok] = adj; return out


def paired(a, b, ratio):
    """a, b: Series indexed by seed. Returns dict of paired statistics on a - b."""
    j = a.index.intersection(b.index); d = (a[j] - b[j]).values
    m, lo, hi, p = corrected_ttest(d, ratio)
    return dict(delta=m, ci95_low=lo, ci95_high=hi, p_corrected=p, p_wilcoxon=wilcoxon_p(d),
                wins=int((d > 0).sum()), n_seeds=len(d))


ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


def e1_tables(fz, out):
    sem = pd.read_csv(f"{fz}/E1_SemEval_raw.csv"); tok = pd.read_csv(f"{fz}/E1_Tokopedia_raw.csv")
    models = [m for m in ORIGINAL_11 + NEW_CONTROLS if m in set(sem.model)]
    piv = {l: sem[sem.lang == l].pivot(index="seed", columns="model", values="macro_f1") for l in ["sun", "jav", "ind", "average", "pooled"]}
    tpiv = tok.pivot(index="seed", columns="model", values="macro_f1")

    t = pd.DataFrame([{"model": m, **{l: ms(piv[l][m]) for l in piv}, "tokopedia_macro_f1": ms(tpiv[m]),
                       "tokopedia_micro_f1": tok[tok.model == m].micro_f1.mean()} for m in models])
    t.to_csv(f"{out}/E1_table_performance.csv", index=False)

    # comparisons with the reference model: family 1 = the 10 original comparators, family 2 = the new controls
    rows = []
    for fam, members in [("original comparators", ORIGINAL_11), ("new controls", NEW_CONTROLS)]:
        for metric, P, ratio in [("SemEval average", piv["average"], RATIO_SEM), ("SemEval pooled", piv["pooled"], RATIO_SEM),
                                 ("SemEval sun", piv["sun"], RATIO_SEM), ("SemEval jav", piv["jav"], RATIO_SEM),
                                 ("SemEval ind", piv["ind"], RATIO_SEM), ("Tokopedia", tpiv, RATIO_TOK)]:
            block = [dict(family=fam, metric=metric, comparison=f"{REF} vs {m}", **paired(P[REF], P[m], ratio))
                     for m in members if m != REF and m in P]
            if not block: continue
            b = pd.DataFrame(block); b["p_holm"] = holm(b.p_corrected); rows.append(b)
    pd.concat(rows).to_csv(f"{out}/E1_table_comparisons.csv", index=False)

    # one-factor contrasts and the weighting x regularisation factorials (R2-3, R1-3)
    C = [("Whole imbalance-aware head, hybrid", REF, "E5 Hybrid"),
         ("Weighting at C=1, hybrid", "E5 Hybrid (weighted)", "E5 Hybrid"),
         ("Weighting at C=0.1, hybrid", REF, "E5 Hybrid (C=0.1)"),
         ("C=0.1 without weighting, hybrid", "E5 Hybrid (C=0.1)", "E5 Hybrid"),
         ("C=0.1 with weighting, hybrid", REF, "E5 Hybrid (weighted)"),
         ("Whole imbalance-aware head, LR only", "E5 LR (imbalance-aware)", "E5 LR"),
         ("Weighting at C=1, LR only", "E5 LR (weighted)", "E5 LR"),
         ("Weighting at C=0.1, LR only", "E5 LR (imbalance-aware)", "E5 LR (C=0.1)"),
         ("C=0.1 without weighting, LR only", "E5 LR (C=0.1)", "E5 LR"),
         ("C=0.1 with weighting, LR only", "E5 LR (imbalance-aware)", "E5 LR (weighted)"),
         ("Boosting branch, imbalance-aware (hybrid IA - LR IA)", REF, "E5 LR (imbalance-aware)"),
         ("Boosting branch, plain (hybrid - LR)", "E5 Hybrid", "E5 LR"),
         ("Cost-sensitive XGBoost (XGB CS - XGB)", "E5 XGB (cost-sensitive)", "E5 XGB"),
         ("Boosting branch, XLM-R imbalance-aware", "XLM-R Hybrid (imbalance-aware)", "XLM-R LR (imbalance-aware)"),
         ("Whole imbalance-aware head, XLM-R hybrid", "XLM-R Hybrid (imbalance-aware)", "XLM-R Hybrid"),
         ("Encoder E5 vs XLM-R, imbalance-aware hybrid", REF, "XLM-R Hybrid (imbalance-aware)")]
    INTER = [("Interaction weighting x C, hybrid", REF, "E5 Hybrid (C=0.1)", "E5 Hybrid (weighted)", "E5 Hybrid"),
             ("Interaction weighting x C, LR only", "E5 LR (imbalance-aware)", "E5 LR (C=0.1)", "E5 LR (weighted)", "E5 LR")]
    rows = []
    for metric, P, ratio in [("SemEval average", piv["average"], RATIO_SEM), ("SemEval pooled", piv["pooled"], RATIO_SEM),
                             ("Tokopedia", tpiv, RATIO_TOK)]:
        block = [dict(metric=metric, contrast=lab, a=a, b=b, **paired(P[a], P[b], ratio))
                 for lab, a, b in C if a in P and b in P]
        for lab, w1c1, w0c1, w1c0, w0c0 in INTER:      # (IA - C0.1 only) - (weighted - plain)
            if all(x in P for x in [w1c1, w0c1, w1c0, w0c0]):
                d = (P[w1c1] - P[w0c1]) - (P[w1c0] - P[w0c0])
                block.append(dict(metric=metric, contrast=lab, a="(w,C0.1)-(C0.1)", b="(w,C1)-(C1)",
                                  **paired(d, d * 0, ratio)))
        b = pd.DataFrame(block); b["p_holm"] = holm(b.p_corrected); rows.append(b)
    pd.concat(rows).to_csv(f"{out}/E1_table_factorial_contrasts.csv", index=False)

    # procedure sensitivity (R1-4)
    sp = f"{fz}/E1_SemEval_sensitivity_raw.csv"
    if os.path.exists(sp):
        s = pd.read_csv(sp); rows = []
        for (v, m), g in s.groupby(["variant", "model"]):
            for lang in ["average", "pooled", "sun", "jav", "ind"]:
                a = g[g.lang == lang].set_index("seed").macro_f1; b = piv[lang][m]
                rows.append(dict(variant=v, model=m, metric=lang, variant_mean=a.mean(), primary_mean=b.mean(),
                                 **paired(a, b, RATIO_SEM)))
        pd.DataFrame(rows).to_csv(f"{out}/E1_table_sensitivity.csv", index=False)

    # selection diagnostics (R1-4): fallbacks and constant classifiers
    pp = f"{fz}/selected_params.csv"
    if os.path.exists(pp):
        p = pd.read_csv(pp)
        d = (p[p.dataset == "SemEval"].groupby(["variant", "model"])
             .agg(runs=("seed", "size"), mean_weight_xgb=("weight_xgb", "mean"),
                  runs_with_threshold_fallback=("threshold_fallbacks", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
                  total_threshold_fallbacks=("threshold_fallbacks", lambda x: int(pd.to_numeric(x, errors="coerce").sum())),
                  total_constant_fits=("constant_fits", lambda x: int(pd.to_numeric(x, errors="coerce").sum()))).reset_index())
        d.to_csv(f"{out}/E1_table_selection_diagnostics.csv", index=False)
    print("E1 tables written")


def lc_tables(fz, ft, out):
    lf = pd.read_csv(f"{fz}/LC_frozen_raw.csv")
    fr = pd.read_csv(f"{ft}/ft_runs.csv") if ft and os.path.exists(f"{ft}/ft_runs.csv") else pd.DataFrame()
    pairs = [("A: label-matched", "A_label_matched", "A_label_matched", "FT (tuned)"),
             ("B: selection-matched", "B_selection_matched", "B_selection_matched", "FT (tuned)"),
             ("B: selection-matched", "B_selection_matched", "B_selection_matched", "LoRA (tuned)"),
             ("C: frozen fits on n + validation (supplementary)", "C_fit_with_val", "B_selection_matched", "FT (tuned)")]
    rows = []
    for label, freg, treg, comp in pairs:
        if fr.empty: break
        F_ = fr[(fr.regime == treg) & (fr.comparator == comp)]
        for model in lf.model.unique():
            Z = lf[(lf.regime == freg) & (lf.model == model)]
            block = []
            for n in sorted(set(Z.train_size) & set(F_.train_size)):
                a = Z[Z.train_size == n].set_index("seed").macro_f1; b = F_[F_.train_size == n].set_index("seed").macro_f1
                prim = paired(a, b, RATIO_TOK); cons = paired(a, b, N_TOK_TEST / n)
                block.append(dict(regime=label, frozen_model=model, comparator=comp, train_size=n,
                                  frozen_mean=a.mean(), frozen_sd=a.std(ddof=1), ft_mean=b[a.index.intersection(b.index)].mean(),
                                  ft_sd=b.std(ddof=1), delta=prim["delta"], wins_frozen=prim["wins"], n_seeds=prim["n_seeds"],
                                  ci95_low=prim["ci95_low"], ci95_high=prim["ci95_high"], p_corrected=prim["p_corrected"],
                                  ci95_low_conservative=cons["ci95_low"], ci95_high_conservative=cons["ci95_high"],
                                  p_corrected_conservative=cons["p_corrected"], p_wilcoxon=prim["p_wilcoxon"]))
            if not block: continue
            b = pd.DataFrame(block)
            b["p_holm"] = holm(b.p_corrected); b["p_holm_conservative"] = holm(b.p_corrected_conservative)
            rows.append(b)
    if rows:
        t = pd.concat(rows); t.to_csv(f"{out}/LC_table_comparisons.csv", index=False)
        summ = []
        for (lab, m, c), g in t.groupby(["regime", "frozen_model", "comparator"], sort=False):
            for col in ["p_holm", "p_holm_conservative"]:
                fz_better = g[(g[col] < 0.05) & (g.delta > 0)].train_size; ft_better = g[(g[col] < 0.05) & (g.delta < 0)].train_size
                summ.append(dict(regime=lab, frozen_model=m, comparator=c, test=col,
                                 frozen_significantly_better_at=",".join(map(str, fz_better)) or "-",
                                 ft_significantly_better_at=",".join(map(str, ft_better)) or "-",
                                 crossover_between=(f"{fz_better.max() if len(fz_better) else 'none'} and "
                                                    f"{ft_better.min() if len(ft_better) else 'none'}")))
        pd.DataFrame(summ).to_csv(f"{out}/LC_table_crossover_summary.csv", index=False)
    perf = lf.groupby(["regime", "model", "train_size"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
    if not fr.empty:
        perf = pd.concat([perf, fr.groupby(["regime", "comparator", "train_size"]).macro_f1.agg(["mean", "std", "count"])
                          .reset_index().rename(columns={"comparator": "model"})])
        diag = (fr.groupby(["regime", "comparator", "train_size"])
                .agg(seeds=("seed", "nunique"), mean_best_epoch=("best_epoch", "mean"), mean_max_epochs=("max_epochs", "mean"),
                     share_stopped_at_cap=("stopped_at_cap", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                     mean_pred_classes=("n_pred_classes", "mean"), mean_train_time_s=("train_time_s", "mean")).reset_index())
        diag.to_csv(f"{out}/LC_table_finetune_diagnostics.csv", index=False)
    perf.to_csv(f"{out}/LC_table_performance.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, (title, freg, treg) in zip(axes, [("Regime A: label-matched", "A_label_matched", "A_label_matched"),
                                                  ("Regime B: selection-matched", "B_selection_matched", "B_selection_matched")]):
            for (m, g) in lf[lf.regime == freg].groupby("model"):
                s = g.groupby("train_size").macro_f1.agg(["mean", "std"]); ax.errorbar(s.index, s["mean"], s["std"], marker="o", capsize=3, label=f"Frozen: {m}")
            if not fr.empty:
                for (c, g) in fr[fr.regime == treg].groupby("comparator"):
                    s = g.groupby("train_size").macro_f1.agg(["mean", "std"]); ax.errorbar(s.index, s["mean"], s["std"], marker="s", ls="--", capsize=3, label=c)
            ax.set_title(title); ax.set_xlabel("Training texts"); ax.grid(alpha=.25)
        axes[0].set_ylabel("Macro F1"); axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/LC_figure.png", dpi=300)
    except Exception as e:
        print("figure skipped:", e)
    print("Learning-curve tables written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True); ap.add_argument("--ft", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    e1_tables(a.frozen, a.out)
    if os.path.exists(f"{a.frozen}/LC_frozen_raw.csv"): lc_tables(a.frozen, a.ft, a.out)
    det = f"{a.frozen}/determinism_check.csv"
    if os.path.exists(det): print(pd.read_csv(det).drop(columns=["cpu"], errors="ignore").to_string(index=False))
