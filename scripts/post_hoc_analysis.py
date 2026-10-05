"""Post-hoc analyses reported in the manuscript, computed from the exported per-seed predictions.

Outputs (written to results/post_hoc/):
  per_class_f1.csv          Table 6: per-class F1 of E5 Hybrid vs. E5 Hybrid (imbalance-aware), 10 seeds
  semeval_pooled_macro_f1.csv  Table 3 "Pooled" column: Macro F1 over all SemEval test texts of a seed
  semeval_support_ci.csv    Table 7: test-set support, Macro F1 and Δ vs. E5 Hybrid with corrected 95% CIs

Run from the repository root:  python scripts/post_hoc_analysis.py
Note: in the result files the imbalance-aware E5 hybrid is labelled "ME-XBAL" and the
imbalance-aware XLM-R hybrid "ME-XBAL (XLM-R)" (working names used during the experiments).
"""
from pathlib import Path
import glob, re
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "main"
PRED = RES / "predictions"
if not PRED.is_dir():                      # predictions are shipped as one zip to keep the repository small
    import zipfile
    with zipfile.ZipFile(RES / "predictions.zip") as z: z.extractall(RES)
OUT = ROOT / "results" / "post_hoc"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 77, 101, 202, 303, 404, 505, 606]
LANGS = ["sun", "jav", "ind"]
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
TOKO = ["Anger", "Fear", "Happy", "Love", "Sadness"]
RATIO = 0.25                      # n_test / n_dev used by the corrected resampled t-test
OURS, BASE = "ME_XBAL", "E5_Hybrid"


def corrected_t(d, ratio=RATIO):
    d = np.asarray(d, float); J = len(d)
    se = np.sqrt((1 / J + ratio) * d.var(ddof=1)); tc = stats.t.ppf(0.975, J - 1)
    p = 2 * stats.t.sf(abs(d.mean() / se), J - 1) if se > 0 else 0.0
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, p


def load(ds, model, seed):
    return joblib.load(PRED / f"{ds}_{model}_seed{seed}.joblib")


# ---- Table 6: per-class F1 -----------------------------------------------------------
rows = []
for model in [BASE, OURS]:
    for s in SEEDS:
        a = load("SemEval", model, s)
        yt = np.vstack([a[l]["y_true"] for l in LANGS]); yp = np.vstack([a[l]["y_pred"] for l in LANGS])
        for k, e in enumerate(EMOTIONS):
            rows.append(dict(dataset="SemEval", cls=e, model=model, seed=s,
                             f1=f1_score(yt[:, k], yp[:, k], zero_division=0), support=int(yt[:, k].sum())))
        b = load("Tokopedia", model, s)
        f1 = f1_score(b["y_true"], b["y_pred"], average=None, labels=range(len(TOKO)), zero_division=0)
        for k, c in enumerate(TOKO):
            rows.append(dict(dataset="Tokopedia", cls=c, model=model, seed=s, f1=f1[k], support=int((b["y_true"] == k).sum())))
pc = pd.DataFrame(rows)
wide = pc.pivot_table(index=["dataset", "cls", "seed"], columns="model", values="f1").reset_index()
wide["delta"] = wide[OURS] - wide[BASE]
tab6 = wide.groupby(["dataset", "cls"]).agg(e5_hybrid=(BASE, "mean"), imbalance_aware=(OURS, "mean"),
                                           delta=("delta", "mean"), wins=("delta", lambda x: int((x > 0).sum()))).reset_index()
tab6["support_mean"] = pc[pc.model == OURS].groupby(["dataset", "cls"]).support.mean().values
tab6.round(4).to_csv(OUT / "per_class_f1.csv", index=False)

# ---- Table 3 pooled column ---------------------------------------------------------------
models = sorted({re.match(r"SemEval_(.+)_seed\d+\.joblib", Path(f).name).group(1) for f in glob.glob(str(PRED / "SemEval_*.joblib"))})
pooled = {}
for m in models:
    v = []
    for s in SEEDS:
        a = load("SemEval", m, s)
        yt = np.vstack([a[l]["y_true"] for l in LANGS]); yp = np.vstack([a[l]["y_pred"] for l in LANGS])
        v.append(f1_score(yt, yp, average="macro", zero_division=0))
    pooled[m] = v
pool = pd.DataFrame(pooled, index=SEEDS); pool.index.name = "seed"
pool.round(6).to_csv(OUT / "semeval_pooled_macro_f1.csv")

# ---- Table 7: support and corrected CIs ----------------------------------------------------
raw = pd.read_csv(RES / "SemEval_final_raw.csv")
rows = []
for l, name in zip(LANGS, ["Sundanese", "Javanese", "Indonesian"]):
    me = raw[(raw.model == "ME-XBAL") & (raw.lang == l)].set_index("seed").macro_f1.loc[SEEDS]
    e5 = raw[(raw.model == "E5 Hybrid") & (raw.lang == l)].set_index("seed").macro_f1.loc[SEEDS]
    sup = np.mean([load("SemEval", OURS, s)[l]["y_true"].sum(0) for s in SEEDS], 0)
    n = [len(load("SemEval", OURS, s)[l]["y_true"]) for s in SEEDS]
    m, lo, hi, _ = corrected_t(me); dm, dlo, dhi, p = corrected_t(me - e5)
    rows.append(dict(test_set=name, n_min=min(n), n_max=max(n), positives="/".join(f"{x:.1f}" for x in sup),
                     macro_f1=m, ci_low=lo, ci_high=hi, delta=dm, delta_ci_low=dlo, delta_ci_high=dhi, p=p, wins=int(((me - e5) > 0).sum())))
avg = raw.groupby(["model", "seed"]).macro_f1.mean().unstack(0).loc[SEEDS]
m, lo, hi, _ = corrected_t(avg["ME-XBAL"]); dm, dlo, dhi, p = corrected_t(avg["ME-XBAL"] - avg["E5 Hybrid"])
rows.append(dict(test_set="Language average", macro_f1=m, ci_low=lo, ci_high=hi, delta=dm, delta_ci_low=dlo, delta_ci_high=dhi, p=p,
                 wins=int(((avg["ME-XBAL"] - avg["E5 Hybrid"]) > 0).sum())))
n = [sum(len(load("SemEval", OURS, s)[l]["y_true"]) for l in LANGS) for s in SEEDS]
sup = np.mean([np.vstack([load("SemEval", OURS, s)[l]["y_true"] for l in LANGS]).sum(0) for s in SEEDS], 0)
m, lo, hi, _ = corrected_t(pool[OURS]); dm, dlo, dhi, p = corrected_t(pool[OURS] - pool[BASE])
rows.append(dict(test_set="Pooled (all texts)", n_min=min(n), n_max=max(n), positives="/".join(f"{x:.1f}" for x in sup),
                 macro_f1=m, ci_low=lo, ci_high=hi, delta=dm, delta_ci_low=dlo, delta_ci_high=dhi, p=p, wins=int(((pool[OURS] - pool[BASE]) > 0).sum())))
pd.DataFrame(rows).round(4).to_csv(OUT / "semeval_support_ci.csv", index=False)
print("written to", OUT)
