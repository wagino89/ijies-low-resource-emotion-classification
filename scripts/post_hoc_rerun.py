"""Tables 6-7 for the rerun (per-class F1; SemEval support and corrected CIs) from notebook 03 predictions.
Usage: python scripts/post_hoc_rerun.py --frozen results/rerun_frozen --out results/rerun_tables"""
import argparse, joblib, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser(); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
SEEDS = [42, 7, 13, 77, 101, 202, 303, 404, 505, 606]; LANGS = ["sun", "jav", "ind"]
EMO = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]; TOKO = ["Anger", "Fear", "Happy", "Love", "Sadness"]
OURS, BASE, LRIA = "E5_Hybrid_imbalance_aware", "E5_Hybrid", "E5_LR_imbalance_aware"
def ct(d, ratio):
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1 / J + ratio) * d.var(ddof=1)); tc = stats.t.ppf(.975, J - 1)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, 2 * stats.t.sf(abs(d.mean() / se), J - 1)
L = lambda ds, m, s: joblib.load(f"{a.frozen}/predictions/{ds}_{m}_seed{s}.joblib")
rows = []
for m in [BASE, OURS, LRIA]:
    for s in SEEDS:
        x = L("SemEval", m, s); yt = np.vstack([x[l]["y_true"] for l in LANGS]); yp = np.vstack([x[l]["y_pred"] for l in LANGS])
        for k, e in enumerate(EMO): rows.append(dict(dataset="SemEval", cls=e, model=m, seed=s, f1=f1_score(yt[:, k], yp[:, k], zero_division=0), support=int(yt[:, k].sum())))
        b = L("Tokopedia", m, s); f = f1_score(b["y_true"], b["y_pred"], average=None, labels=range(5), zero_division=0)
        for k, c in enumerate(TOKO): rows.append(dict(dataset="Tokopedia", cls=c, model=m, seed=s, f1=f[k], support=int((b["y_true"] == k).sum())))
pc = pd.DataFrame(rows); w = pc.pivot_table(index=["dataset", "cls", "seed"], columns="model", values="f1").reset_index()
w["d"] = w[OURS] - w[BASE]
t6 = w.groupby(["dataset", "cls"], sort=False).agg(e5_hybrid=(BASE, "mean"), imbalance_aware=(OURS, "mean"), lr_imbalance_aware=(LRIA, "mean"),
                                                   delta=("d", "mean"), wins=("d", lambda x: int((x > 0).sum()))).reset_index()
t6["support_mean"] = pc[pc.model == OURS].groupby(["dataset", "cls"], sort=False).support.mean().values
t6.round(4).to_csv(f"{a.out}/T6_per_class_f1.csv", index=False)
raw = pd.read_csv(f"{a.frozen}/E1_SemEval_raw.csv"); R = "E5 Hybrid (imbalance-aware)"
rows = []
for l, name in zip(LANGS + ["average", "pooled"], ["Sundanese", "Javanese", "Indonesian", "Language average", "Pooled (all texts)"]):
    me = raw[(raw.model == R) & (raw.lang == l)].set_index("seed").macro_f1.loc[SEEDS]
    e5 = raw[(raw.model == "E5 Hybrid") & (raw.lang == l)].set_index("seed").macro_f1.loc[SEEDS]
    if l in LANGS:
        sup = np.mean([L("SemEval", OURS, s)[l]["y_true"].sum(0) for s in SEEDS], 0); n = [len(L("SemEval", OURS, s)[l]["y_true"]) for s in SEEDS]
    elif l == "pooled":
        sup = np.mean([np.vstack([L("SemEval", OURS, s)[q]["y_true"] for q in LANGS]).sum(0) for s in SEEDS], 0)
        n = [sum(len(L("SemEval", OURS, s)[q]["y_true"]) for q in LANGS) for s in SEEDS]
    else: sup, n = None, None
    m, lo, hi, _ = ct(me, .25); dm, dlo, dhi, p = ct(me - e5, .25)
    rows.append(dict(test_set=name, n_min=min(n) if n else "", n_max=max(n) if n else "", positives="/".join(f"{x:.1f}" for x in sup) if sup is not None else "",
                     macro_f1=m, ci_low=lo, ci_high=hi, delta=dm, delta_ci_low=dlo, delta_ci_high=dhi, p=p, wins=int(((me - e5) > 0).sum())))
pd.DataFrame(rows).round(4).to_csv(f"{a.out}/T7_semeval_support_ci.csv", index=False); print("ok")
