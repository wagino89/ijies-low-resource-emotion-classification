"""Paket D (R2-2): re-implemented SemEval-2025 systems vs. the frozen pipeline on the same seeds and splits.
Usage: python scripts/analyze_paketD.py --frozen results/rerun_frozen --paketd results/rerun_paketD --out results/rerun_tables"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analyze_rerun import paired, holm, RATIO_SEM
ap = argparse.ArgumentParser(); ap.add_argument("--frozen", required=True); ap.add_argument("--paketd", required=True, nargs="+", help="one or more output folders; a later folder replaces the systems it contains"); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
fz = pd.read_csv(f"{a.frozen}/E1_SemEval_raw.csv"); pdD = pd.DataFrame()
for d in a.paketd:
    new = pd.read_csv(f"{d}/paketD_SemEval_raw.csv")
    pdD = pd.concat([pdD[~pdD.system.isin(new.system.unique())] if len(pdD) else pdD, new], ignore_index=True)
pdD["model"] = pdD.system + " (" + pdD.encoder.map({"xlmr": "XLM-R", "e5": "E5"}) + ")"
REFS = ["E5 Hybrid (imbalance-aware)", "E5 LR (imbalance-aware)"]
both = pd.concat([fz[fz.model.isin(REFS)][["model", "seed", "lang", "macro_f1"]], pdD[["model", "seed", "lang", "macro_f1"]]])
perf = both.groupby(["model", "lang"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
perf.to_csv(f"{a.out}/paketD_performance.csv", index=False)
rows = []
for ref in REFS:
    for lang in ["average", "pooled", "sun", "jav", "ind"]:
        P = both[both.lang == lang].pivot(index="seed", columns="model", values="macro_f1")
        blk = [dict(reference=ref, metric=lang, comparator=m, **paired(P[[ref, m]].dropna()[ref], P[[ref, m]].dropna()[m], RATIO_SEM)) for m in sorted(pdD.model.unique())]
        b = pd.DataFrame(blk); b["p_holm"] = holm(b.p_corrected); rows.append(b)
t = pd.concat(rows); t.to_csv(f"{a.out}/paketD_comparisons.csv", index=False)
pd.set_option("display.width", 250)
print(perf.pivot(index="model", columns="lang", values="mean")[["sun", "jav", "ind", "average", "pooled"]].round(4))
print(t[t.metric.isin(["average", "pooled"])][["reference", "metric", "comparator", "delta", "ci95_low", "ci95_high", "p_corrected", "p_holm", "wins", "n_seeds"]].round(4).to_string(index=False))
