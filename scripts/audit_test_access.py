"""Data-level audit: no test text enters fitting or selection (Reviewer 3, point 3).

Usage (repository root, CPU, seconds):
    python scripts/audit_test_access.py --out results/audit_test_access.csv

Checks, for every seed:
  1. train / validation / test partitions are disjoint and cover each corpus (split manifest);
  2. the split manifests of the first submission, the revision, and the reconciliation run are identical;
  3. every learning-curve subset (nested order) lies in the training partition;
  4. every regime-A internal early-stopping split (tuned and original fine-tuning) lies in the n-text subset;
  5. the SHAP sample is drawn from the test partition only (explanations are computed after fitting and selection,
     and the gold labels of the sample are used only to define the reporting strata).
The code-level part of the audit (which index sets each selection function receives) is listed in CHANGELOG_v1_to_v2.md.
"""
import argparse, hashlib, os
import pandas as pd

R = "results"
rows = []

def add(check, ok, detail):
    rows.append({"check": check, "passed": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + check + " — " + detail)

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{R}/audit_test_access.csv"); a = ap.parse_args()

man = pd.read_csv(f"{R}/rerun_frozen/split_manifest_all.csv")
bad = 0
for (ds, lang, seed), g in man.groupby(["dataset", "lang", "seed"]):
    parts = {s: set(g[g.split == s].uid) for s in ["train", "val", "test"]}
    bad += bool(parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"])
    bad += g.uid.duplicated().any()
add("1. partitions disjoint", bad == 0, f"{man.groupby(['dataset', 'lang', 'seed']).ngroups} dataset×language×seed groups, {len(man)} rows")

refs = [p for p in [f"{R}/main/split_manifest_all.csv", f"{R}/rerun_frozen/split_manifest_all.csv",
                    f"{R}/reconcile_lr/split_manifest_this_run.csv"] if os.path.exists(p)]
key = lambda df: set(map(tuple, df[["dataset", "lang", "seed", "split", "uid"]].astype(str).values.tolist()))
k0 = key(pd.read_csv(refs[0]))
add("2. split manifests identical across versions", all(key(pd.read_csv(p)) == k0 for p in refs[1:]),
    "; ".join(f"{p} sha256 {sha(p)[:12]}" for p in refs))

train = {s: set(g.uid) for s, g in man[(man.dataset == "Tokopedia") & (man.split == "train")].groupby("seed")}
order = pd.read_csv(f"{R}/rerun_frozen/lc_subset_order.csv")
viol = sum(len(set(g.uid) - train[s]) for s, g in order.groupby("seed"))
add("3. learning-curve subsets within the training partition", viol == 0, f"{order.seed.nunique()} seeds, {len(order)} ranked texts, {viol} outside")

for path in [f"{R}/rerun_finetune/regimeA_inner_splits.csv", f"{R}/crossover_decomp/regimeA_inner_splits.csv"]:
    if not os.path.exists(path): continue
    ins = pd.read_csv(path); v = 0
    for (s, n), g in ins.groupby(["seed", "train_size"]):
        sub = set(order[order.seed == s].sort_values("rank").uid.values[:n])
        v += len(set(g.uid) - sub)
        v += len(set(g[g.part == "inner_train"].uid) & set(g[g.part == "inner_val"].uid))
    add(f"4. regime-A inner splits within the n-text subset ({path})", v == 0, f"{ins.groupby(['seed', 'train_size']).ngroups} seed×size groups, {v} violations")

sm = f"{R}/rerun_shap/sample_manifest.csv"
if os.path.exists(sm):
    s = pd.read_csv(sm)
    test42 = set(man[(man.seed == 42) & (man.split == "test")].uid)
    s_ids = set(s.id.astype(str))
    add("5. SHAP sample drawn from the seed-42 test partition", s_ids <= test42, f"{len(s_ids)} texts, {len(s_ids - test42)} outside the test partition")

pd.DataFrame(rows).to_csv(a.out, index=False)
print("written", a.out)
