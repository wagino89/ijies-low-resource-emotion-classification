"""Regenerate Figs. 2-6 of the manuscript from the exported results.

  Fig. 2  controlled learning curve on Tokopedia (mean +/- 1 SD over 5 seeds)
  Fig. 3  SHAP token attributions, Sundanese SemEval example
  Fig. 4  SHAP token attributions, Javanese SemEval example
  Fig. 5  SHAP token attributions, Indonesian SemEval example
  Fig. 6  SHAP token attributions, Tokopedia example

Fig. 1 (pipeline diagram) is a draw.io drawing: figures/fig1_pipeline.drawio.
Run from the repository root:  python scripts/make_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "main"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "savefig.dpi": 300})
BLUE, RED, GREY = "#1f4e79", "#b22222", "grey"

# ---- Fig. 2: learning curve --------------------------------------------------------------
series = [("learning_curve_ME_XBAL_matched_budget.csv", "Frozen pipeline (equal data budget)", BLUE, "-", "o"),
          ("learning_curve_ME_XBAL.csv", "Frozen pipeline (+validation data)", GREY, ":", "s"),
          ("learning_curve_E5_finetuned.csv", "Fine-tuned multilingual-E5", RED, "--", "^")]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for f, lab, col, ls, mk in series:
    g = pd.read_csv(RES / f).groupby("train_size").macro_f1.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], label=lab, color=col, ls=ls, marker=mk, ms=5, capsize=3, lw=1.5)
ax.set_xticks([100, 500, 1000, 1500, 2000, 2500, 3000, 3510])
ax.set_xlabel("Number of training instances"); ax.set_ylabel("Macro F1")
ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig2_learning_curve.png"); plt.close(fig)


# ---- Figs. 3-6: SHAP bar charts ----------------------------------------------------------
def shap_bar(csv, png, top=15):
    d = pd.read_csv(csv)
    d["token"] = d.token.astype(str).str.strip()
    d = d.reindex(d.shap_value.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(d) + 1.0))
    ax.barh(d.token, d.shap_value, color=[BLUE if v > 0 else RED for v in d.shap_value], height=0.65)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (contribution to predicted probability)"); ax.set_ylabel("Token")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / png); plt.close(fig)


for i, lang in enumerate(["sun", "jav", "ind"], start=3):
    shap_bar(RES / "SHAP" / f"SemEval_SHAP_{lang}.csv", f"fig{i}_shap_semeval_{lang}.png")
shap_bar(RES / "SHAP" / "Tokopedia_SHAP_example.csv", "fig6_shap_tokopedia.png")
print("figures written to", OUT)
