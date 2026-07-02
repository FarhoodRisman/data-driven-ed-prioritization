"""
make_figures.py
Regenerate the result figures as vector PDF from the experiment CSVs,
so the figures always match the result tables.
Outputs: Figure_Results/Figure_Robustness.pdf,
         Figure_Results/Figure_WeightSensitivity.pdf
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("Figure_Results", exist_ok=True)

# ---------- Robustness ----------
r = pd.read_csv("robustness_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for pol, g in r.groupby("policy"):
    g = g.sort_values("alpha")
    name = g["policy_name"].iloc[0]
    ax[0].plot(g["alpha"], g["RATP"], marker="o", label=name)
    ax[1].plot(g["alpha"], g["TTB"], marker="o", label=name)
ax[0].set_title("Risk-adjusted tardiness (RATP) vs prediction error")
ax[0].set_ylabel("RATP")
ax[1].set_title("Overall time-to-bed (TTB) vs prediction error")
ax[1].set_ylabel("TTB (min)")
for a in ax:
    a.set_xlabel(r"prediction-error level $\alpha$  (1 = observed)")
    a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig("Figure_Results/Figure_Robustness.pdf"); plt.close(fig)

# ---------- Weight sensitivity ----------
w = pd.read_csv("weight_sensitivity_results.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
for wA, g in w.groupby("wA"):
    g = g.sort_values("alpha")
    lbl = f"$w_A$={wA:.2f}" + ("  (selected)" if abs(wA-0.4) < 1e-9 else "")
    ax.plot(g["alpha"], g["RATP_rel"], marker="o", label=lbl)
ax.axhline(1.0, ls="--", color="gray", lw=1, label="baseline CTAS+FIFO")
ax.set_xlabel(r"prediction-error level $\alpha$  (1 = observed)")
ax.set_ylabel("RATP relative to baseline  (lower = better)")
ax.set_title("Weight sensitivity: prediction-heavy policies degrade under error")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("Figure_Results/Figure_WeightSensitivity.pdf"); plt.close(fig)
print("wrote Figure_Results/Figure_Robustness.pdf and Figure_WeightSensitivity.pdf")
