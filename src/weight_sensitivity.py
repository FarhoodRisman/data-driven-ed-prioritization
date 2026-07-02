"""
weight_sensitivity.py
===================================================================
Weight-sensitivity analysis for the multi-attribute policy.

Re-runs the prediction-error sweep across a family of weight vectors
that shift emphasis from observed CTAS (w_A) toward the predicted
attributes. If the error injection is biting, prediction-heavy
weightings degrade steeply as alpha rises while CTAS-heavy weightings
stay flat; the chosen operating point (w_A = 0.4) sits at a robust
sweet spot.

Outputs:
  * weight_sensitivity_results.csv
  * weight_sensitivity.png   (RATP vs alpha per weight config)
"""
import numpy as np, pandas as pd
import ed_simulation as ed
from ed_simulation import Policy, POLICIES, run_replications
from run_robustness import make_noise

# Target codes are those of the multi-attribute policy.
Q_MAIN   = POLICIES["multi"].q_main
Q_URGENT = POLICIES["multi"].q_urgent

# Relative split of the PREDICTED-attribute block (B,C,D,E,F,G) from the
# multi-attribute policy weights: B .3, C .1, D .1, E .05, F 0, G .05.
RATIO_PRED = np.array([0.30, 0.10, 0.10, 0.05, 0.0, 0.05])
RATIO_PRED = RATIO_PRED / RATIO_PRED.sum()

def weights_for(wA):
    """w = [w_A, then predicted block scaled to (1-w_A)]."""
    return np.concatenate([[wA], (1.0 - wA) * RATIO_PRED])

def policy_for(wA):
    return Policy(name=f"wA={wA:.2f}", q_main=Q_MAIN, q_urgent=Q_URGENT,
                  weights=weights_for(wA), route="sim", rank="sim")

def run(wA_grid=(0.7, 0.4, 0.2, 0.0), alphas=(0.0, 1.0, 2.0),
        n_reps=3, days=28, warmup=7, base_seed=700,
        csv="weight_sensitivity_results.csv"):
    # baseline reference (prediction-independent)
    b = run_replications("baseline", n_reps=n_reps, days=days, warmup_days=warmup,
                         base_seed=base_seed, noise=make_noise(0.0))
    base_ratp = b["RATP"]
    print(f"baseline RATP = {base_ratp:.3f} (reference = 1.00)\n")
    rows = []
    hdr = "wA,alpha,TTB,LOS,RATP,RATP_rel,improvement_pct\n"
    open(csv, "w").write(hdr)
    for wA in wA_grid:
        # inject the custom policy under a temporary key
        POLICIES["_wsens"] = policy_for(wA)
        for a in alphas:
            r = run_replications("_wsens", n_reps=n_reps, days=days,
                                 warmup_days=warmup, base_seed=base_seed,
                                 noise=make_noise(a))
            rel = r["RATP"] / base_ratp
            imp = (base_ratp - r["RATP"]) / base_ratp * 100
            rows.append(dict(wA=wA, alpha=a, TTB=r["TTB"], LOS=r["LOS"],
                             RATP=r["RATP"], RATP_rel=rel, improvement_pct=imp))
            with open(csv, "a") as fh:
                fh.write(f"{wA},{a},{r['TTB']:.2f},{r['LOS']:.2f},{r['RATP']:.4f},"
                         f"{rel:.4f},{imp:.1f}\n")
            print(f"wA={wA:.2f}  a={a:<4}  TTB={r['TTB']:6.1f}  "
                  f"RATP_rel={rel:.3f}  improvement={imp:5.1f}%", flush=True)
    return pd.DataFrame(rows), base_ratp

def plot(df, path="weight_sensitivity.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for wA, g in df.groupby("wA"):
        g = g.sort_values("alpha")
        lbl = f"$w_A$={wA:.2f}" + ("  (chosen)" if abs(wA-0.4) < 1e-9 else "")
        ax.plot(g["alpha"], g["RATP_rel"], marker="o", label=lbl)
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="baseline CTAS+FIFO")
    ax.set_xlabel(r"prediction-error level $\alpha$  (1 = observed)")
    ax.set_ylabel("RATP relative to baseline  (lower = better)")
    ax.set_title("Weight sensitivity: prediction-heavy policies degrade under error")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); print("saved", path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Weight-sensitivity sweep")
    ap.add_argument("--reps", type=int, default=200, help="replications per condition")
    ap.add_argument("--days", type=int, default=35, help="sim days (incl. warm-up)")
    args = ap.parse_args()
    df, base = run(n_reps=args.reps, days=args.days)
    plot(df)
    print("\nDONE. Outputs: weight_sensitivity_results.csv, weight_sensitivity.png")
