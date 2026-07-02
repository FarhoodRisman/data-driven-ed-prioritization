"""
run_robustness.py
===================================================================
Prediction-error robustness experiment.

Re-runs the calibrated ED simulation for each policy under increasing
prediction error (alpha-scaled, using the fitted models' confusion
structure). TRUE attributes always drive the simulation physics; only
the PREDICTED attributes that feed the routing/priority policy are
corrupted.

Outputs:
  * robustness_results.csv      - policy x alpha x metrics
  * robustness_degradation.png  - RATP & low-acuity TTB vs alpha

The baseline (CTAS+FIFO) ignores predictions, so it is flat across
alpha and serves as the reference line.
"""
import numpy as np, pandas as pd
import ed_simulation as ed
from prediction_noise import scale_offdiagonal
from confusion_matrices import build_attr_specs_real as _specs

SPECS = _specs()
# (spec key, patient TRUE attr, patient PRED attr)
ATTRS = [("service_time","b_true","b_pred"), ("admission","c_true","c_pred"),
         ("disposition","d_true","d_pred"), ("diag_test","e_true","e_pred"),
         ("resource_use","f_true","f_pred"), ("consult","g_true","g_pred")]

def make_noise(alpha):
    """Return noise(patient, rng) that corrupts predicted attrs at level alpha."""
    scaled, cidx, classes = {}, {}, {}
    for key,_,_ in ATTRS:
        P = scale_offdiagonal(np.asarray(SPECS[key]["cm"], float), alpha)
        scaled[key] = P
        classes[key] = SPECS[key]["classes"]
        cidx[key] = {c:i for i,c in enumerate(classes[key])}
    def noise(p, rng):
        for key, tg, pr in ATTRS:
            t = getattr(p, tg)
            if key == "resource_use" and t == 4:      # zero-use: no test, known
                setattr(p, pr, 4); continue
            i = cidx[key].get(t)
            if i is None:
                setattr(p, pr, t); continue
            row = scaled[key][i]
            setattr(p, pr, classes[key][rng.choice(len(row), p=row)])
    return noise


def run_sweep(policies=("baseline","multi"),
              alphas=(0.0,0.5,1.0,1.5,2.0), n_reps=500,
              days=28, warmup=7, base_seed=700, csv="robustness_results.csv"):
    rows = []
    open(csv,"w").write("policy,policy_name,alpha,TTB,LOS,RATP,TTB95_CTAS1,TTB95_CTAS2\n")
    for pol in policies:
        for a in alphas:
            res = ed.run_replications(pol, n_reps=n_reps, days=days,
                                      warmup_days=warmup, base_seed=base_seed,
                                      noise=make_noise(a))
            pc = res["per_ctas"]
            t1 = float(pc.loc[1,"TTB_p95"]) if 1 in pc.index else float("nan")
            t2 = float(pc.loc[2,"TTB_p95"]) if 2 in pc.index else float("nan")
            row = dict(policy=pol, policy_name=res["policy"], alpha=a,
                       TTB=res["TTB"], LOS=res["LOS"], RATP=res["RATP"],
                       TTB95_CTAS1=t1, TTB95_CTAS2=t2)
            rows.append(row)
            with open(csv,"a") as fh:
                fh.write(f"{pol},{res['policy']},{a},{res['TTB']:.2f},{res['LOS']:.2f},"
                         f"{res['RATP']:.4f},{t1:.1f},{t2:.1f}\n")
            print(f"{pol:>9}  a={a:<4}  TTB={res['TTB']:6.1f}  LOS={res['LOS']:6.1f}  "
                  f"RATP={res['RATP']:.3f}", flush=True)
    return pd.DataFrame(rows)

def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol, g in df.groupby("policy"):
        g = g.sort_values("alpha")
        ax[0].plot(g["alpha"], g["RATP"], marker="o", label=g["policy_name"].iloc[0])
        ax[1].plot(g["alpha"], g["TTB"], marker="o", label=g["policy_name"].iloc[0])
    for a, ttl, yl in ((0,"Risk score (RATP) vs prediction error","RATP"),
                       (1,"Overall TTB vs prediction error","TTB (min)")):
        ax[a].set_title(ttl); ax[a].set_xlabel(r"prediction-error level $\alpha$  (1 = observed)")
        ax[a].set_ylabel(yl); ax[a].grid(alpha=.3); ax[a].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prediction-error robustness sweep")
    ap.add_argument("--reps", type=int, default=200, help="replications per condition")
    ap.add_argument("--days", type=int, default=35, help="sim days (incl. warm-up)")
    args = ap.parse_args()
    print(f"Running robustness sweep: reps={args.reps}, days={args.days}\n")
    df = run_sweep(n_reps=args.reps, days=args.days)
    df.to_csv("robustness_results.csv", index=False)
    make_plot(df, "robustness_degradation.png")
    piv = df.pivot_table(index="alpha", columns="policy", values="RATP")
    if {"baseline","multi"}.issubset(piv.columns):
        piv["multi_gain_%"] = (piv["baseline"]-piv["multi"])/piv["baseline"]*100
        print("\n=== Multi-attribute advantage over baseline ===")
        print(piv.round(3).to_string())
    print("\nDONE. Outputs: robustness_results.csv, robustness_degradation.png")
