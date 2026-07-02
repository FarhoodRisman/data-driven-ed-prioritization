"""
benchmark.py
===================================================================
Benchmark the proposed heuristic against a prior dynamic prioritization
method to establish novelty (policy benchmark).

Methods compared (all on the same simulation, common random numbers):
  * CTAS+FIFO            - static triage priority (prior, baseline)
  * APQ                  - Accumulating Priority Queue: priority accrues
                           with waiting time at a class rate b_k; the prior
                           DYNAMIC prioritization benchmark
                           (Stanford et al. 2014; Vanbrabant et al. 2021)
  * Ours (CTAS routing)  - our multi-attribute similarity RANKING with the
                           same CTAS-based pod routing  -> isolates the
                           prioritization contribution, head-to-head vs APQ
  * Ours (full)          - our method with prediction-driven routing + ranking

Run at alpha = 0 (perfect predictions) and alpha = 1 (observed error).
APQ and CTAS+FIFO use no predictions, so they are alpha-invariant.

Outputs: benchmark_results.csv  +  printed table.
"""
import numpy as np, pandas as pd
import ed_simulation as ed
from run_robustness import make_noise

def row(policy_key, alpha, n_reps, days, warmup, seed):
    r = ed.run_replications(policy_key, n_reps=n_reps, days=days,
                            warmup_days=warmup, base_seed=seed, noise=make_noise(alpha))
    pc = r["per_ctas"]
    def ttb(c): return float(pc.loc[c, "TTB"]) if c in pc.index else float("nan")
    return dict(policy=r["policy"], alpha=alpha, TTB=r["TTB"], LOS=r["LOS"],
                RATP=r["RATP"], TTB_CTAS1=ttb(1), TTB_CTAS2=ttb(2), TTB_CTAS5=ttb(5))

def run(n_reps=12, days=28, warmup=7, seed=700, csv="benchmark_results.csv"):
    rows = []
    # prior methods (prediction-free -> report once, label "any alpha")
    rows.append(row("baseline", 0.0, n_reps, days, warmup, seed)); rows[-1]["alpha"]="--"
    rows.append(row("apq",      0.0, n_reps, days, warmup, seed)); rows[-1]["alpha"]="--"
    # our method, ranking-only (CTAS routing) and full, at alpha 0 and 1
    for key in ["multi_ctasroute", "multi"]:
        for a in [0.0, 1.0]:
            rows.append(row(key, a, n_reps, days, warmup, seed))
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark vs prior dynamic prioritization")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--days", type=int, default=35)
    args = ap.parse_args()
    print(f"Benchmark: reps={args.reps}, days={args.days}\n")
    run(n_reps=args.reps, days=args.days)
    print("\nDONE. Output: benchmark_results.csv")
