"""
tail_risk.py -- 90th/95th-percentile TTB by CTAS (tail-risk analysis).
Reports worst-case waits for baseline CTAS+FIFO vs the multi-attribute policy.
Usage:  python tail_risk.py --reps 500
"""
import argparse, pandas as pd, ed_simulation as ed
from run_robustness import make_noise

def run(n_reps=200, days=35, warmup=7, seed=900, csv="tail_risk_results.csv"):
    rows=[]
    for key in ["baseline","multi"]:
        r=ed.run_replications(key, n_reps=n_reps, days=days, warmup_days=warmup,
                              base_seed=seed, noise=make_noise(0.0))
        pc=r["per_ctas"]
        for c in [1,2,3,4,5]:
            if c in pc.index:
                rows.append(dict(policy=r["policy"], ctas=c,
                                 mean=round(pc.loc[c,"TTB"],1),
                                 p90=round(pc.loc[c,"TTB_p90"],1),
                                 p95=round(pc.loc[c,"TTB_p95"],1)))
    df=pd.DataFrame(rows); df.to_csv(csv,index=False)
    print(df.pivot_table(index="ctas",columns="policy",values=["mean","p90","p95"]).round(1).to_string())
    return df

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=200); ap.add_argument("--days",type=int,default=35)
    a=ap.parse_args(); print(f"Tail-risk: reps={a.reps}, days={a.days}\n"); run(n_reps=a.reps,days=a.days)
