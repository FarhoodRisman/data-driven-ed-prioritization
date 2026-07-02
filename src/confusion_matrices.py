"""
confusion_matrices.py
===================================================================
Attribute noise specifications built from the fitted predictive
models' row-normalized confusion matrices P(pred | true).

Maps the 4 fitted tasks -> the simulation's 6 coded attributes
(B service, C admission, D disposition, E diagnostic test,
 F resource, G consultation), reordering to the coding order and
deriving the binary C and E rows via priors.

Matrices are embedded verbatim from the fitted-model output
(see fit_models.py).
"""
import numpy as np

# ---- row-normalized P(pred|true) from the full-data re-fit (MLR) ----
REAL = {
 "service_time": {"classes":["average","long","short","very_long","very_short"], "cm":[
   [0.8094925203145191,0.010578881661081542,0.0,0.03556957312131765,0.14435902490308167],
   [0.8401999873425733,0.02790962597303968,0.0,0.08974115562306183,0.04214923106132523],
   [0.6608249148853045,0.00273483284031925,0.0,0.014176480437573254,0.32226377183680305],
   [0.745671887881286,0.03528441879637263,0.0,0.19530090684253915,0.023742786479802144],
   [0.5802190309796735,0.0015340691183363872,0.0,0.009374866834277922,0.4088720330677121]]},
 "disposition": {"classes":["acute_admit","home","nonacute_admit","transfer"], "cm":[
   [0.0,0.6875,0.3125,0.0],
   [0.0,0.9891952309985097,0.010804769001490314,0.0],
   [0.0,0.8257581486803297,0.17424185131967027,0.0],
   [0.0,0.8762886597938144,0.12371134020618557,0.0]]},
 "diagnostics": {"classes":["high","low","super","zero"], "cm":[
   [0.15469682689286837,0.7131636820609488,0.00018850141376060322,0.13195098963242224],
   [0.029865125240847785,0.6831468044111015,4.0995367523469845e-05,0.2869470749805272],
   [0.3126469205453691,0.5881523272214386,0.0004701457451810061,0.09873060648801128],
   [0.015142073778664007,0.19994184114323696,2.0771020272515784e-05,0.7848953140578265]]},
 "consultation": {"classes":["no","yes"], "cm":[
   [0.9767069302489507,0.02329306975104931],
   [0.8248277801209054,0.17517221987909462]]},
}
# class priors (counts) from the full-data run, in REAL class order
PRIORS = {
 "disposition": {"acute_admit":107,"home":662056,"nonacute_admit":88154,"transfer":16163},
 "diagnostics": {"high":106101,"low":325239,"super":14178,"zero":320962},
}

def _reorder(task, target_names):
    names=REAL[task]["classes"]; cm=np.asarray(REAL[task]["cm"],float)
    idx=[names.index(t) for t in target_names]
    M=cm[np.ix_(idx,idx)]; return (M/M.sum(1,keepdims=True))

def _collapse(task, yes_names, no_names):
    names=REAL[task]["classes"]; cm=np.asarray(REAL[task]["cm"],float)
    pri=PRIORS[task]; n2i={n:i for i,n in enumerate(names)}; groups=[yes_names,no_names]
    out=np.zeros((2,2))
    for gi,G in enumerate(groups):
        w=np.array([pri[t] for t in G],float); w/=w.sum()
        rows=np.array([cm[n2i[t]] for t in G])
        for hj,H in enumerate(groups):
            out[gi,hj]=float((w*rows[:,[n2i[t] for t in H]].sum(1)).sum())
    return out/out.sum(1,keepdims=True)

def build_attr_specs_real():
    return {
     "service_time":{"classes":[1,2,3,4,5],
        "cm":_reorder("service_time",["very_long","long","average","short","very_short"]).tolist()},
     "disposition":{"classes":[1,2,3,4],
        "cm":_reorder("disposition",["acute_admit","nonacute_admit","transfer","home"]).tolist()},
     "resource_use":{"classes":[1,2,3,4],
        "cm":_reorder("diagnostics",["super","high","low","zero"]).tolist()},
     "consult":{"classes":[1,2],
        "cm":_reorder("consultation",["yes","no"]).tolist()},
     "admission":{"classes":[1,2],
        "cm":_collapse("disposition",["acute_admit","nonacute_admit"],["home","transfer"]).tolist()},
     "diag_test":{"classes":[1,2],
        "cm":_collapse("diagnostics",["high","low","super"],["zero"]).tolist()},
    }

if __name__=="__main__":
    for k,v in build_attr_specs_real().items():
        cm=np.array(v["cm"])
        print(f"{k:13s} classes={v['classes']} recall(diag)={[round(cm[i,i],3) for i in range(len(cm))]} "
              f"rowsums={cm.sum(1).round(3).tolist()}")
