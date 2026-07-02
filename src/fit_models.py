"""
fit_models.py -- Fit MLR and Random Forest predictive models
===================================================================
Requires scikit-learn (`pip install scikit-learn`).

Fits the triage-time predictive models and produces, for each task:
  * accuracy and per-class precision & recall
  * a confusion matrix (csv + pdf)
  * the fitted hyper-parameters
  * row-normalized confusion matrices exported to
    confusion_matrices.json, used by the robustness and benchmark
    noise experiments.

Tasks (all from triage-time features only):
  - disposition  (admit / discharge / transfer; + acute/non-acute variant)
  - service_time (5 classes: very short .. very long)
  - diagnostics  (zero / low / high / super utilization)
  - consultation (yes / no)

Usage:
  python fit_models.py --data your_data.xlsx
  python fit_models.py --data sample.csv --quick   # smoke test
"""
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd

# ---------------------------------------------------------------- read
def read_data(path, nrows=None):
    if str(path).lower().endswith((".xlsx", ".xls")):
        cols=['Client.ID..Anonymized.','EmergencyDepartment','ModeOfArrival','gender',
        'Age.at.extract','CTAS.Score','EDComplaintCategory','Triage.Start.Time',
        'Total.Lab.orders','Total.LAB.Tests.ordered','Total.Diagnostic.orders',
        'Total.Diagnostics.Tests.ordered','Consult.Service','Disposition.Decision',
        'LOS.Hours','Wait.Time.Hours']
        return pd.read_excel(path, usecols=lambda c: c in cols, nrows=nrows)
    return pd.read_csv(path, low_memory=False, on_bad_lines='skip', nrows=nrows)

# ---------------------------------------------------------------- helpers
def norm_complaint(s):
    s=str(s).strip().lower().replace('&','and').replace('/',' ').replace(',',' ')
    s=' '.join(s.split())
    return s

def arrival_mode(m):
    m=str(m).lower()
    if any(k in m for k in ['ems','ambulance','police','carried','air']): return 'Emergency'
    return 'Walk-in'

# ---------------------------------------------------------------- feature/target engineering
def load_and_engineer(path, nrows=None, verbose=True):
    df=read_data(path, nrows=nrows)
    df.columns=[c.strip() for c in df.columns]
    # ---- numeric base ----
    df['ctas']=pd.to_numeric(df['CTAS.Score'], errors='coerce')
    df['age']=pd.to_numeric(df['Age.at.extract'], errors='coerce').clip(0,110)
    df['los']=pd.to_numeric(df['LOS.Hours'], errors='coerce')*60.0
    df['wait']=pd.to_numeric(df['Wait.Time.Hours'], errors='coerce')*60.0
    df['svc']=df['los']-df['wait']
    df['tri']=pd.to_datetime(df['Triage.Start.Time'], errors='coerce')
    disp=df['Disposition.Decision'].fillna('')
    # ---- analytic cohort (matches the cleaning waterfall) ----
    incomplete=disp.str.contains('LWBS')|disp.str.contains('LAMA|Against Medical',case=False,regex=True)\
               |disp.str.contains('Death|Deceased',case=False,regex=True)
    keep=(df['ctas'].between(1,5))&df['gender'].isin(['Male','Female'])&~incomplete\
         &df['los'].gt(0)&df['wait'].ge(0)&df['svc'].gt(0)&df['tri'].notna()
    df=df[keep].copy()
    def iqr(s):
        q1,q3=s.quantile(.25),s.quantile(.75); i=q3-q1; return s.between(q1-1.5*i,q3+1.5*i)
    df=df[iqr(df['wait'])&iqr(df['los'])&iqr(df['svc'])].copy()

    # ---- FEATURES (triage-time only) ----
    X=pd.DataFrame(index=df.index)
    X['ctas']=df['ctas'].astype(int)
    X['arrival_mode']=df['ModeOfArrival'].map(arrival_mode)
    X['sex']=df['gender']
    X['age_cat']=pd.cut(df['age'],[0,18,40,60,80,200],labels=['0-17','18-39','40-59','60-79','80+'])
    X['complaint']=df['EDComplaintCategory'].map(norm_complaint)
    X['period_of_year']=pd.cut(df['tri'].dt.month,[0,6,9,12],labels=['Jan-Jun','Jul-Sep','Oct-Dec'])
    X['day_of_week']=df['tri'].dt.dayofweek
    hr=df['tri'].dt.hour
    pod=pd.Series('Night',index=df.index)
    pod[(hr>=6)&(hr<12)]='Morning'; pod[(hr>=12)&(hr<17)]='Afternoon'; pod[(hr>=17)&(hr<22)]='Evening'
    X['period_of_day']=pod
    # recent revisit: same client seen in prior 72h
    d=df[['Client.ID..Anonymized.','tri']].sort_values(['Client.ID..Anonymized.','tri'])
    prevt=d.groupby('Client.ID..Anonymized.')['tri'].shift(1)
    rr=((d['tri']-prevt).dt.total_seconds()/3600.0<=72)
    X['recent_revisit']=rr.reindex(df.index).fillna(False).astype(int)
    # arrival rate: arrivals at the same facility within the prior 60 min
    ar=pd.Series(0,index=df.index)
    for fac,g in df.groupby('EmergencyDepartment'):
        t=g['tri'].sort_values(); tv=t.values.astype('datetime64[m]').astype(np.int64)
        lo=np.searchsorted(tv, tv-60, side='left'); cnt=np.arange(len(tv))-lo
        ar.loc[t.index]=cnt
    X['arrival_rate']=ar.values

    # ---- TARGETS ----
    y={}
    laborders=pd.to_numeric(df['Total.Lab.orders'],errors='coerce').fillna(0)
    dxorders =pd.to_numeric(df['Total.Diagnostic.orders'],errors='coerce').fillna(0)
    tot=laborders+dxorders          # ORDER counts (utilization), not individual tests
    y['diagnostics']=pd.cut(tot,[-1,0,5,10,1e9],labels=['zero','low','high','super']).astype(str)
    y['consultation']=np.where(df['Consult.Service'].notna(),'yes','no')
    # service-time class by fixed thresholds (min); central 'average' band
    y['service_time']=pd.cut(df['svc'],[-1,30,60,240,420,1e9],labels=['very_short','short','average','long','very_long']).astype(str)
    # disposition 4-class
    def dispo(x):
        if 'Critical Care' in x or 'OR' in x: return 'acute_admit'
        if 'TRSF to IP' in x: return 'nonacute_admit'
        if 'TRSF' in x: return 'transfer'
        return 'home'
    y['disposition']=df['Disposition.Decision'].fillna('').map(dispo)

    if verbose:
        print(f"analytic cohort: {len(df):,} rows")
        print("feature dtypes:\n", X.dtypes.to_dict())
        print("missing per feature:", X.isna().sum().to_dict())
        for k,v in y.items():
            print(f"  target {k}: {pd.Series(v).value_counts().to_dict()}")
    return X, y

# ---------------------------------------------------------------- fitting (needs sklearn)
def fit_and_report(X, y, out_prefix="model"):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    cat=['arrival_mode','sex','age_cat','complaint','period_of_year','period_of_day']
    num=['ctas','day_of_week','recent_revisit','arrival_rate']
    pre=ColumnTransformer([('oh',OneHotEncoder(handle_unknown='ignore'),cat)],remainder='passthrough')
    Xenc=pre.fit_transform(X[cat+num].astype({c:str for c in cat}))

    HP={'MLR':dict(max_iter=500,C=1.0,solver='lbfgs'),  # sklearn>=1.7: multinomial auto
        'RF' :dict(n_estimators=200,max_depth=None,random_state=42,n_jobs=-1)}
    real_cm={}
    summary=[]
    for task, yt in y.items():
        yt=pd.Series(yt).astype(str).values
        try:
            Xtr,Xte,ytr,yte=train_test_split(Xenc,yt,test_size=0.15,random_state=42,stratify=yt)
        except ValueError:
            Xtr,Xte,ytr,yte=train_test_split(Xenc,yt,test_size=0.15,random_state=42)
        for name,Model,hp in [('MLR',LogisticRegression,HP['MLR']),('RF',RandomForestClassifier,HP['RF'])]:
            clf=Model(**hp).fit(Xtr,ytr); pred=clf.predict(Xte)
            labels=sorted(np.unique(yt))
            cm=confusion_matrix(yte,pred,labels=labels)
            acc=accuracy_score(yte,pred)
            np.savetxt(f"{out_prefix}_{task}_{name}_cm.csv", cm, fmt='%d', delimiter=',',
                       header=",".join(map(str,labels)), comments='')
            # heatmap
            fig,ax=plt.subplots(figsize=(1.2+0.7*len(labels),1.2+0.7*len(labels)))
            ax.imshow(cm/cm.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=45,ha='right',fontsize=7)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=7)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=6)
            ax.set_title(f"{task} ({name}) acc={acc:.2f}",fontsize=8)
            ax.set_xlabel("predicted",fontsize=7); ax.set_ylabel("true",fontsize=7)
            fig.tight_layout(); fig.savefig(f"{out_prefix}_{task}_{name}_cm.pdf"); plt.close(fig)
            print(f"[{task:13s}] {name}: acc={acc:.3f}")
            summary.append(dict(task=task,model=name,acc=round(acc,3)))
            if name=='MLR':   # export MLR row-normalized cm for noise pipeline
                rn=cm/cm.sum(1,keepdims=True)
                real_cm[task]={'classes':list(map(str,labels)),'cm':rn.tolist()}
    json.dump(real_cm, open(f"{out_prefix}_confusion_matrices.json","w"), indent=1)
    pd.DataFrame(summary).to_csv(f"{out_prefix}_summary.csv",index=False)
    json.dump(HP, open(f"{out_prefix}_hyperparams.json","w"), indent=1)
    print("\nWrote confusion matrices (*_cm.csv/.pdf), confusion_matrices.json, summary.csv, hyperparams.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="your_data.xlsx")
    ap.add_argument("--quick", action="store_true", help="smoke test on 40k rows")
    args=ap.parse_args()
    X,y=load_and_engineer(args.data, nrows=40000 if args.quick else None)
    fit_and_report(X,y)
