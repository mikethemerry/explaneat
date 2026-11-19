"""Heart: does boosting beat logistic regression, or only on the paper split? 100 random stratified
242/61 splits of all 303 rows, each comparator at the setting CV chose on the paper split (read from
benchmarks.csv, as are the fixed-split test AUCs). Needs cache/heart.npz (export_data.py) and
`interpret` (EBM). Exploratory, 20 Nov 2026. Output: heart_splits.json (the table
thesis/ch7/appendix/heart_splits_table.py reads its summaries; `per_split` holds every split's test AUC per
model, read by thesis/ch7/figure_candidates).

    OMP_NUM_THREADS=2 <venv with interpret>/python experiments/benchmarks/heart_splits.py [cache dir]
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
import csv, sys, json, pathlib
HERE=pathlib.Path(__file__).parent
CACHE=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else HERE/"cache"
d=np.load(CACHE/"heart.npz"); X=np.vstack([d["Xtr"],d["Xte"]]); y=np.concatenate([d["ytr"],d["yte"]]).ravel()
# the fixed-split rows of benchmarks.csv (make_csv.py): CV-chosen settings and test AUCs
KEYS={"L2":"lr_l2","L1":"lr_l1","RF":"rf","GB":"gb","EBM":"ebm"}
ROWS={r["model_key"]:r for r in csv.DictReader(open(HERE/"benchmarks.csv")) if r["dataset"]=="heart" and r["experiment"]=="1,2"}
P={k:json.loads(ROWS[v]["params"]) for k,v in KEYS.items()}
paper={k:float(ROWS[v]["test_value"]) for k,v in KEYS.items()}
M={"L2":lambda:LogisticRegression(**P["L2"],max_iter=5000),
 "L1":lambda:LogisticRegression(**P["L1"],penalty="l1",solver="liblinear",random_state=0),
 "RF":lambda:RandomForestClassifier(500,**P["RF"],random_state=0,n_jobs=4),
 "GB":lambda:HistGradientBoostingClassifier(**P["GB"],random_state=0),
 "EBM":lambda:ExplainableBoostingClassifier(**P["EBM"],random_state=0,n_jobs=4)}
R={k:[] for k in M}
sss=StratifiedShuffleSplit(n_splits=100,test_size=61,random_state=1)
for tr,te in sss.split(X,y):
    for k,f in M.items():
        m=f().fit(X[tr],y[tr]); R[k].append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
R={k:np.array(v) for k,v in R.items()}
for k,v in R.items(): print(f"{k}: mean {v.mean():.3f}  sd {v.std():.3f}  5-95% {np.percentile(v,5):.3f}-{np.percentile(v,95):.3f}")
gap=paper["GB"]-paper["L2"]   # the fixed split's GB-over-L2 margin
for k in ["RF","GB","EBM","L1"]:
    dd=R[k]-R["L2"]; print(f"{k}-L2: mean {dd.mean():+.3f}  share>0 {np.mean(dd>0):.2f}  share>=fixed-split GB margin {gap:+.3f}: {np.mean(dd>=gap):.2f}")

out={k:{"mean":float(v.mean()),"sd":float(v.std()),"p5":float(np.percentile(v,5)),"p95":float(np.percentile(v,95)),
        "share_beats_L2":(None if k=="L2" else float(np.mean(v>R["L2"]))),"paper_split":paper[k]} for k,v in R.items()}
# per-split test AUCs (split order of StratifiedShuffleSplit(random_state=1)), for the figure candidates
per_split={k:[float(x) for x in v] for k,v in R.items()}
json.dump({"n_splits":100,"test_size":61,"seed":1,"models":out,"per_split":per_split},open(HERE/"heart_splits.json","w"),indent=1)
