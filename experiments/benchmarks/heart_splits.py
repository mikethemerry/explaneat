"""Heart: does boosting beat logistic regression, or only on the paper split? 100 random stratified
242/61 splits of all 303 rows, each comparator at the setting CV chose on the paper split (benchmarks.csv).
Needs cache/heart.npz (export_data.py). Exploratory, 20 Nov 2026. Output: heart_splits.txt"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
d=np.load(__import__("pathlib").Path(__file__).parent / "cache/heart.npz"); X=np.vstack([d["Xtr"],d["Xte"]]); y=np.concatenate([d["ytr"],d["yte"]]).ravel()
M={"L2":lambda:LogisticRegression(C=0.1,max_iter=5000),"L1":lambda:LogisticRegression(C=0.3,penalty="l1",solver="liblinear"),
 "RF":lambda:RandomForestClassifier(500,max_depth=5,min_samples_leaf=5,random_state=0,n_jobs=4),
 "GB":lambda:HistGradientBoostingClassifier(l2_regularization=1.0,learning_rate=0.03,max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,random_state=0),
 "EBM":lambda:ExplainableBoostingClassifier(interactions=0,max_bins=32,random_state=0,n_jobs=4)}
R={k:[] for k in M}
sss=StratifiedShuffleSplit(n_splits=100,test_size=61,random_state=1)
for tr,te in sss.split(X,y):
    for k,f in M.items():
        m=f().fit(X[tr],y[tr]); R[k].append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
R={k:np.array(v) for k,v in R.items()}
for k,v in R.items(): print(f"{k}: mean {v.mean():.3f}  sd {v.std():.3f}  5-95% {np.percentile(v,5):.3f}-{np.percentile(v,95):.3f}")
for k in ["RF","GB","EBM","L1"]:
    dd=R[k]-R["L2"]; print(f"{k}-L2: mean {dd.mean():+.3f}  share>0 {np.mean(dd>0):.2f}  share>=+0.026 {np.mean(dd>=0.026):.2f}")
