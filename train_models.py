import numpy as np, pandas as pd, json, time, joblib
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.inspection import permutation_importance
from generate_data import LABELS, rule
W = 6; SEED = 42
df = pd.read_csv('vitals_dataset.csv')

FEATS = ['hr_last','temp_last','hr_mean','hr_std','hr_min','hr_max','hr_slope','hr_delta','temp_mean','temp_std','temp_min','temp_max','temp_slope','motion_mean','motion_max']
def slope(a): x = np.arange(a.shape[1]) - (a.shape[1]-1)/2; return (a * x).sum(1) / (x**2).sum()
def windows(g):
    hr = swv(g.hr.values.astype(float), W); tp = swv(g.temp.values, W); mo = swv(g.motion.values, W)
    X = np.column_stack([hr[:,-1], tp[:,-1], hr.mean(1), hr.std(1), hr.min(1), hr.max(1), slope(hr), hr[:,-1]-hr[:,-2],
                         tp.mean(1), tp.std(1), tp.min(1), tp.max(1), slope(tp), mo.mean(1), mo.max(1)])
    return X, g.label.values[W-1:], np.full(len(X), g.patient.iloc[0]), g.t.values[W-1:], g.event.values[W-1:]
parts = [windows(g) for _, g in df.groupby('patient')]
X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); grp = np.concatenate([p[2] for p in parts]); tt = np.concatenate([p[3] for p in parts]); ev = np.concatenate([p[4] for p in parts])
rng = np.random.default_rng(SEED); perm = rng.permutation(30); test_pat = set(perm[:10]); train_pat = set(perm[10:])
tr = np.isin(grp, list(train_pat)); te = np.isin(grp, list(test_pat))
Xtr, ytr, gtr = X[tr], y[tr], grp[tr]; Xte, yte, gte, tte, evte = X[te], y[te], grp[te], tt[te], ev[te]
print('train', Xtr.shape, 'test', Xte.shape, 'test patients', sorted(test_pat))

# ---------- threshold baseline (rule on last sample) ----------
def metrics(yt, yp):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=[0,1,2,3], zero_division=0)
    return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'), precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), support=s.tolist(), cm=confusion_matrix(yt, yp, labels=[0,1,2,3]).tolist())
def runs(mask):  # count maximal runs of True
    m = np.concatenate([[0], mask.astype(int), [0]]); return int(((m[1:] - m[:-1]) == 1).sum())
def alert_stats(yt, yp, g):
    fa = 0; hours = 0; ev_total = 0; ev_hit = 0; delays = []; crit_total = 0; crit_hit = 0
    for p in np.unique(g):
        i = g == p; a = yt[i]; b = yp[i]; hours += i.sum() * 10 / 3600
        fa += runs((a == 0) & (b != 0))
        # events = contiguous runs of label 1/2
        m = np.concatenate([[0], ((a == 1) | (a == 2)).astype(int), [0]]); st = np.where(np.diff(m) == 1)[0]; en = np.where(np.diff(m) == -1)[0]
        for s_, e_ in zip(st, en):
            ev_total += 1; hit = np.where((b[s_:e_] == 1) | (b[s_:e_] == 2))[0]
            if len(hit): ev_hit += 1; delays.append(hit[0] * 10)
            if a[s_] == 2 or (a[s_:e_] == 2).any():
                crit_total += 1; crit_hit += int((b[s_:e_] == 2).any())
    return dict(false_alarms_per_hour=fa / hours, event_recall=ev_hit / ev_total, events=ev_total, median_delay_s=float(np.median(delays)), critical_event_recall=crit_hit / max(crit_total, 1), hours=hours)

res = {}
yb = rule(Xte[:,0], Xte[:,1]); res['Threshold rule'] = dict(**metrics(yte, yb), **alert_stats(yte, yb, gte))
models = {
 'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
 'Neural Network (MLP)': make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=SEED)),
 'Random Forest': RandomForestClassifier(300, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=SEED),
 'Gradient Boosting': HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight='balanced', random_state=SEED)}
cv = {}; fitted = {}; train_time = {}
for nm, m in models.items():
    sc = []
    for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
        mm = __import__('sklearn.base', fromlist=['clone']).clone(m).fit(Xtr[a], ytr[a]); sc.append(f1_score(ytr[b], mm.predict(Xtr[b]), average='macro'))
    cv[nm] = (float(np.mean(sc)), float(np.std(sc)))
    t0 = time.time(); m.fit(Xtr, ytr); train_time[nm] = time.time() - t0; fitted[nm] = m
    yp = m.predict(Xte); res[nm] = dict(**metrics(yte, yp), **alert_stats(yte, yp, gte), cv_macro_f1=cv[nm][0], cv_std=cv[nm][1], train_s=train_time[nm])
    print(nm, round(res[nm]['acc'],4), round(res[nm]['macro_f1'],4), 'cv', round(cv[nm][0],4))
best = max(cv, key=lambda k: cv[k][0]); print('BEST by CV:', best)
bm = fitted[best]; ybest = bm.predict(Xte)

# ---------- ablation: no motion features ----------
nomo = [i for i, f in enumerate(FEATS) if not f.startswith('motion')]
from sklearn.base import clone
am = clone(models[best]).fit(Xtr[:, nomo], ytr); ya = am.predict(Xte[:, nomo])
res['ablation_no_motion'] = dict(**metrics(yte, ya), **alert_stats(yte, ya, gte))

# ---------- isolation forest (unsupervised) ----------
sc_ = StandardScaler().fit(Xtr[ytr == 0]); iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(sc_.transform(Xtr[ytr == 0]))
s_tr = -iso.score_samples(sc_.transform(Xtr[ytr == 0])); thr = float(np.percentile(s_tr, 98)); s_te = -iso.score_samples(sc_.transform(Xte))
ab = (yte != 0).astype(int); pred = (s_te > thr).astype(int)
p_, r_, f_, _ = precision_recall_fscore_support(ab, pred, average='binary', zero_division=0)
res['isolation_forest'] = dict(auc=float(roc_auc_score(ab, s_te)), precision=float(p_), recall=float(r_), f1=float(f_), thr=thr)
# anomaly detection by the supervised best model (abnormal = any class != 0) for fair comparison
pb = (ybest != 0).astype(int); p2, r2, f2, _ = precision_recall_fscore_support(ab, pb, average='binary', zero_division=0)
res['best_binary'] = dict(precision=float(p2), recall=float(r2), f1=float(f2))
pr = (yb != 0).astype(int); p3, r3, f3, _ = precision_recall_fscore_support(ab, pr, average='binary', zero_division=0)
res['rule_binary'] = dict(precision=float(p3), recall=float(r3), f1=float(f3))

# ---------- feature importance (permutation) ----------
idx = np.random.default_rng(1).choice(len(Xte), 4000, replace=False)
pi = permutation_importance(bm, Xte[idx], yte[idx], scoring='f1_macro', n_repeats=3, random_state=SEED, n_jobs=-1)
order = np.argsort(-pi.importances_mean); res['importance'] = [(FEATS[i], float(pi.importances_mean[i])) for i in order]

# ---------- latency / size ----------
x1 = Xte[:1]; t0 = time.time()
for _ in range(500): bm.predict(x1)
res['latency_ms_single'] = (time.time() - t0) / 500 * 1000
t0 = time.time(); bm.predict(Xte); res['latency_ms_batch_per_window'] = (time.time() - t0) / len(Xte) * 1000
joblib.dump(dict(model=bm, features=FEATS, window=W, labels=LABELS), 'health_model.joblib')
import os; res['model_kb'] = os.path.getsize('health_model.joblib') / 1024
res['best'] = best; res['cv'] = cv
res['data'] = dict(rows=len(df), windows=len(X), train_windows=int(tr.sum()), test_windows=int(te.sum()), train_patients=len(train_pat), test_patients=len(test_pat), test_patient_ids=sorted(int(i) for i in test_pat),
    class_counts_all={LABELS[k]: int((y == k).sum()) for k in range(4)}, class_counts_test={LABELS[k]: int((yte == k).sum()) for k in range(4)}, class_counts_train={LABELS[k]: int((ytr == k).sum()) for k in range(4)})
# exercise false-alarm check: normal windows with high motion
ex = (yte == 0) & (Xte[:, 13] > 0.6)
res['exercise'] = dict(n=int(ex.sum()), rule_flagged=float((yb[ex] != 0).mean()), model_flagged=float((ybest[ex] != 0).mean()))
# fault type recall
res['fault_type'] = {}
for k in ['fault_dropout','fault_spike','fault_stuck']:
    m_ = evte == k; res['fault_type'][k] = dict(n=int(m_.sum()), rule=float((yb[m_] == 3).mean()), model=float((ybest[m_] == 3).mean()))

# ---------- persistence filter: alert only after k consecutive abnormal predictions ----------
def persist(pred, g, k):
    out = np.zeros_like(pred)
    for p in np.unique(g):
        i = np.where(g == p)[0]; a = pred[i]; c = np.convolve((a != 0).astype(int), np.ones(k, int))[:len(a)]; out[i] = np.where(c >= k, a, 0)
    return out
res['persistence'] = {}
for k in (1, 2, 3, 4):
    res['persistence'][str(k)] = dict(rule=alert_stats(yte, persist(yb, gte, k), gte), model=alert_stats(yte, persist(ybest, gte, k), gte),
                                      model_macro_f1=float(f1_score(yte, persist(ybest, gte, k), average='macro')))
json.dump(res, open('results.json', 'w'), indent=1, default=float)
print(json.dumps({k: res[k] for k in ['best','isolation_forest','best_binary','rule_binary','latency_ms_single','latency_ms_batch_per_window','model_kb','exercise','fault_type','data']}, indent=1, default=float))
for k in ['Threshold rule', best, 'ablation_no_motion']:
    r = res[k]; print(k, {a: (round(r[a],4) if isinstance(r[a], float) else r[a]) for a in ['acc','macro_f1','false_alarms_per_hour','event_recall','events','median_delay_s','critical_event_recall']}, 'recall', np.round(r['recall'],3).tolist())
print(res['importance'][:8]); print(np.array(res[best]['cm'])); print(np.array(res['Threshold rule']['cm']))
