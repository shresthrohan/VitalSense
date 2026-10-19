"""Smoke check: python test_pipeline.py  (needs vitals_dataset.csv + health_model.joblib)."""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
import infer_demo as live

d = pd.read_csv('vitals_dataset.csv'); d = d[d.patient == 5].head(300)

# live features must equal the training-time window features (same column order as FEATS)
W = live.W; hr = swv(d.hr.values.astype(float), W); tp = swv(d.temp.values, W); mo = swv(d.motion.values, W)
x = np.arange(W) - (W - 1) / 2; slope = lambda a: (a * x).sum(1) / (x ** 2).sum()
train = np.column_stack([hr[:, -1], tp[:, -1], hr.mean(1), hr.std(1), hr.min(1), hr.max(1), slope(hr), hr[:, -1] - hr[:, -2],
                         tp.mean(1), tp.std(1), tp.min(1), tp.max(1), slope(tp), mo.mean(1), mo.max(1)])
liveX = np.array([live.features(hr[i], tp[i], mo[i]) for i in range(len(hr))])
assert np.allclose(train, liveX), 'live feature extraction drifted from training'

# alert logic: K-persistence, escalation re-alerts, de-escalation stays quiet
class Fake:
    def __init__(s, seq): s.seq = iter(seq)
    def predict(s, X): return [next(s.seq)]
preds = [0, 1, 1, 1, 1, 2, 1, 2, 0, 3, 3, 3]
live.model = Fake(preds)
alerts = [a for _, _, a in live.run([(70, 36.7, 0.1)] * (W - 1 + len(preds)))]
assert alerts == [None, None, None, 1, None, 2, None, None, None, None, None, 3], alerts
print('ok')
