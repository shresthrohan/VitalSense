"""Live-inference demo: feeds a stream of (hr, temp, motion) samples to the trained model.
Usage: python infer_demo.py [patient_id]   (uses vitals_dataset.csv as the 'sensor stream')
Test patients (never seen in training): 5 7 10 16 21 22 24 25 26 29."""
import sys, collections, joblib, numpy as np, pandas as pd
bundle = joblib.load('health_model.joblib'); model = bundle['model']; W = bundle['window']; NAMES = bundle['labels']; K = 3  # persistence: alert after K consecutive abnormal windows
def features(hr, tp, mo):
    hr, tp, mo = (np.asarray(a, float) for a in (hr, tp, mo)); x = np.arange(W) - (W - 1) / 2; sl = lambda a: float((a * x).sum() / (x ** 2).sum())
    return [hr[-1], tp[-1], hr.mean(), hr.std(), hr.min(), hr.max(), sl(hr), hr[-1] - hr[-2], tp.mean(), tp.std(), tp.min(), tp.max(), sl(tp), mo.mean(), mo.max()]
def run(stream, clf=None):
    """Yields (t_index, raw_prediction, alert_class_or_None) per sample once the window is full.
    Alerts fire after K consecutive abnormal windows, and again only if the episode escalates (WARNING -> CRITICAL)."""
    clf = clf or model; buf = collections.deque(maxlen=W); streak = 0; alerted = 0
    for t, s in enumerate(stream):
        buf.append(s)
        if len(buf) < W: continue
        pred = int(clf.predict(np.array([features(*zip(*buf))]))[0]); streak = streak + 1 if pred else 0
        if not streak: alerted = 0
        alert = pred if streak >= K and pred > alerted else None
        if alert: alerted = alert
        yield t, pred, alert
if __name__ == '__main__':
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = pd.read_csv('vitals_dataset.csv'); d = d[d.patient == pid]; hr, tp, mo = d.hr.values, d.temp.values, d.motion.values
    for t, _, a in run(zip(hr, tp, mo)):
        if a: print(f't={t*10:5d}s  ALERT  {NAMES[a]:<12} HR={hr[t]:3d} T={tp[t]:.2f} motion={mo[t]:.2f}')
