"""Replays every unseen test patient through the live alert loop (AI model and threshold rule)
and writes dashboard/data.js for the static dashboard.  Usage: python export_dashboard.py"""
import json, pandas as pd
from generate_data import rule
import infer_demo as live

class Rule:  # threshold baseline behind the same predict() interface, so both go through identical alert logic
    def predict(self, X): return rule(X[:, 0], X[:, 1])

R = json.load(open('results.json')); d = pd.read_csv('vitals_dataset.csv'); out = {}
for pid in R['data']['test_patient_ids']:
    p = d[d.patient == pid]; stream = list(zip(p.hr, p.temp, p.motion)); rec = {}
    for name, clf in (('ai', None), ('rule', Rule())):
        steps = list(live.run(stream, clf))
        rec[name] = [0] * (live.W - 1) + [s[1] for s in steps]
        rec[name + '_alerts'] = [[t, a] for t, _, a in steps if a]
    out[pid] = dict(hr=p.hr.tolist(), temp=p.temp.tolist(), motion=p.motion.tolist(), label=p.label.tolist(), **rec)
    print(pid, 'AI alerts', len(rec['ai_alerts']), 'rule alerts', len(rec['rule_alerts']))
summary = {k: R[k] for k in ('best', 'exercise', 'latency_ms_single', 'model_kb', 'data')}
summary['models'] = {n: {k: R[n][k] for k in ('acc', 'macro_f1', 'false_alarms_per_hour', 'median_delay_s', 'critical_event_recall', 'recall')}
                     for n in ('Threshold rule', 'Logistic Regression', 'Gradient Boosting', 'Random Forest', 'Neural Network (MLP)')}
summary['persistence'] = R['persistence']['3']
with open('dashboard/data.js', 'w') as f:
    f.write('window.PATIENTS = ' + json.dumps(out, separators=(',', ':')) + ';\nwindow.RESULTS = ' + json.dumps(summary, separators=(',', ':')) + ';\n')
