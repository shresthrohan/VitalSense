"""Report figures from results.json + vitals_dataset.csv -> figures/*.png.  Usage: python make_figures.py"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = json.load(open('results.json')); BEST = R['best']; CLASSES = ['NORMAL', 'WARNING', 'CRITICAL', 'SENSOR_FAULT']
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#c3c2b7'
STATUS = {0: None, 1: '#fab219', 2: '#d03b3b', 3: '#6250d6'}   # warning, critical, fault (violet, not a status hue)
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.dpi': 200,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.axisbelow': True})
os.makedirs('figures', exist_ok=True)
def save(fig, name): fig.tight_layout(); fig.savefig(f'figures/{name}.png', bbox_inches='tight'); plt.close(fig)

# 1. model comparison: test macro-F1, best highlighted, rule baseline as reference
names = ['Threshold rule', 'Logistic Regression', 'Gradient Boosting', 'Random Forest', 'Neural Network (MLP)']
f1 = [R[n]['macro_f1'] for n in names]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(names, f1, color=[BLUE if n == BEST else GRAY for n in names], height=0.6)
for i, v in enumerate(f1): ax.text(v + 0.01, i, f'{v:.3f}', va='center', color=INK2)
ax.set_xlim(0, 1.05); ax.set_xlabel('Macro F1 on 10 unseen test patients'); ax.grid(axis='y', visible=False)
ax.set_title('Model comparison (patient-wise hold-out)'); save(fig, 'model_comparison')

# 2. confusion matrices, row-normalised (recall per true class)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, n in zip(axes, ['Threshold rule', BEST]):
    cm = np.array(R[n]['cm'], float); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
    for i in range(4):
        for j in range(4): ax.text(j, i, f'{cmn[i, j]:.0%}', ha='center', va='center', color='white' if cmn[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(4), [c.replace('_', '\n') for c in CLASSES], fontsize=9); ax.set_yticks(range(4), [c.replace('_', '\n') for c in CLASSES], fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f"{n}  (macro F1 {R[n]['macro_f1']:.2f})")
save(fig, 'confusion_matrices')

# 3. per-class recall: rule vs best model
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(8, 3.8))
for k, (n, c) in enumerate([('Threshold rule', BLUE), (BEST, ORANGE)]):
    ax.bar(x + (k - 0.5) * w, R[n]['recall'], w - 0.03, color=c, label=n)
ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1.1); ax.set_ylabel('Recall'); ax.grid(axis='x', visible=False)
ax.legend(frameon=False, loc='upper right', ncol=2); ax.set_title('Per-class recall'); save(fig, 'per_class_recall')

# 4. persistence trade-off: false alarms/hour vs median detection delay, K = 1..4
fig, ax = plt.subplots(figsize=(7, 4.2))
for who, c, lab in [('rule', BLUE, 'Threshold rule'), ('model', ORANGE, BEST)]:
    pts = [(R['persistence'][k][who]['median_delay_s'], R['persistence'][k][who]['false_alarms_per_hour']) for k in '1234']
    ax.plot(*zip(*pts), '-o', color=c, lw=2, ms=8, label=lab, markeredgecolor=SURF, markeredgewidth=2)
    for k, (dx, fy) in zip('1234', pts): ax.annotate(f'K={k}', (dx, fy), xytext=(6, 6), textcoords='offset points', fontsize=9, color=INK2)
ax.set_xlabel('Median detection delay (s)'); ax.set_ylabel('False alarms per hour'); ax.set_ylim(0, None)
ax.legend(frameon=False); ax.set_title('Alert persistence: fewer false alarms, small delay cost'); save(fig, 'persistence_tradeoff')

# 5. permutation feature importance (top 10)
imp = R['importance'][:10][::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([a for a, _ in imp], [b for _, b in imp], color=BLUE, height=0.6); ax.grid(axis='y', visible=False)
ax.set_xlabel('Drop in macro F1 when shuffled'); ax.set_title('What the model looks at'); save(fig, 'feature_importance')

# 6. one unseen test patient: vitals with true labels shaded
pid = R['data'].get('test_patient_ids', [5])[0]
d = pd.read_csv('vitals_dataset.csv'); d = d[d.patient == pid]; hrs = d.t.values * 10 / 3600
fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
for ax, col, lab in zip(axes, ['hr', 'temp', 'motion'], ['Heart rate (bpm)', 'Temperature (°C)', 'Motion index']):
    ax.plot(hrs, d[col].values, color=INK2, lw=0.8); ax.set_ylabel(lab, fontsize=9); ax.grid(axis='x', visible=False)
    if col == 'temp': ax.set_ylim(34, 41)   # clip fault spikes so fever/hypothermia stay readable
    m = np.concatenate([[0], d.label.values, [0]]); edges = np.flatnonzero(np.diff(m) != 0)
    for a, b in zip(edges[:-1], edges[1:]):
        if STATUS.get(m[a + 1]): ax.axvspan(hrs[a], hrs[min(b, len(hrs) - 1)], color=STATUS[m[a + 1]], alpha=0.25, lw=0)
axes[0].set_title(f'Unseen test patient {pid}: 6 h of simulated wearable data'); axes[-1].set_xlabel('Hours')
axes[0].legend(handles=[Patch(color=STATUS[k], alpha=0.4, label=CLASSES[k]) for k in (1, 2, 3)], frameon=False, ncol=3, loc='upper right', fontsize=9)
save(fig, 'patient_trace')
print('wrote', sorted(os.listdir('figures')))
