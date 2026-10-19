"""Simulated wearable-vitals dataset: heart rate, body temperature, motion.
Labels: 0 NORMAL, 1 WARNING, 2 CRITICAL, 3 SENSOR_FAULT.  10 s sampling."""
import numpy as np, pandas as pd

N_PATIENTS = 30; N_SAMPLES = 2160   # 6 hours per patient
LABELS = {0: 'NORMAL', 1: 'WARNING', 2: 'CRITICAL', 3: 'SENSOR_FAULT'}

def rule(hr, t):  # threshold baseline on the latest sample (textbook adult ranges)
    hr, t = np.asarray(hr), np.asarray(t); out = np.zeros(len(hr), int)
    out[(hr < 60) | (hr > 100) | (t < 36.1) | (t > 37.5)] = 1
    out[(hr < 40) | (hr > 140) | (t < 35.0) | (t > 39.0)] = 2
    out[(hr <= 0) | (hr > 220) | (t < 30) | (t > 43)] = 3
    return out

def simulate_patient(pid, rng):
    n = N_SAMPLES
    base_hr = float(np.clip(rng.normal(70, 6), 58, 84)); base_t = rng.normal(36.7, 0.15); bias_t = rng.normal(0, 0.12)
    # ---- activity timeline ----
    state = np.zeros(n, int); i = 0
    while i < n:
        d = int(rng.integers(30, 180)); s = rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]); state[i:i+d] = s; i += d
    motion = np.where(state == 0, np.abs(rng.normal(0.05, 0.03, n)), np.where(state == 1, rng.normal(0.35, 0.08, n), rng.normal(0.8, 0.12, n)))
    hr_gain = np.where(state == 0, 0, np.where(state == 1, rng.normal(18, 6), rng.normal(45, 12)))
    # ---- events (non-overlapping) ----
    ev_type = np.array(['none'] * n, dtype=object); label = np.zeros(n, int)
    hr_target = np.full(n, np.nan); t_target = np.full(n, np.nan)
    kinds = [('tachy_w', 1), ('tachy_c', 2), ('brady_w', 1), ('brady_c', 2), ('fever_w', 1), ('fever_c', 2), ('hypo_w', 1), ('hypo_c', 2)]
    occupied = np.zeros(n, bool)
    def place(length):
        for _ in range(50):
            a = int(rng.integers(60, n - length - 10))
            if not occupied[max(0, a-20):a+length+20].any(): return a
        return None
    for _ in range(int(rng.integers(5, 9))):
        k, sev = kinds[int(rng.integers(0, len(kinds)))]; L = int(rng.integers(30, 120)); a = place(L)
        if a is None: continue
        b = a + L; occupied[a-20:b+20] = True; ev_type[a:b] = k; label[a:b] = sev
        if rng.random() < 0.7: state[a:b] = 0; motion[a:b] = np.abs(rng.normal(0.05, 0.03, L)); hr_gain[a:b] = 0
        else: state[a:b] = 1; motion[a:b] = rng.normal(0.35, 0.08, L); hr_gain[a:b] = 0
        if k == 'tachy_w': hr_target[a:b] = rng.uniform(105, 135)
        if k == 'tachy_c': hr_target[a:b] = rng.uniform(145, 185)
        if k == 'brady_w': hr_target[a:b] = rng.uniform(46, 58)
        if k == 'brady_c': hr_target[a:b] = rng.uniform(28, 40)
        if k == 'fever_w': t_target[a:b] = rng.uniform(37.6, 38.8)
        if k == 'fever_c': t_target[a:b] = rng.uniform(39.1, 40.4)
        if k == 'hypo_w': t_target[a:b] = rng.uniform(35.2, 35.9)
        if k == 'hypo_c': t_target[a:b] = rng.uniform(33.5, 34.9)
    # ---- latent physiology with first-order lag ----
    hr = np.zeros(n); t = np.zeros(n); hr[0] = base_hr; t[0] = base_t
    for i in range(1, n):
        tgt_h = base_hr + hr_gain[i] if np.isnan(hr_target[i]) else hr_target[i]
        tgt_t = base_t + (0.1 if state[i] == 1 else 0.4 if state[i] == 2 else 0.0) if np.isnan(t_target[i]) else t_target[i]
        if not np.isnan(t_target[i]): tgt_h = max(min(tgt_h + 8 * (t_target[i] - 37.0), 130), 40) if np.isnan(hr_target[i]) else tgt_h
        hr[i] = hr[i-1] + 0.15 * (tgt_h - hr[i-1]); t[i] = t[i-1] + (0.08 if not np.isnan(t_target[i]) else 0.02) * (tgt_t - t[i-1])
    hr_obs = hr + rng.normal(0, 1.8, n); t_obs = t + bias_t + rng.normal(0, 0.05, n)
    # ---- sensor faults ----
    for _ in range(int(rng.integers(4, 7))):
        L = int(rng.integers(3, 25)); a = place(L)
        if a is None: continue
        b = a + L; occupied[a-20:b+20] = True; kind = rng.choice(['dropout', 'spike', 'stuck'])
        if kind == 'dropout':
            hr_obs[a:b] = 0; t_obs[a:b] = np.clip(t_obs[a] - np.cumsum(rng.uniform(0.5, 2.0, L)), rng.uniform(24, 31), None)
        elif kind == 'spike':
            idx = rng.choice(np.arange(a, b), size=min(3, L), replace=False); hr_obs[idx] = rng.uniform(190, 255, len(idx)); t_obs[idx] = rng.uniform(41, 48, len(idx))
        else:
            hr_obs[a:b] = hr_obs[a]; t_obs[a:b] = t_obs[a]
        label[a:b] = 3; ev_type[a:b] = 'fault_' + kind
    return pd.DataFrame(dict(patient=pid, t=np.arange(n), motion=np.round(motion, 3), hr=np.round(hr_obs).astype(int), temp=np.round(t_obs, 2), label=label, event=ev_type))

def build(seed=7):
    rng = np.random.default_rng(seed)
    return pd.concat([simulate_patient(p, rng) for p in range(N_PATIENTS)], ignore_index=True)

if __name__ == '__main__':
    df = build(); df.to_csv('vitals_dataset.csv', index=False); print(df.shape); print(df.label.map(LABELS).value_counts())
