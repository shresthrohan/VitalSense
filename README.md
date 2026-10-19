# AI-Based Health Monitoring System

A wearable-vitals monitor that classifies every 10-second sample of heart rate, body temperature and motion as
**NORMAL**, **WARNING**, **CRITICAL** or **SENSOR_FAULT**, and raises alerts only when the condition persists.
It is compared against a textbook threshold rule on patients it never saw during training.



https://github.com/user-attachments/assets/7332b819-fb78-4998-ad65-891150b7faef


## Run it

Python 3.10+:

    pip install -r requirements.txt
    python generate_data.py      # simulated vitals for 30 virtual patients -> vitals_dataset.csv
    python train_models.py       # trains/compares models, saves health_model.joblib and results.json (~1 min)
    python infer_demo.py 5       # streams test patient 5 through the saved model and prints alerts
    python test_pipeline.py      # smoke check: live features == training features, alert logic
    python make_figures.py       # report charts -> figures/
    python export_dashboard.py   # replays all test patients -> dashboard/data.js

Then open `dashboard/index.html` in a browser (no server needed).

## Deliverables

- `AI_Health_Monitoring_Project_Report.pptx`: 15-slide project report (problem, method, results, dashboard, limitations).
- `brag-output/brag.mp4`: 21-second launch video; `brag-output/share-copy.txt` has a caption. Rebuild with `brag-output/work/render.py` and `audio.py`.
- `figures/`: report charts and a dashboard screenshot.

## How it works

| Step | File | What happens |
|---|---|---|
| Data | `generate_data.py` | 30 patients x 6 h. Activity bouts (rest/walk/run), injected tachy/brady-cardia, fever, hypothermia (warning and critical grades) and sensor faults (dropout, spike, stuck value). Heart rate and temperature follow first-order physiological lag plus sensor noise and a per-patient temperature bias. |
| Features | `train_models.py` | Sliding window of 6 samples (1 min). 15 features: last, mean, std, min, max and slope of HR and temperature, last HR change, mean/max motion. |
| Models | `train_models.py` | Threshold rule (baseline), Logistic Regression, MLP, Random Forest, Gradient Boosting. Patient-wise split: 20 train / 10 test patients, 5-fold GroupKFold CV on train picks the deployed model. Also an Isolation Forest (unsupervised), a no-motion ablation, permutation importance, latency and alert-persistence analysis. |
| Live loop | `infer_demo.py` | Rolling buffer, one prediction per sample. Alerts after K=3 consecutive abnormal windows and again only if the episode escalates (e.g. WARNING to CRITICAL). |
| Dashboard | `dashboard/` | Ward-monitor style replay of the 10 test patients: live strip chart, AI vs rule verdicts against ground truth, alert log marking true and false alarms, evaluation table. |

## Results (10 unseen test patients, 60 hours)

| Model | Accuracy | Macro F1 | False alarms / h | Critical events caught |
|---|---|---|---|---|
| Threshold rule | 74.7% | 0.683 | 9.6 | 100% |
| Logistic Regression | 36.6% | 0.367 | 4.5 | 81% |
| Gradient Boosting | 92.0% | 0.868 | 10.4 | 100% |
| Random Forest | 94.4% | 0.895 | 6.7 | 100% |
| **Neural Network (MLP), deployed** | **95.4%** | **0.889** | **3.8** | **100%** |

- With 3-window persistence: 1.1 false alarms/h (rule: 3.0), median detection delay 30 s (rule: 50 s), every event caught.
- Healthy exercise flagged as abnormal: 0.2% by the AI vs 94% by the rule. Motion context is what the rule lacks; removing motion features drops macro F1 to 0.843.
- Sensor faults: the AI catches 66% of spike and 37% of stuck-value faults; the rule catches 23% and 0%.
- 0.4 ms per prediction, 86 KB model.

Figures for these are in `figures/`. Full numbers in `results.json`.

## Limitations

The data is simulated, so reported metrics are optimistic and are not clinical evidence. Labels come from the
simulator, the faults are synthetic, and the model would need re-validation on real wearable recordings
(e.g. PhysioNet/MIMIC waveform data) before any clinical use. Stuck-value sensor faults remain the weakest class.
