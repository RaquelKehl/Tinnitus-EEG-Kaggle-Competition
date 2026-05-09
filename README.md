# Tinnitus EEG Kaggle Competition – Modellvergleich

Dieses Repository enthält **zwei parallele Pipelines** für einen fairen Vergleich zwischen:
- **Pipeline A**: XGBoost + LightGBM Ensemble
- **Pipeline B**: TabPFN (Tabular Foundation Model)

Ziel ist es, das Modell mit der höchsten Accuracy zu identifizieren und die finale `submission.csv` für die Kaggle-Competition zu generieren.

### Repo-Struktur
```text
tinnitus-comparison/
├── pipeline_xgboost_lightgbm.py   # Pipeline A
├── pipeline_tabpfn.py             # Pipeline B
├── decision_script.py             # Finale Entscheidung & Submission-Generierung
├── config.yaml
├── requirements.txt
├── src/
│   ├── load_features.py
│   └── utils.py
├── data/processed/                # ← Hier deine 5 Feature-Dateien + labels.csv ablegen
├── models/
└── reports/                       # Ergebnisse, Plots, JSONs
Voraussetzungen
Deine 5 Feature-Dateien (z.B. features_pow_freq_bands_rest.csv, etc.) und die labels.csv müssen im Ordner data/processed/ liegen.

Wichtig: Die labels.csv muss mindestens die Spalten subject_id und tinnitus (Target-Variable) enthalten.

Installation
Es wird empfohlen, ein virtuelles Environment (z.B. mit venv oder conda) zu verwenden. Installiere danach die Abhängigkeiten:

Bash
pip install -r requirements.txt
Ausführungsreihenfolge (wichtig!)
Führe die Skripte zwingend in dieser Reihenfolge aus:

Pipeline A ausführen:

Bash
python pipeline_xgboost_lightgbm.py
Pipeline B ausführen:

Bash
python pipeline_tabpfn.py
Finale Entscheidung treffen:

Bash
python decision_script.py
Das decision_script.py vergleicht beide Modelle automatisch und entscheidet:

Entweder wird Pipeline A (XGBoost + LightGBM) genommen, oder

es wird ein Bagging-Ensemble aus beiden Modellen empfohlen.

Zusätzlich generiert das Skript die finale submission.csv für den Kaggle-Upload.

Ergebnisse
Nach dem Durchlauf findest du im Ordner reports/ detaillierte Auswertungen:

results_xgboost_lightgbm.json & results_tabpfn.json

final_decision.json

Confusion Matrices

SHAP-Plots (XGBoost)

Top-20 Feature Importance (TabPFN)
