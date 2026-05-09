# Tinnitus-EEG-Kaggle-Competition

Dieses Repository enthält **zwei parallele Pipelines** für einen fairen Vergleich zwischen:
- **Pipeline A**: XGBoost + LightGBM Ensemble
- **Pipeline B**: TabPFN (Tabular Foundation Model)

Ziel ist es, das Modell mit der höchsten Accuracy für die Kaggle-Submission zu finden.

### Repo-Struktur
```

tinnitus-comparison/ ├── pipeline_xgboost_lightgbm.py # Pipeline A ├── pipeline_tabpfn.py # Pipeline B ├── decision_script.py # Finale Entscheidung ├── config.yaml ├── requirements.txt ├── src/ │ ├── load_features.py │ └── utils.py ├── data/processed/ # ← Hier deine 5 Feature-Dateien + labels.csv ├── models/ └── reports/ # Ergebnisse, Plots, JSONs
text

````
### Installation

```bash
pip install -r requirements.txt
````

Ausführungsreihenfolge (wichtig!)

1. Pipeline A ausführen
   Bash

   ```
   python pipeline_xgboost_lightgbm.py
   ```

2. Pipeline B ausführen
   Bash

   ```
   python pipeline_tabpfn.py
   ```

3. Finale Entscheidung treffen
   Bash

   ```
   python decision_script.py
   ```

Das decision_script.py vergleicht beide Modelle automatisch und entscheidet:

* Entweder wird Pipeline A (XGBoost + LightGBM) genommen, oder

* Es wird ein Bagging-Ensemble aus beiden Modellen empfohlen.

Ergebnisse
Nach dem Durchlauf findest du in reports/:

* results_xgboost_lightgbm.json

* results_tabpfn.json

* final_decision.json

* Confusion Matrices

* SHAP-Plots (XGBoost)

* Top-20 Feature Importance (TabPFN)

Voraussetzungen

* Deine 5 Feature-Dateien (features_*.csv / .h5) und labels.csv müssen im Ordner data/processed/ liegen.

* labels.csv muss mindestens die Spalten subject_id und tinnitus enthalten.
