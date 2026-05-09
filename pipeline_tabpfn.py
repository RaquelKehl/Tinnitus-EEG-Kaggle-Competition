import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tabpfn import TabPFNClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.load_features import load_and_merge_features
from src.utils import check_leakage, plot_confusion_matrix

@hydra.main(version_base=None, config_path=".", config_name="config")
def run_pipeline(cfg: DictConfig):
    X, y, groups = load_and_merge_features(cfg)
    
    outer_cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    auc_scores, acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = TabPFNClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        
        auc_scores.append(auc)
        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)

        check_leakage(groups[train_idx], groups[test_idx])

        print(f"Fold {fold+1:2d} | AUC: {auc:.4f} | Acc: {acc:.4f}")

    results = {
        "mean_auc": float(np.mean(auc_scores)),
        "std_auc": float(np.std(auc_scores)),
        "mean_acc": float(np.mean(acc_scores)),
        "mean_precision": float(np.mean(prec_scores)),
        "mean_recall": float(np.mean(rec_scores)),
        "mean_f1": float(np.mean(f1_scores))
    }

    print(f"\n=== PIPELINE B: TabPFN ===")
    print(f"Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"Mean Accuracy: {results['mean_acc']:.4f}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/results_tabpfn.json", "w") as f:
        json.dump(results, f, indent=2)

    # Permutation Importance (einmalig)
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean
    }).sort_values('importance_mean', ascending=False)
    importance.head(20).to_csv("reports/tabpfn_top20_importance.csv", index=False)

    return results

if __name__ == "__main__":
    run_pipeline()