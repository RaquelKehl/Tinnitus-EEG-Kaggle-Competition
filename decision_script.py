import json
from pathlib import Path

def load_results():
    """Lädt die Ergebnisse der beiden Pipelines"""
    with open("reports/results_xgboost_lightgbm.json", "r") as f:
        res_a = json.load(f)
    
    with open("reports/results_tabpfn.json", "r") as f:
        res_b = json.load(f)
    
    return res_a, res_b

def make_decision():
    res_a, res_b = load_results()
    
    mean_acc_a = res_a["mean_acc"]
    mean_acc_b = res_b["mean_acc"]
    diff = mean_acc_a - mean_acc_b
    
    print("\n" + "="*60)
    print("FINAL DECISION – TINNITUS KAGGLE COMPETITION")
    print("="*60)
    print(f"Pipeline A (XGBoost + LightGBM) → Accuracy: {mean_acc_a:.4f}")
    print(f"Pipeline B (TabPFN)            → Accuracy: {mean_acc_b:.4f}")
    print(f"Differenz (A - B):              {diff:.4f}")
    print("-" * 60)

    if diff > 0.005:
        print("✅ Entscheidung: Pipeline A gewinnt klar")
        print("   → XGBoost + LightGBM wird für die finale Kaggle-Submission verwendet")
        chosen = "A"
    else:
        print("✅ Entscheidung: Differenz ≤ 0.005 → Bagging-Ensemble wird empfohlen")
        print("   → Soft-Voting / Bagging Ensemble aus beiden Modellen wird für Kaggle verwendet")
        chosen = "ENSEMBLE"
    
    print("\nEmpfohlene finale Submission:", chosen)
    
    # Speichern der Entscheidung
    decision = {
        "chosen_pipeline": chosen,
        "accuracy_a": mean_acc_a,
        "accuracy_b": mean_acc_b,
        "difference": float(diff),
        "recommendation": "Use Pipeline A" if chosen == "A" else "Use Bagging Ensemble"
    }
    
    Path("reports").mkdir(exist_ok=True)
    with open("reports/final_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    
    print("\nEntscheidung wurde in 'reports/final_decision.json' gespeichert.")

if __name__ == "__main__":
    make_decision()