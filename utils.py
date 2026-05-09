import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

def check_leakage(groups_train, groups_test):
    """Strikte Patient-Level Leakage-Prüfung"""
    overlap = set(groups_train) & set(groups_test)
    if len(overlap) > 0:
        raise ValueError(f"⚠️ Patient-Level Leakage erkannt! Überlappende Subjects: {overlap}")
    print("✅ Kein Patient-Level Leakage")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", path="reports/confusion_matrix.png"):
    """Speichert eine schöne Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Kein Tinnitus", "Tinnitus"])
    disp.plot(cmap="Blues")
    plt.title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix gespeichert: {path}")