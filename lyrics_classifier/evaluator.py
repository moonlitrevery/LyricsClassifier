from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class MetricsReport:
    accuracy: float
    f1_macro: float
    report: Dict[str, Dict[str, float]]
    labels: List[str]
    confusion: List[List[int]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "accuracy": self.accuracy,
                "f1_macro": self.f1_macro,
                "report": self.report,
                "labels": self.labels,
                "confusion": self.confusion,
            },
            ensure_ascii=False,
            indent=2,
        )


def evaluate_predictions(y_true, y_pred, labels: Optional[List[str]] = None) -> MetricsReport:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    if labels is None:
        labels = sorted(list({*y_true, *y_pred}))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return MetricsReport(accuracy=acc, f1_macro=f1m, report=rep, labels=list(labels), confusion=cm.tolist())


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, title: str = "Matriz de Confusão"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
