from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import json
import numpy as np

from .version import __version__


@dataclass
class LyricsClassifier:
    classes_: List[str]
    pipeline: Any  # dict with either {vec, tfidf, preproc, clf, type='tfidf'} or {embed, clf, type='embeddings'}
    pipeline_type: str
    metadata: Dict[str, Any]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "LyricsClassifier":
        p = Path(artifact_dir)
        payload = joblib.load(p / "component.joblib")
        return cls(
            classes_=payload["classes"],
            pipeline=payload["pipeline"],
            pipeline_type=payload.get("pipeline_type", "tfidf"),
            metadata={
                "version": payload.get("version", __version__),
                "model_name": payload.get("model_name"),
                "config": payload.get("config"),
                "artifact_dir": str(p.resolve()),
            },
        )

    def predict(self, texts: List[str]) -> List[str]:
        if self.pipeline_type == "tfidf":
            vec = self.pipeline["vec"]
            tfidf = self.pipeline["tfidf"]
            preproc = self.pipeline["preproc"]
            clf = self.pipeline["clf"]
            X = tfidf.transform(vec, texts)
            y_enc = clf.predict(X)
        else:
            embed = self.pipeline["embed"]
            clf = self.pipeline["clf"]
            X = embed.transform(texts)
            y_enc = clf.predict(X)
        # classes_ correspondem à codificação usada no treino
        classes = self.classes_
        preds = [classes[i] for i in y_enc]
        return preds

    def predict_proba(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self.pipeline_type == "tfidf":
            vec = self.pipeline["vec"]
            tfidf = self.pipeline["tfidf"]
            clf = self.pipeline["clf"]
            X = tfidf.transform(vec, texts)
        else:
            embed = self.pipeline["embed"]
            clf = self.pipeline["clf"]
            X = embed.transform(texts)

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X)
            return prob.tolist()
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            # Converte para pseudo-probabilidade com softmax por estabilidade
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            prob = e / e.sum(axis=1, keepdims=True)
            return prob.tolist()
        else:
            return None

    def topk(self, texts: List[str], k: int = 3) -> List[List[Dict[str, float]]]:
        prob = self.predict_proba(texts)
        if prob is None:
            preds = self.predict(texts)
            return [[{"label": p, "prob": 1.0}] for p in preds]
        out = []
        classes = self.classes_
        for row in prob:
            arr = np.array(row)
            idx = np.argsort(-arr)[:k]
            out.append([{"label": classes[i], "prob": float(arr[i])} for i in idx])
        return out


def find_latest_artifact(base_dir: str | Path) -> Optional[Path]:
    base = Path(base_dir)
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest
