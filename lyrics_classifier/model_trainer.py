from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from .config import TrainingConfig
from .text_preprocessor import TextPreprocessor
from .feature_extractor import TfidfFeatures, EmbeddingFeatures
from .evaluator import evaluate_predictions, save_confusion_matrix
from .version import __version__


@dataclass
class TrainedModel:
    pipeline: Any
    label_encoder: LabelEncoder
    classes_: List[str]
    model_name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    artifact_dir: Path


class ModelTrainer:
    def __init__(self, cfg: TrainingConfig, workdir: Path):
        self.cfg = cfg
        self.workdir = workdir
        self.reports_dir = workdir / "reports"
        self.artifacts_dir = workdir / "artifacts"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _build_candidates(self, use_embeddings: bool) -> List[Tuple[str, Any, bool]]:
        # (nome, estimador, needs_calibration)
        cands: List[Tuple[str, Any, bool]] = []
        if "nb_multinomial" in self.cfg.models and not use_embeddings:
            cands.append(("nb_multinomial", MultinomialNB(), False))
        if "nb_bernoulli" in self.cfg.models and not use_embeddings:
            cands.append(("nb_bernoulli", BernoulliNB(), False))
        if "logreg" in self.cfg.models:
            cands.append(("logreg", LogisticRegression(max_iter=1000, n_jobs=None), False))
        if "linear_svm" in self.cfg.models:
            cands.append(("linear_svm", LinearSVC(), True))
        if "random_forest" in self.cfg.models:
            cands.append(("random_forest", RandomForestClassifier(n_estimators=300, random_state=self.cfg.random_state), False))
        if "xgboost" in self.cfg.models and HAS_XGB:
            cands.append(("xgboost", XGBClassifier(n_estimators=400, max_depth=6, subsample=0.9, colsample_bytree=0.9, learning_rate=0.1, eval_metric="mlogloss", random_state=self.cfg.random_state), False))
        return cands

    def train(self, X_texts: List[str], y_labels: List[str]) -> TrainedModel:
        # Handle small/imbalanced datasets: only stratify if every class has at least 2 samples
        try:
            from collections import Counter
            counts = Counter(y_labels)
            stratify_vec = y_labels if all(c >= 2 for c in counts.values()) else None
        except Exception:
            stratify_vec = None
        X_train, X_test, y_train, y_test = train_test_split(
            X_texts,
            y_labels,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify_vec,
        )

        le = LabelEncoder()
        # Fit on all labels to avoid issues if a class ends up only in the test split
        le.fit(y_labels)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        use_emb = self.cfg.use_embeddings
        preproc = TextPreprocessor(
            language=self.cfg.language,
            lowercase=self.cfg.lowercase,
            remove_stopwords=self.cfg.remove_stopwords,
            use_lemmatization=self.cfg.use_lemmatization,
        )

        tfidf = TfidfFeatures(preprocessor=preproc, ngram_range=self.cfg.ngram_range, min_df=self.cfg.min_df, max_df=self.cfg.max_df, max_features=self.cfg.max_features)

        if use_emb:
            embed = EmbeddingFeatures(self.cfg.embedding_model)

        best = None
        best_f1 = -1.0
        best_art_dir = None

        for name, base_est, needs_cal in self._build_candidates(use_embeddings=use_emb):
            est = base_est
            if needs_cal and self.cfg.calibrate:
                # Use the largest feasible cv given the smallest class in training, otherwise skip calibration
                try:
                    from collections import Counter as _Counter
                    _counts = _Counter(y_train)
                    _min_count = min(_counts.values()) if _counts else 0
                    if _min_count >= 2:
                        _cv = min(3, _min_count)
                        est = CalibratedClassifierCV(base_est, method="sigmoid", cv=_cv)
                    else:
                        est = base_est
                except Exception:
                    est = base_est

            if use_emb:
                # embeddings -> clf
                pipe = Pipeline(steps=[("embed", embed), ("clf", est)])
                # Pipeline compatibility shim: use a wrapper providing fit/transform
                # We'll fit embeddings outside since sklearn pipeline expects fit/transform API
                # Instead of using pipeline, we handle manually for embeddings
                Xtr = embed.fit_transform(X_train)
                est.fit(Xtr, y_train_enc)
                Xte = embed.transform(X_test)
                y_pred_enc = est.predict(Xte)
                pipe_fitted = {"embed": embed, "clf": est, "type": "embeddings"}
            else:
                # tfidf -> clf
                vec, Xtr = tfidf.fit_transform(X_train)
                est.fit(Xtr, y_train_enc)
                Xte = tfidf.transform(vec, X_test)
                y_pred_enc = est.predict(Xte)
                pipe_fitted = {"vec": vec, "clf": est, "preproc": preproc, "tfidf": tfidf, "type": "tfidf"}

            y_pred = le.inverse_transform(y_pred_enc)
            report = evaluate_predictions(y_test, y_pred, labels=list(le.classes_))

            # Save interim report
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_id = f"{name}_{timestamp}"
            metrics_path = self.reports_dir / f"metrics_{run_id}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                f.write(report.to_json())
            # Confusion matrix
            save_confusion_matrix(
                np.array(report.confusion), report.labels, self.reports_dir / f"confusion_{run_id}.png", title=f"CM - {name}"
            )

            if report.f1_macro > best_f1:
                best_f1 = report.f1_macro
                best = (name, pipe_fitted, report)

        assert best is not None
        best_name, best_pipe, best_report = best

        # Export artifact
        artifact_dir = self._export_artifact(best_name, best_pipe, le, best_report)

        return TrainedModel(
            pipeline=best_pipe,
            label_encoder=le,
            classes_=list(le.classes_),
            model_name=best_name,
            config=self._cfg_dict(),
            metrics=json.loads(best_report.to_json()),
            artifact_dir=artifact_dir,
        )

    def _cfg_dict(self) -> Dict[str, Any]:
        return {
            "version": __version__,
            "config": self.cfg.__dict__,
            "time": time.time(),
        }

    def _export_artifact(self, model_name: str, pipe: Any, le: LabelEncoder, report) -> Path:
        import joblib
        from hashlib import sha1
        import json as _json

        payload = {
            "version": __version__,
            "model_name": model_name,
            "classes": list(le.classes_),
            "label_encoder": le,
            "pipeline_type": pipe.get("type"),
            "pipeline": pipe,
            "config": self._cfg_dict(),
        }
        blob = _json.dumps({"classes": payload["classes"], "model_name": model_name, "version": __version__}).encode()
        short = sha1(blob).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = self.artifacts_dir / f"{model_name}_{timestamp}_{short}"
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(payload, out_dir / "component.joblib")

        # Save metrics and config
        (out_dir / "config.json").write_text(json.dumps(self._cfg_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "metrics.json").write_text(report.to_json(), encoding="utf-8")
        # Copy VERSION
        (out_dir / "VERSION").write_text(__version__, encoding="utf-8")

        # Optional: export ONNX if feasible
        if self.cfg.export_onnx and pipe.get("type") == "tfidf":
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                import numpy as _np

                vec = pipe["tfidf"].build()
                # Fake-fit vectorizer with training texts length; ONNX export for full pipeline is non-trivial
                # Here we try to export only the classifier when possible
                clf = pipe["clf"]
                n_features = pipe["vec"].vocabulary_.__len__()
                initial_type = [("float_input", FloatTensorType([None, n_features]))]
                onx = convert_sklearn(clf, initial_types=initial_type)
                with open(out_dir / "model.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
            except Exception as e:
                (out_dir / "onnx_export_error.txt").write_text(str(e), encoding="utf-8")

        return out_dir
