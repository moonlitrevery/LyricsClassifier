from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    language: str = "pt"  # "pt" ou "en"
    use_lemmatization: bool = True
    remove_stopwords: bool = True
    lowercase: bool = True
    use_embeddings: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: Optional[int] = 50000

    models: List[str] = field(
        default_factory=lambda: [
            "nb_multinomial",
            "nb_bernoulli",
            "logreg",
            "linear_svm",
            "random_forest",
            "xgboost",
        ]
    )

    test_size: float = 0.2
    random_state: int = 42
    calibrate: bool = True  # calibração de probas p/ modelos sem predict_proba

    export_onnx: bool = False


@dataclass
class InferenceConfig:
    top_k: int = 3
