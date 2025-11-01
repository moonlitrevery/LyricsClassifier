from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from .text_preprocessor import TextPreprocessor


@dataclass
class TfidfFeatures:
    preprocessor: TextPreprocessor
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: Optional[int] = 50000

    def build(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            preprocessor=None,
            tokenizer=None,
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
        )

    def fit_transform(self, texts: List[str]):
        clean_texts = self.preprocessor.transform(texts)
        vec = self.build()
        X = vec.fit_transform(clean_texts)
        return vec, X

    def transform(self, vec: TfidfVectorizer, texts: List[str]):
        clean_texts = self.preprocessor.transform(texts)
        return vec.transform(clean_texts)


class EmbeddingFeatures:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]):
        X = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return self, np.asarray(X)

    def transform(self, texts: List[str]):
        X = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(X)
