from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


@dataclass
class TextPreprocessor:
    language: str = "pt"  # "pt" ou "en"
    lowercase: bool = True
    remove_stopwords: bool = True
    use_lemmatization: bool = True

    def __post_init__(self):
        ensure_nltk_resources()
        lang = "portuguese" if self.language.startswith("pt") else "english"
        self._stop = set(stopwords.words(lang)) if self.remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if self.use_lemmatization else None

    def clean(self, text: str) -> str:
        t = text
        if self.lowercase:
            t = t.lower()
        # Remove caracteres que não sejam letras/números/básicos de pontuação
        t = re.sub(r"[^\w\s'áéíóúâêîôûãõç-]", " ", t, flags=re.IGNORECASE)
        # Normaliza espaços
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def tokenize(self, text: str) -> List[str]:
        # Tokenização simples por espaços após limpeza
        return [tok for tok in self.clean(text).split(" ") if tok]

    def normalize(self, tokens: List[str]) -> List[str]:
        toks = [t for t in tokens if t not in self._stop] if self._stop else tokens
        if self._lemmatizer:
            # Lemmatizer funciona melhor em inglês; em pt, atua de forma limitada
            toks = [self._lemmatizer.lemmatize(t) for t in toks]
        return toks

    def transform(self, texts: List[str]) -> List[str]:
        out = []
        for t in texts:
            toks = self.tokenize(t)
            toks = self.normalize(toks)
            out.append(" ".join(toks))
        return out
