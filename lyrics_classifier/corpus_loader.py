from pathlib import Path
from typing import Tuple
import pandas as pd


def load_dataset(csv_path: str | Path, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {p}")
    df = pd.read_csv(p)
    if text_col not in df.columns or label_col not in df.columns:
        cols = ", ".join(df.columns)
        raise ValueError(f"Colunas ausentes. Esperado: {text_col},{label_col}. Encontradas: {cols}")
    X = df[text_col].astype(str).fillna("")
    y = df[label_col].astype(str)
    return X, y
