from pathlib import Path
from typing import Tuple
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def _read_table(p: Path) -> pd.DataFrame:
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato de dataset não suportado: {suffix}. "
            "Use CSV, XLSX/XLS ou Parquet."
        )
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise AssertionError("Extensão não tratada")


def load_dataset(data_path: str | Path, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {p}")
    df = _read_table(p)
    if text_col not in df.columns or label_col not in df.columns:
        cols = ", ".join(df.columns)
        raise ValueError(f"Colunas ausentes. Esperado: {text_col},{label_col}. Encontradas: {cols}")
    X = df[text_col].astype(str).fillna("")
    y = df[label_col].astype(str)
    return X, y
