"""Funciones para cargar datos y construir el Dataset de PyTorch."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_COLUMNS


def load_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Carga un DataFrame desde CSV o SAV."""

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".sav":
        try:
            import pyreadstat
        except ImportError as error:
            raise ImportError(
                "Para leer archivos .sav debes instalar pyreadstat."
            ) from error

        dataframe, _ = pyreadstat.read_sav(path)
        return dataframe

    raise ValueError(
        "Formato no soportado. Usa un archivo .csv o .sav dentro de dataset/."
    )


def validate_feature_columns(
    dataframe: pd.DataFrame, feature_columns: list[str] | None = None
) -> None:
    """Verifica que las columnas de entrada existan en el dataset."""

    columns = feature_columns or FEATURE_COLUMNS
    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "Faltan columnas de entrada en el dataset: "
            + ", ".join(missing_columns)
        )


def build_input_matrix(
    dataframe: pd.DataFrame, feature_columns: list[str] | None = None
) -> np.ndarray:
    """Extrae la matriz X usando las columnas de entrada del laboratorio."""

    columns = feature_columns or FEATURE_COLUMNS
    validate_feature_columns(dataframe, columns)
    return dataframe[columns].astype("float32").to_numpy()


class CognitiveMultiLabelDataset(Dataset):
    """Dataset simple de PyTorch para entradas tabulares y targets multilabel."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]
