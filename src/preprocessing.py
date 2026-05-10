"""Funciones para preparar el target multilabel del experimento activo."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import TARGET_COLUMNS
from data_loader import build_input_matrix


def encode_target_as_one_hot(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Toma una sola columna objetivo y la convierte a formato one-hot.

    Ejemplo:
    Si la clase original es 2 y el experimento tiene tres clases,
    entonces el vector queda como [0, 1, 0].
    """

    if target_name not in TARGET_COLUMNS:
        raise ValueError(
            f"Target invalido: {target_name}. Debe ser uno de {TARGET_COLUMNS}."
        )

    if target_name not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_name} no existe en el dataset.")

    y_raw = dataframe[target_name].astype(int)
    classes = sorted(y_raw.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}

    y_idx = y_raw.map(class_to_idx).to_numpy()
    y_encoded = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    y_encoded[np.arange(len(y_idx)), y_idx] = 1.0

    return y_encoded, classes, class_to_idx


def prepare_experiment_data(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """Prepara X e Y para un experimento puntual."""

    X = build_input_matrix(dataframe)
    Y, classes, class_to_idx = encode_target_as_one_hot(dataframe, target_name)
    return X, Y, classes, class_to_idx


def split_for_validation(
    Y: np.ndarray, n_splits: int, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """API publica para crear folds estratificados del laboratorio."""

    return build_nested_splits(Y, n_splits=n_splits, random_state=random_state)


def build_stratification_labels(Y: np.ndarray) -> np.ndarray:
    """
    Construye etiquetas auxiliares para estratificar los folds.

    En esta version minima del laboratorio, cada experimento toma una sola
    columna objetivo y la codifica como one-hot. Eso permite estratificar
    usando la clase activa de cada fila.
    """

    Y = np.asarray(Y, dtype=np.int64)

    if Y.ndim != 2:
        raise ValueError("Y debe ser una matriz 2D para construir folds estratificados.")

    label_counts = Y.sum(axis=1)
    if np.any(label_counts <= 0):
        raise ValueError("Cada muestra debe activar al menos una etiqueta en Y.")

    if np.all(label_counts == 1):
        return np.argmax(Y, axis=1)

    return np.asarray(["|".join(map(str, row.tolist())) for row in Y], dtype=object)


def build_nested_splits(
    Y: np.ndarray, n_splits: int, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera folds estratificados para la validacion del laboratorio.

    Si luego el curso adopta una estrategia mas avanzada de estratificacion
    multilabel, este es el lugar natural para reemplazar la implementacion.
    """

    labels = build_stratification_labels(Y)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) < 2:
        raise ValueError(
            "Se requieren al menos dos clases distintas para realizar validacion."
        )

    if counts.min() < n_splits:
        raise ValueError(
            "No se puede crear una validacion estratificada con "
            f"{n_splits} folds porque la clase menos frecuente solo tiene "
            f"{counts.min()} muestras."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    dummy_inputs = np.zeros(len(labels), dtype=np.float32)
    return list(splitter.split(dummy_inputs, labels))
