"""Metricas para evaluar salidas multilabel."""

import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades a etiquetas binarias usando un umbral fijo."""

    return (probabilities >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Hamming loss.

    La idea es simple:
    - comparar cada componente del vector real con la predicha,
    - contar cuantas componentes quedan distintas,
    - y promediar ese error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular Hamming loss."
        )

    mistakes = np.not_equal(y_true, y_pred)
    return float(mistakes.mean())


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraccion de muestras donde el vector de etiquetas coincide exactamente.

    Es la metrica mas estricta: un solo bit incorrecto cuenta como error total.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular Exact Match."
        )

    exact_matches = np.all(y_true == y_pred, axis=1)
    return float(exact_matches.mean())


def f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro F1 multilabel: promedio del F1 calculado por cada clase.

    Para cada clase se calcula TP, FP y FN, luego se promedia el F1
    de todas las clases (macro averaging).
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular F1 multilabel."
        )

    n_classes = y_true.shape[1]
    f1_scores = []

    for class_idx in range(n_classes):
        tp = float(np.sum((y_pred[:, class_idx] == 1) & (y_true[:, class_idx] == 1)))
        fp = float(np.sum((y_pred[:, class_idx] == 1) & (y_true[:, class_idx] == 0)))
        fn = float(np.sum((y_pred[:, class_idx] == 0) & (y_true[:, class_idx] == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))
