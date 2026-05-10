"""Visualizaciones y guardado de resultados del laboratorio."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_experiment_results(
    results: dict,
    output_dir: str | Path = "results",
) -> None:
    """
    Guarda los resultados completos de un experimento en JSON
    y una fila de resumen en el CSV acumulativo.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    target = results["target_name"]

    # JSON con todos los detalles del experimento
    json_path = output_dir / f"results_{target}.json"
    serializable = _make_serializable(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {json_path}")

    # CSV acumulativo con el resumen
    csv_path = output_dir / "summary.csv"
    file_exists = csv_path.exists()
    s = results["summary"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "target", "n_clases",
                "hl_mean", "hl_std",
                "ema_mean", "ema_std",
                "f1_mean", "f1_std",
            ])
        writer.writerow([
            target,
            len(results["classes"]),
            f"{s['mean_hamming_loss']:.4f}",
            f"{s['std_hamming_loss']:.4f}",
            f"{s['mean_exact_match']:.4f}",
            f"{s['std_exact_match']:.4f}",
            f"{s['mean_f1_macro']:.4f}",
            f"{s['std_f1_macro']:.4f}",
        ])
    print(f"Actualizado: {csv_path}")


def _make_serializable(obj):
    """Convierte recursivamente tipos numpy/torch a tipos nativos de Python."""

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def plot_uncertainty_heatmap(
    outer_folds_results: list[dict],
    target_name: str,
    classes: list,
    output_dir: str | Path = "plots",
) -> None:
    """
    Mapa de calor de varianza MC Dropout por fold y clase.

    Filas = folds externos, columnas = clases del experimento.
    Color mas intenso = mayor incertidumbre en esa clase/fold.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    n_folds = len(outer_folds_results)
    n_classes = len(classes)

    matrix = np.zeros((n_folds, n_classes))
    for i, fold_result in enumerate(outer_folds_results):
        variances = fold_result["uncertainty"]["mean_variance_per_class"]
        matrix[i] = variances

    fig, ax = plt.subplots(figsize=(max(4, n_classes * 1.5), max(3, n_folds * 0.8)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Varianza MC Dropout")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"Clase {c}" for c in classes])
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {r['outer_fold']}" for r in outer_folds_results])

    for i in range(n_folds):
        for j in range(n_classes):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=8)

    ax.set_title(
        f"Incertidumbre MC Dropout por fold y clase — {target_name}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = output_dir / f"uncertainty_heatmap_{target_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {path}")


def plot_class_distribution(
    dataframe,
    target_columns: list[str],
    output_dir: str | Path = "plots",
) -> None:
    """
    Genera un grafico de barras por cada columna objetivo mostrando
    cuantas muestras hay por clase (desbalance de clases).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    n_targets = len(target_columns)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    for ax, target in zip(axes, target_columns):
        if target not in dataframe.columns:
            ax.set_visible(False)
            continue

        counts = dataframe[target].value_counts().sort_index()
        classes = [str(c) for c in counts.index]
        values = counts.values

        bars = ax.bar(classes, values, color="steelblue", edgecolor="white")
        ax.set_title(target, fontsize=12, fontweight="bold")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Muestras")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Distribución de clases por experimento", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {path}")


def plot_loss_curves(
    outer_folds_results: list[dict],
    target_name: str,
    output_dir: str | Path = "plots",
) -> None:
    """
    Grafica la curva de loss de entrenamiento por fold externo.

    Cada fold tiene su propia linea. Permite ver si el modelo converge
    de forma consistente entre folds.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for fold_result in outer_folds_results:
        fold_idx = fold_result["outer_fold"]
        losses = fold_result["train_losses"]
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", markersize=3, label=f"Fold {fold_idx}")

    ax.set_title(f"Curva de Loss — {target_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = output_dir / f"loss_curve_{target_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {path}")
