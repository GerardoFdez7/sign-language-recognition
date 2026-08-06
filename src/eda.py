"""Gráficos y descriptores visuales para el análisis exploratorio."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def plot_class_distribution(inventory: pd.DataFrame, ax=None):
    """Grafica el número de imágenes de cada clase."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    counts = inventory["label"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, color="#2563EB", ax=ax)
    ax.axhline(counts.mean(), color="#DC2626", linestyle="--", label="media")
    ax.set(title="Distribución de clases", xlabel="Clase", ylabel="Imágenes")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return ax


def show_examples(
    inventory: pd.DataFrame,
    labels: tuple[str, ...] = ("A", "B", "M", "U", "V"),
    examples_per_label: int = 3,
    seed: int = 42,
):
    """Muestra varias observaciones por clase para estudiar variabilidad."""
    fig, axes = plt.subplots(
        len(labels), examples_per_label, figsize=(3 * examples_per_label, 2.8 * len(labels))
    )
    axes = np.atleast_2d(axes)
    for row_index, label in enumerate(labels):
        candidates = inventory[inventory["label"].str.upper() == label.upper()]
        if len(candidates) < examples_per_label:
            raise ValueError(f"No hay {examples_per_label} ejemplos para {label}")
        selected = candidates.sample(examples_per_label, random_state=seed + row_index)
        for column_index, sample in enumerate(selected.itertuples(index=False)):
            with Image.open(sample.path) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].set_title(label)
            axes[row_index, column_index].axis("off")
    fig.suptitle("Variabilidad dentro y entre clases", fontsize=15)
    fig.tight_layout()
    return fig


def show_confusion_candidates(
    inventory: pd.DataFrame,
    groups: tuple[tuple[str, ...], ...] = (("M", "N", "S"), ("U", "V", "R")),
    seed: int = 42,
):
    """Presenta juntas las formas manuales que requieren comparación cuidadosa."""
    labels = [label for group in groups for label in group]
    fig, axes = plt.subplots(len(groups), max(map(len, groups)), figsize=(10, 7))
    axes = np.atleast_2d(axes)
    for row_index, group in enumerate(groups):
        for column_index, label in enumerate(group):
            candidates = inventory[inventory["label"].str.upper() == label]
            sample = candidates.sample(1, random_state=seed + row_index + column_index).iloc[0]
            with Image.open(sample["path"]) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].set_title(label, fontsize=14)
            axes[row_index, column_index].axis("off")
    fig.suptitle("Grupos con diferencias manuales sutiles")
    fig.tight_layout()
    return fig


def image_statistics(
    inventory: pd.DataFrame,
    per_class: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Calcula brillo, contraste, saturación y densidad de bordes."""
    frames: list[pd.DataFrame] = []
    for _, group in inventory.groupby("label"):
        frames.append(group.sample(min(per_class, len(group)), random_state=seed))
    selected = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, object]] = []
    for sample in selected.itertuples(index=False):
        try:
            with Image.open(sample.path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
        except OSError:
            continue

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        saturation = np.divide(
            maximum - minimum,
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 0,
        )
        horizontal_gradient = np.abs(np.diff(gray, axis=1))
        vertical_gradient = np.abs(np.diff(gray, axis=0))
        edge_density = (
            (horizontal_gradient > 30).mean() + (vertical_gradient > 30).mean()
        ) / 2
        rows.append(
            {
                "path": sample.path,
                "label": sample.label,
                "brightness": float(gray.mean()),
                "contrast": float(gray.std()),
                "saturation": float(saturation.mean()),
                "edge_density": float(edge_density),
            }
        )
    return pd.DataFrame(rows)


def plot_visual_statistics(statistics: pd.DataFrame):
    """Compara la dispersión de descriptores por clase."""
    long = statistics.melt(
        id_vars=["path", "label"],
        value_vars=["brightness", "contrast", "saturation", "edge_density"],
        var_name="metric",
        value_name="value",
    )
    grid = sns.catplot(
        data=long,
        x="label",
        y="value",
        col="metric",
        col_wrap=2,
        kind="box",
        sharey=False,
        height=4,
        aspect=1.6,
        showfliers=False,
        color="#60A5FA",
    )
    grid.set_xticklabels(rotation=60)
    grid.set_axis_labels("Clase", "Valor")
    grid.fig.suptitle("Variación visual por clase", y=1.02)
    return grid
