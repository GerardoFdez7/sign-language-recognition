"""Canal de imágenes con Pillow y NumPy, sin dependencias de entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image

from .config import CONFIG, ProjectConfig


def load_rgb_image(
    path: str | Path,
    label: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Decodifica JPEG, redimensiona y normaliza a [0, 1]."""
    height, width = image_size
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        values = np.asarray(image, dtype=np.float32) / 255.0
    return values, label


@dataclass
class ImageBatchDataset:
    """Iterable que carga lotes de imágenes bajo demanda."""

    manifest: pd.DataFrame
    class_names: list[str]
    training: bool
    config: ProjectConfig

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.manifest))
        if self.training:
            generator = np.random.default_rng(self.config.seed)
            generator.shuffle(indices)

        label_lookup = {name: index for index, name in enumerate(self.class_names)}
        for start in range(0, len(indices), self.config.batch_size):
            batch_indices = indices[start : start + self.config.batch_size]
            images: list[np.ndarray] = []
            labels: list[int] = []
            for index in batch_indices:
                row = self.manifest.iloc[int(index)]
                label = label_lookup[str(row["label"])]
                image, encoded_label = load_rgb_image(
                    row["path"],
                    label,
                    (self.config.image_height, self.config.image_width),
                )
                images.append(image)
                labels.append(encoded_label)
            yield np.stack(images), np.asarray(labels, dtype=np.int64)


def dataset_from_manifest(
    manifest: pd.DataFrame,
    class_names: list[str],
    training: bool,
    config: ProjectConfig = CONFIG,
):
    """Construye un iterable sin cargar todo el conjunto en memoria."""
    if manifest.empty:
        raise ValueError("El manifiesto no contiene imágenes")
    missing = set(manifest["label"]).difference(class_names)
    if missing:
        raise ValueError(f"Hay etiquetas sin codificación: {sorted(missing)}")
    return ImageBatchDataset(
        manifest=manifest.reset_index(drop=True),
        class_names=class_names,
        training=training,
        config=config,
    )


def inspect_preprocessing(dataset) -> dict[str, object]:
    """Devuelve forma, tipo y rango del primer lote como control de calidad."""
    images, labels = next(iter(dataset))
    return {
        "image_shape": tuple(images.shape),
        "label_shape": tuple(labels.shape),
        "dtype": images.dtype.name,
        "minimum": float(images.min()),
        "maximum": float(images.max()),
    }
