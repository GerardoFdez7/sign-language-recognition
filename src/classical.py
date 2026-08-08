"""Descriptor HOG y modelos SVM para la comparación clásica."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from .neural import SafeAugmentation


def hog_descriptor(image: Image.Image, cell_size: int = 8, bins: int = 9) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    bin_indices = np.floor(orientation / (180.0 / bins)).astype(int) % bins
    cells_y = gray.shape[0] // cell_size
    cells_x = gray.shape[1] // cell_size
    histograms = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    for row in range(cells_y):
        for column in range(cells_x):
            row_slice = slice(row * cell_size, (row + 1) * cell_size)
            column_slice = slice(column * cell_size, (column + 1) * cell_size)
            histograms[row, column] = np.bincount(
                bin_indices[row_slice, column_slice].ravel(),
                weights=magnitude[row_slice, column_slice].ravel(),
                minlength=bins,
            )
    blocks: list[np.ndarray] = []
    for row in range(cells_y - 1):
        for column in range(cells_x - 1):
            block = histograms[row : row + 2, column : column + 2].ravel()
            blocks.append(block / np.sqrt(np.dot(block, block) + 1e-6))
    return np.concatenate(blocks).astype(np.float32)


def extract_hog_features(
    manifest: pd.DataFrame,
    augment: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    transformation = SafeAugmentation(seed) if augment else None
    total = len(manifest)
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        with Image.open(row.path) as source:
            image = source.convert("RGB")
        if transformation is not None:
            image = transformation(image)
        features.append(hog_descriptor(image))
        labels.append(row.label)
        if position % 2_000 == 0 or position == total:
            print(f"HOG: {position:,}/{total:,}")
    return np.stack(features), np.asarray(labels)


def train_linear_svm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    c_value: float,
    output_path: Path,
    augmented_features: np.ndarray | None = None,
    augmented_labels: np.ndarray | None = None,
) -> tuple[LinearSVC, dict[str, float]]:
    if augmented_features is not None and augmented_labels is not None:
        train_features = np.concatenate([train_features, augmented_features])
        train_labels = np.concatenate([train_labels, augmented_labels])
    started = time.perf_counter()
    model = LinearSVC(C=c_value, dual="auto", max_iter=5_000, random_state=42)
    model.fit(train_features, train_labels)
    elapsed = time.perf_counter() - started
    validation_predictions = model.predict(validation_features)
    test_predictions = model.predict(test_features)
    metrics = {
        "validation_macro_f1": float(
            f1_score(validation_labels, validation_predictions, average="macro")
        ),
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "macro_f1": float(f1_score(test_labels, test_predictions, average="macro")),
        "elapsed_seconds": elapsed,
        "epochs_ran": 1,
        "parameters": int(model.coef_.size + model.intercept_.size),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "metrics": metrics, "c_value": c_value},
        output_path,
    )
    return model, metrics

