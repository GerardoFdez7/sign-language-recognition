"""Descarga, inspección y particionado del conjunto ASL Alphabet."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .config import CONFIG, ProjectConfig


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASSES = tuple(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"])


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet"


def _safe_extract(archive: ZipFile, destination: Path) -> None:
    """Extrae el ZIP sin permitir rutas fuera del destino."""
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Ruta insegura dentro del ZIP: {member.filename}")
    archive.extractall(destination)


def download_dataset(destination: str | Path = "data/raw") -> Path:
    """Descarga y extrae la versión pública con la biblioteca estándar."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "asl-alphabet.zip"
    request = Request(DATASET_URL, headers={"User-Agent": "SignBridge/1.0"})

    print("Descargando ASL Alphabet; el archivo puede superar 1 GB...")
    with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)

    print("Extrayendo imágenes...")
    with ZipFile(archive_path) as archive:
        _safe_extract(archive, destination)
    archive_path.unlink()
    return destination


def _class_directory_score(path: Path) -> int:
    names = {item.name.lower() for item in path.iterdir() if item.is_dir()}
    return len(names.intersection({name.lower() for name in EXPECTED_CLASSES}))


def find_training_directory(root: str | Path) -> Path:
    """Encuentra la carpeta que contiene las 29 subcarpetas de entrenamiento."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta: {root}")

    candidates = [root, *(path for path in root.rglob("*") if path.is_dir())]
    scored = sorted(
        ((_class_directory_score(path), path) for path in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scored or scored[0][0] < 26:
        raise FileNotFoundError(
            "No se encontró una carpeta con las clases A-Z, del, nothing y space."
        )
    return scored[0][1]


def build_inventory(training_dir: str | Path) -> pd.DataFrame:
    """Crea una fila por imagen sin modificar los archivos de origen."""
    training_dir = Path(training_dir).resolve()
    rows: list[dict[str, object]] = []
    class_dirs = sorted((path for path in training_dir.iterdir() if path.is_dir()), key=lambda p: p.name)

    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "label": class_dir.name,
                    "extension": image_path.suffix.lower(),
                    "bytes": image_path.stat().st_size,
                }
            )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError(f"No se encontraron imágenes en {training_dir}")
    return inventory


def inspect_image_headers(
    inventory: pd.DataFrame,
    sample_size: int | None = None,
    seed: int = CONFIG.seed,
) -> pd.DataFrame:
    """Lee metadatos y detecta archivos que Pillow no puede verificar."""
    selected = inventory
    if sample_size is not None and sample_size < len(inventory):
        selected = inventory.sample(sample_size, random_state=seed)

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        result: dict[str, object] = {
            "path": row.path,
            "label": row.label,
            "width": np.nan,
            "height": np.nan,
            "mode": None,
            "format": None,
            "valid": False,
            "error": None,
        }
        try:
            with Image.open(row.path) as image:
                result.update(
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    format=image.format,
                )
                image.verify()
                result["valid"] = True
        except (OSError, UnidentifiedImageError) as exc:
            result["error"] = str(exc)
        rows.append(result)
    return pd.DataFrame(rows)


def balanced_subsample(
    inventory: pd.DataFrame,
    per_class: int = CONFIG.images_per_class,
    seed: int = CONFIG.seed,
) -> pd.DataFrame:
    """Toma como máximo la misma cantidad de archivos de cada clase."""
    minimum = int(inventory.groupby("label").size().min())
    target = min(per_class, minimum)
    sampled = (
        inventory.groupby("label", group_keys=False)
        .sample(n=target, random_state=seed)
        .sort_values(["label", "path"])
        .reset_index(drop=True)
    )
    return sampled


def stratified_split(
    sample: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> pd.DataFrame:
    """Asigna train/validation/test preservando la proporción de clases."""
    config.validate()
    train, remainder = train_test_split(
        sample,
        train_size=config.train_fraction,
        random_state=config.seed,
        shuffle=True,
        stratify=sample["label"],
    )
    relative_test = config.test_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=config.seed,
        shuffle=True,
        stratify=remainder["label"],
    )

    result = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    if result["path"].duplicated().any():
        raise AssertionError("Una imagen fue asignada a más de una partición")
    return result.sort_values(["split", "label", "path"]).reset_index(drop=True)


def save_manifests(split_data: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Guarda un CSV general y uno para cada partición."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in [("all", split_data), *split_data.groupby("split")]:
        destination = output_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        written.append(destination)
    return written


def assert_disjoint_splits(split_data: pd.DataFrame) -> None:
    """Comprueba exclusividad y presencia de clases en todas las particiones."""
    if split_data["path"].duplicated().any():
        raise AssertionError("Las particiones comparten archivos")
    expected = set(split_data["label"].unique())
    for split_name, frame in split_data.groupby("split"):
        missing = expected.difference(frame["label"].unique())
        if missing:
            raise AssertionError(f"Faltan clases en {split_name}: {sorted(missing)}")


def class_counts(frame: pd.DataFrame, group_columns: Iterable[str] = ("label",)) -> pd.DataFrame:
    """Resume conteos y proporciones para una o varias columnas."""
    columns = list(group_columns)
    counts = frame.groupby(columns).size().rename("count").reset_index()
    counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts
