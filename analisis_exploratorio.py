"""Análisis exploratorio y preparación del conjunto ASL Alphabet.

Uso:
    python analisis_exploratorio.py --data-dir "ruta/al/dataset"
    python analisis_exploratorio.py --download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CONFIG
from src.data import (
    assert_disjoint_splits,
    balanced_subsample,
    build_inventory,
    download_dataset,
    find_training_directory,
    inspect_image_headers,
    save_manifests,
    stratified_split,
)
from src.eda import (
    image_statistics,
    plot_class_distribution,
    plot_visual_statistics,
    show_confusion_candidates,
    show_examples,
)
from src.model_catalog import model_plan_table
from src.preprocessing import dataset_from_manifest, inspect_preprocessing


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "artifacts" / "figures"
TABLE_DIR = ROOT / "artifacts" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, help="Ruta local al conjunto")
    source.add_argument(
        "--download",
        action="store_true",
        help="Descargar el conjunto público de Kaggle",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=CONFIG.images_per_class,
        help="Máximo de imágenes seleccionadas por clase",
    )
    parser.add_argument(
        "--header-sample",
        type=int,
        default=1_000,
        help="Cantidad de cabeceras que se verificarán",
    )
    parser.add_argument(
        "--visual-sample",
        type=int,
        default=100,
        help="Imágenes por clase para estadísticas visuales",
    )
    return parser.parse_args()


def print_table(value: pd.DataFrame | pd.Series) -> None:
    print(value.to_string())


def find_local_dataset(explicit_path: Path | None) -> Path | None:
    """Busca una descarga local en ubicaciones conocidas."""
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    candidates = [ROOT / "data" / "raw"]
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "grassknoted" / "asl-alphabet"
    if cache_root.exists():
        candidates.extend(sorted(cache_root.glob("versions/*"), reverse=True))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            find_training_directory(candidate)
            return candidate
        except FileNotFoundError:
            continue
    return None


def save_figure(figure, filename: str) -> None:
    destination = FIGURE_DIR / filename
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Figura guardada: {destination.relative_to(ROOT)}")


def run_analysis(source: Path, args: argparse.Namespace) -> None:
    """Ejecuta el análisis completo y escribe resultados reproducibles."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    pd.set_option("display.max_columns", 30)

    training_dir = find_training_directory(source)
    inventory = build_inventory(training_dir)
    counts = inventory.groupby("label").size().rename("images").sort_index()
    summary = pd.Series(
        {
            "training_directory": str(training_dir),
            "total_images": len(inventory),
            "classes": inventory["label"].nunique(),
            "minimum_per_class": counts.min(),
            "maximum_per_class": counts.max(),
            "min_max_ratio": counts.min() / counts.max(),
            "coefficient_of_variation": counts.std() / counts.mean(),
        },
        name="value",
    )
    print("\nRESUMEN DEL CONJUNTO")
    print_table(summary)
    summary.to_csv(TABLE_DIR / "dataset_summary.csv", header=True)
    counts.to_csv(TABLE_DIR / "class_counts.csv", header=True)

    distribution_axis = plot_class_distribution(inventory)
    save_figure(distribution_axis.figure, "01_class_distribution.png")

    header_sample = min(args.header_sample, len(inventory))
    headers = inspect_image_headers(inventory, sample_size=header_sample)
    header_summary = (
        headers.groupby(["format", "width", "height", "mode"], dropna=False)
        .size()
        .rename("files")
        .reset_index()
    )
    print("\nFORMATOS Y RESOLUCIONES")
    print_table(header_summary)
    print("Archivos inválidos en la muestra:", int((~headers["valid"]).sum()))
    header_summary.to_csv(TABLE_DIR / "image_headers.csv", index=False)

    examples_figure = show_examples(
        inventory,
        labels=("A", "B", "M", "U", "V"),
        examples_per_label=3,
    )
    save_figure(examples_figure, "02_examples.png")

    visual_stats = image_statistics(
        inventory,
        per_class=args.visual_sample,
    )
    visual_stats.to_csv(TABLE_DIR / "visual_statistics.csv", index=False)
    visual_grid = plot_visual_statistics(visual_stats)
    save_figure(visual_grid.figure, "03_visual_statistics.png")

    confusion_figure = show_confusion_candidates(inventory)
    save_figure(confusion_figure, "04_similar_letters.png")

    sample = balanced_subsample(
        inventory,
        per_class=args.images_per_class,
    )
    split_data = stratified_split(sample)
    assert_disjoint_splits(split_data)
    manifest_paths = save_manifests(split_data, ROOT / CONFIG.manifest_dir)
    split_summary = split_data.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("\nPARTICIONES POR CLASE")
    print_table(split_summary)
    print("Manifiestos generados:")
    for manifest_path in manifest_paths:
        print(f"  - {manifest_path.relative_to(ROOT)}")

    class_names = sorted(split_data["label"].unique())
    train_manifest = split_data[split_data["split"] == "train"]
    train_dataset = dataset_from_manifest(
        train_manifest,
        class_names,
        training=True,
    )
    preprocessing_check = inspect_preprocessing(train_dataset)
    print("\nCONTROL DE PREPROCESAMIENTO")
    print(preprocessing_check)
    pd.Series(preprocessing_check, name="value").to_csv(
        TABLE_DIR / "preprocessing_check.csv",
        header=True,
    )

    plan = model_plan_table()
    plan.to_csv(TABLE_DIR / "model_plan.csv", index=False)
    print("\nMODELOS SELECCIONADOS")
    print_table(plan)
    print("\nAnálisis finalizado correctamente.")


def main() -> int:
    args = parse_args()
    try:
        if args.download:
            source = download_dataset(ROOT / "data" / "raw")
        else:
            source = find_local_dataset(args.data_dir)
            if source is None:
                print(
                    "No se encontró ASL Alphabet en data/raw.\n"
                    "Ejecute con --download o indique su ubicación con --data-dir."
                )
                return 0
        run_analysis(source, args)
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"No fue posible completar el análisis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
