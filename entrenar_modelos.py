"""Entrena, ajusta y compara CNN, MLP y SVM para ASL Alphabet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.classical import extract_hog_features, train_linear_svm
from src.neural import (
    ASLImageDataset,
    NeuralExperiment,
    load_neural_checkpoint,
    predict_loader,
    train_experiment,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
METRIC_DIR = ARTIFACTS / "metrics"
FIGURE_DIR = ARTIFACTS / "figures"
CACHE_DIR = ROOT / "data" / "processed"


NEURAL_EXPERIMENTS = (
    NeuralExperiment("cnn_base_lr001", "cnn_baseline", 1e-3, 0.30),
    NeuralExperiment("cnn_base_lr0003", "cnn_baseline", 3e-4, 0.40),
    NeuralExperiment("cnn_reg_drop035", "cnn_regularized", 1e-3, 0.35),
    NeuralExperiment("cnn_reg_drop050", "cnn_regularized", 5e-4, 0.50),
    NeuralExperiment("mlp_256", "mlp", 1e-3, 0.35, width=256),
    NeuralExperiment("mlp_512", "mlp", 3e-4, 0.50, width=512),
    NeuralExperiment("cnn_base_aug", "cnn_baseline", 1e-3, 0.30, augment=True),
    NeuralExperiment("cnn_reg_aug", "cnn_regularized", 1e-3, 0.35, augment=True),
    NeuralExperiment("mlp_aug", "mlp", 1e-3, 0.35, width=256, augment=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "all.csv",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--only",
        choices=("all", "neural", "classical"),
        default="all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repite experimentos aunque ya existan resultados",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Nombres concretos que se ejecutarán; por defecto se ejecutan todos",
    )
    return parser.parse_args()


def prepare_directories() -> None:
    for directory in (MODEL_DIR, METRIC_DIR, FIGURE_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_manifests(path: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    data = pd.read_csv(path)
    required = {"path", "label", "split"}
    if not required.issubset(data.columns):
        raise ValueError(f"El manifiesto debe contener {sorted(required)}")
    missing = [path for path in data["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"No existe la primera imagen requerida: {missing[0]}")
    manifests = {
        split: frame.reset_index(drop=True)
        for split, frame in data.groupby("split")
    }
    if set(manifests) != {"train", "validation", "test"}:
        raise ValueError("Se requieren train, validation y test")
    return manifests, sorted(data["label"].unique())


def save_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def neural_predictions(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    class_names: list[str],
    batch_size: int,
) -> pd.DataFrame:
    dataset = ASLImageDataset(manifest, class_names, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    true_indices, predicted_indices, confidence = predict_loader(
        model,
        loader,
        torch.device("cpu"),
    )
    return pd.DataFrame(
        {
            "path": manifest["path"].to_numpy(),
            "true_label": [class_names[index] for index in true_indices],
            "predicted_label": [class_names[index] for index in predicted_indices],
            "confidence": confidence,
        }
    )


def run_neural_experiments(
    manifests: dict[str, pd.DataFrame],
    class_names: list[str],
    args: argparse.Namespace,
) -> list[dict]:
    results: list[dict] = []
    for experiment in NEURAL_EXPERIMENTS:
        if args.experiments and experiment.name not in args.experiments:
            continue
        checkpoint_path = MODEL_DIR / f"{experiment.name}.pt"
        metrics_path = METRIC_DIR / f"{experiment.name}.json"
        history_path = METRIC_DIR / f"{experiment.name}_history.csv"
        predictions_path = METRIC_DIR / f"{experiment.name}_predictions.csv"
        if checkpoint_path.exists() and metrics_path.exists() and not args.force:
            print(f"Omitiendo {experiment.name}: resultado existente")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append(metrics)
            continue

        print(f"\n=== {experiment.name} ===")
        model, history, raw_metrics = train_experiment(
            experiment,
            manifests,
            class_names,
            checkpoint_path,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        history.to_csv(history_path, index=False)
        predictions = neural_predictions(
            model,
            manifests["test"],
            class_names,
            args.batch_size,
        )
        predictions.to_csv(predictions_path, index=False)
        metrics = {
            "name": experiment.name,
            "family": experiment.architecture,
            "augmented": experiment.augment,
            "learning_rate": experiment.learning_rate,
            "dropout": experiment.dropout,
            "width": experiment.width,
            "model_path": str(checkpoint_path.relative_to(ROOT)),
            "predictions_path": str(predictions_path.relative_to(ROOT)),
            **raw_metrics,
        }
        save_json(metrics_path, metrics)
        results.append(metrics)
    return results


def feature_cache(
    split: str,
    manifest: pd.DataFrame,
    augment: bool = False,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = "_aug" if augment else ""
    path = CACHE_DIR / f"hog_{split}{suffix}.npz"
    if path.exists() and not force:
        cached = np.load(path)
        return cached["features"], cached["labels"]
    features, labels = extract_hog_features(manifest, augment=augment)
    np.savez_compressed(path, features=features, labels=labels)
    return features, labels


def run_classical_experiments(
    manifests: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> list[dict]:
    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        features[split], labels[split] = feature_cache(
            split,
            manifests[split],
            force=args.force,
        )

    results: list[dict] = []
    specifications = (("svm_c01", 0.1, False), ("svm_c1", 1.0, False), ("svm_c01_aug", 0.1, True))
    augmented_features = augmented_labels = None
    for name, c_value, augmented in specifications:
        if args.experiments and name not in args.experiments:
            continue
        model_path = MODEL_DIR / f"{name}.joblib"
        metrics_path = METRIC_DIR / f"{name}.json"
        predictions_path = METRIC_DIR / f"{name}_predictions.csv"
        if model_path.exists() and metrics_path.exists() and not args.force:
            print(f"Omitiendo {name}: resultado existente")
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        if augmented and augmented_features is None:
            augmented_features, augmented_labels = feature_cache(
                "train",
                manifests["train"],
                augment=True,
                force=args.force,
            )
        print(f"\n=== {name} ===")
        model, raw_metrics = train_linear_svm(
            features["train"],
            labels["train"],
            features["validation"],
            labels["validation"],
            features["test"],
            labels["test"],
            c_value,
            model_path,
            augmented_features if augmented else None,
            augmented_labels if augmented else None,
        )
        predicted = model.predict(features["test"])
        decision = model.decision_function(features["test"])
        confidence = decision.max(axis=1)
        pd.DataFrame(
            {
                "path": manifests["test"]["path"],
                "true_label": labels["test"],
                "predicted_label": predicted,
                "confidence": confidence,
            }
        ).to_csv(predictions_path, index=False)
        metrics = {
            "name": name,
            "family": "svm_hog",
            "augmented": augmented,
            "c_value": c_value,
            "model_path": str(model_path.relative_to(ROOT)),
            "predictions_path": str(predictions_path.relative_to(ROOT)),
            **raw_metrics,
        }
        save_json(metrics_path, metrics)
        results.append(metrics)
    return results


def create_result_artifacts(results: list[dict], class_names: list[str]) -> None:
    comparison = pd.DataFrame(results).sort_values(
        "validation_macro_f1",
        ascending=False,
    )
    comparison.to_csv(METRIC_DIR / "model_comparison.csv", index=False)
    best = comparison.iloc[0].to_dict()
    save_json(METRIC_DIR / "best_model.json", best)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 6))
    plot_data = comparison.sort_values("macro_f1")
    colors = ["#0F766E" if value else "#2563EB" for value in plot_data["augmented"]]
    axis.barh(plot_data["name"], plot_data["macro_f1"], color=colors)
    axis.set(xlabel="Macro F1 en prueba", ylabel="Modelo", xlim=(0, 1))
    axis.set_title("Comparación de modelos")
    for index, value in enumerate(plot_data["macro_f1"]):
        axis.text(value + 0.005, index, f"{value:.3f}", va="center")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "05_model_comparison.png", dpi=180)
    plt.close(figure)

    predictions = pd.read_csv(ROOT / best["predictions_path"])
    matrix = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=class_names,
    )
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="Blues", ax=axis, xticklabels=class_names, yticklabels=class_names)
    axis.set(title=f"Matriz de confusión: {best['name']}", xlabel="Predicción", ylabel="Real")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "06_best_confusion_matrix.png", dpi=180)
    plt.close(figure)

    report = classification_report(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(METRIC_DIR / "best_classification_report.csv")

    history_candidates = comparison[
        (~comparison["augmented"])
        & comparison["family"].isin(["cnn_baseline", "cnn_regularized", "mlp"])
    ]
    champions = (
        history_candidates.sort_values("validation_macro_f1", ascending=False)
        .groupby("family", as_index=False)
        .first()
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for champion in champions.itertuples(index=False):
        history_path = METRIC_DIR / f"{champion.name}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            axis.plot(
                history["epoch"],
                history["val_macro_f1"],
                marker="o",
                label=champion.name,
            )
    axis.set(
        title="Evolución de macro F1 en validación",
        xlabel="Época",
        ylabel="Macro F1",
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "08_training_curves.png", dpi=180)
    plt.close(figure)

    augmentation_rows = []
    pairs = {
        "CNN base": ("cnn_base_lr001", "cnn_base_aug"),
        "CNN regularizada": ("cnn_reg_drop035", "cnn_reg_aug"),
        "MLP": ("mlp_256", "mlp_aug"),
        "SVM-HOG": ("svm_c01", "svm_c01_aug"),
    }
    by_name = comparison.set_index("name")
    for family, (plain_name, augmented_name) in pairs.items():
        if plain_name in by_name.index and augmented_name in by_name.index:
            augmentation_rows.extend(
                [
                    {
                        "family": family,
                        "training": "Sin transformaciones",
                        "macro_f1": by_name.loc[plain_name, "macro_f1"],
                    },
                    {
                        "family": family,
                        "training": "Con transformaciones",
                        "macro_f1": by_name.loc[augmented_name, "macro_f1"],
                    },
                ]
            )
    augmentation_data = pd.DataFrame(augmentation_rows)
    augmentation_data.to_csv(METRIC_DIR / "augmentation_comparison.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=augmentation_data,
        x="family",
        y="macro_f1",
        hue="training",
        ax=axis,
    )
    axis.set(
        title="Efecto de las transformaciones en prueba",
        xlabel="Familia",
        ylabel="Macro F1",
        ylim=(0, 1),
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "09_augmentation_comparison.png", dpi=180)
    plt.close(figure)
    print("\nCOMPARACIÓN FINAL")
    print(comparison[["name", "family", "augmented", "accuracy", "macro_f1", "validation_macro_f1"]].to_string(index=False))
    print(f"\nModelo seleccionado por validación: {best['name']}")


def main() -> int:
    args = parse_args()
    prepare_directories()
    manifests, class_names = load_manifests(args.manifest)
    results: list[dict] = []
    if args.only in {"all", "neural"}:
        results.extend(run_neural_experiments(manifests, class_names, args))
    if args.only in {"all", "classical"}:
        results.extend(run_classical_experiments(manifests, args))

    existing = []
    for path in METRIC_DIR.glob("*.json"):
        if path.name == "best_model.json":
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        if "macro_f1" in content and "name" in content:
            existing.append(content)
    if results or existing:
        unique = {row["name"]: row for row in [*existing, *results]}
        create_result_artifacts(list(unique.values()), class_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
