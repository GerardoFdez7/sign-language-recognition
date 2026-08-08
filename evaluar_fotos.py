"""Evalúa el mejor modelo en imágenes oficiales y fotografías del equipo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.classical import hog_descriptor
from src.neural import load_neural_checkpoint


ROOT = Path(__file__).resolve().parent
METRIC_DIR = ROOT / "artifacts" / "metrics"
FIGURE_DIR = ROOT / "artifacts" / "figures"
OFFICIAL_DIR = ROOT / "data" / "raw" / "asl_alphabet_test" / "asl_alphabet_test"
EXTERNAL_DIR = ROOT / "data" / "external"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def label_from_filename(path: Path) -> str | None:
    token = re.split(r"[_\-\s]", path.stem, maxsplit=1)[0]
    if len(token) == 1 and token.isalpha():
        return token.upper()
    token = token.lower()
    if token in {"del", "nothing", "space"}:
        return token
    return None


def discover_images() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if OFFICIAL_DIR.exists():
        for path in sorted(OFFICIAL_DIR.iterdir()):
            label = label_from_filename(path)
            if path.suffix.lower() in VALID_EXTENSIONS and label:
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "true_label": label,
                        "source": "official_demo",
                        "participant": "Kaggle",
                    }
                )
    if EXTERNAL_DIR.exists():
        for participant_dir in sorted(path for path in EXTERNAL_DIR.iterdir() if path.is_dir()):
            for path in sorted(participant_dir.rglob("*")):
                label = label_from_filename(path)
                if path.suffix.lower() in VALID_EXTENSIONS and label:
                    rows.append(
                        {
                            "path": str(path.resolve()),
                            "true_label": label,
                            "source": "team_photos",
                            "participant": participant_dir.name,
                        }
                    )
    return pd.DataFrame(rows)


def predict_neural(paths: list[str], checkpoint_path: Path) -> tuple[list[str], list[float]]:
    model, checkpoint = load_neural_checkpoint(checkpoint_path)
    class_names = checkpoint["class_names"]
    predictions: list[str] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            logits = model(torch.from_numpy(values).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(probabilities.argmax().item())
            predictions.append(class_names[index])
            confidences.append(float(probabilities[index].item()))
    return predictions, confidences


def predict_svm(paths: list[str], model_path: Path) -> tuple[list[str], list[float]]:
    bundle = joblib.load(model_path)
    features = []
    for path in paths:
        with Image.open(path) as image:
            features.append(hog_descriptor(image))
    matrix = np.stack(features)
    model = bundle["model"]
    predictions = model.predict(matrix).tolist()
    confidence = model.decision_function(matrix).max(axis=1).astype(float).tolist()
    return predictions, confidence


def main() -> int:
    best_path = METRIC_DIR / "best_model.json"
    if not best_path.exists():
        print("Primero ejecute entrenar_modelos.py.")
        return 0
    images = discover_images()
    if images.empty:
        print("No se encontraron imágenes externas para evaluar.")
        return 0
    best = json.loads(best_path.read_text(encoding="utf-8"))
    model_path = ROOT / best["model_path"]
    if best["family"] == "svm_hog":
        predictions, confidence = predict_svm(images["path"].tolist(), model_path)
    else:
        predictions, confidence = predict_neural(images["path"].tolist(), model_path)
    images["predicted_label"] = predictions
    images["confidence"] = confidence
    images["correct"] = images["true_label"] == images["predicted_label"]
    images.to_csv(METRIC_DIR / "external_predictions.csv", index=False)

    summary_rows = []
    for (source, participant), frame in images.groupby(["source", "participant"]):
        summary_rows.append(
            {
                "source": source,
                "participant": participant,
                "images": len(frame),
                "distinct_letters": frame["true_label"].nunique(),
                "accuracy": accuracy_score(frame["true_label"], frame["predicted_label"]),
                "macro_f1": f1_score(
                    frame["true_label"],
                    frame["predicted_label"],
                    average="macro",
                    zero_division=0,
                ),
                "meets_minimum": source != "team_photos" or frame["true_label"].nunique() >= 5,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(METRIC_DIR / "external_summary.csv", index=False)
    print(summary.to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary, x="participant", y="accuracy", hue="source", ax=axis)
    axis.set(title="Desempeño en imágenes externas", ylim=(0, 1), ylabel="Exactitud")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "07_external_evaluation.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

