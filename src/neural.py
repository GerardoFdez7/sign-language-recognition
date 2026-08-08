"""Modelos neuronales y entrenamiento reproducible con PyTorch."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class SafeAugmentation:
    """Variaciones moderadas sin reflejo horizontal."""

    def __init__(self, seed: int = 42) -> None:
        self.generator = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        angle = float(self.generator.uniform(-12.0, 12.0))
        scale = float(self.generator.uniform(0.92, 1.08))
        translate_x = float(self.generator.uniform(-0.06, 0.06) * width)
        translate_y = float(self.generator.uniform(-0.06, 0.06) * height)
        inverse_scale = 1.0 / scale
        center_x, center_y = width / 2.0, height / 2.0
        coefficients = (
            inverse_scale,
            0.0,
            center_x - center_x * inverse_scale - translate_x,
            0.0,
            inverse_scale,
            center_y - center_y * inverse_scale - translate_y,
        )
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        image = ImageEnhance.Contrast(image).enhance(
            float(self.generator.uniform(0.88, 1.12))
        )
        image = ImageEnhance.Brightness(image).enhance(
            float(self.generator.uniform(0.92, 1.08))
        )
        return image


class ASLImageDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        class_names: list[str],
        image_size: int = 64,
        augment: bool = False,
        seed: int = 42,
        cache_path: Path | None = None,
    ) -> None:
        self.paths = manifest["path"].astype(str).tolist()
        label_lookup = {name: index for index, name in enumerate(class_names)}
        self.labels = [label_lookup[label] for label in manifest["label"]]
        self.image_size = image_size
        self.augmentation = SafeAugmentation(seed) if augment else None
        self.cache = np.load(cache_path, mmap_mode="r") if cache_path else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.cache is None:
            with Image.open(self.paths[index]) as source:
                image = source.convert("RGB").resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR,
                )
        else:
            image = Image.fromarray(np.asarray(self.cache[index]))
        if self.augmentation is not None:
            image = self.augmentation(image)
        values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values.transpose(2, 0, 1).copy())
        return tensor, self.labels[index]


def build_image_cache(
    manifest: pd.DataFrame,
    cache_path: Path,
    image_size: int = 64,
) -> Path:
    """Materializa imágenes redimensionadas y valida la correspondencia de rutas."""
    paths = manifest["path"].astype(str).tolist()
    fingerprint = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint and metadata.get("count") == len(paths):
            return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(paths), image_size, image_size, 3),
    )
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )
            cache[index] = np.asarray(image, dtype=np.uint8)
        if (index + 1) % 2_000 == 0 or index + 1 == len(paths):
            print(f"Caché {cache_path.stem}: {index + 1:,}/{len(paths):,}")
    cache.flush()
    del cache
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint, "count": len(paths)}),
        encoding="utf-8",
    )
    return cache_path


class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class CNNRegularized(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.40) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in (12, 24, 48):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int, width: int = 256, dropout: float = 0.40) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def build_model(
    architecture: str,
    num_classes: int,
    dropout: float,
    width: int = 256,
) -> nn.Module:
    if architecture == "cnn_baseline":
        return CNNBaseline(num_classes, dropout)
    if architecture == "cnn_regularized":
        return CNNRegularized(num_classes, dropout)
    if architecture == "mlp":
        return SimpleMLP(num_classes, width, dropout)
    raise ValueError(f"Arquitectura desconocida: {architecture}")


@dataclass(frozen=True)
class NeuralExperiment:
    name: str
    architecture: str
    learning_rate: float
    dropout: float
    width: int = 256
    augment: bool = False


def make_loaders(
    manifests: dict[str, pd.DataFrame],
    class_names: list[str],
    batch_size: int,
    augment: bool,
    seed: int,
    cache_dir: Path = Path("data/processed"),
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split, frame in manifests.items():
        cache_path = build_image_cache(frame, cache_dir / f"{split}_64.npy")
        dataset = ASLImageDataset(
            frame,
            class_names,
            augment=augment and split == "train",
            seed=seed,
            cache_path=cache_path,
        )
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            generator=generator,
        )
    return loaders


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(prediction.cpu().numpy().tolist())
        confidences.extend(confidence.cpu().numpy().tolist())
    return (
        np.asarray(true_labels),
        np.asarray(predicted_labels),
        np.asarray(confidences),
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    true_labels, predicted_labels, _ = predict_loader(model, loader, device)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
    }


def train_experiment(
    experiment: NeuralExperiment,
    manifests: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
    epochs: int = 6,
    patience: int = 2,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[nn.Module, pd.DataFrame, dict[str, float]]:
    seed_everything(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        experiment.architecture,
        len(class_names),
        experiment.dropout,
        experiment.width,
    ).to(device)
    loaders = make_loaders(
        manifests,
        class_names,
        batch_size,
        experiment.augment,
        seed,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=1e-4,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        observation_count = 0
        correct = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size_actual = labels.size(0)
            loss_sum += float(loss.item()) * batch_size_actual
            observation_count += batch_size_actual
            correct += int((logits.argmax(dim=1) == labels).sum().item())

        validation = evaluate_loader(model, loaders["validation"], device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / observation_count,
            "train_accuracy": correct / observation_count,
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_row)
        print(
            f"{experiment.name} epoch={epoch}/{epochs} "
            f"loss={epoch_row['train_loss']:.4f} "
            f"val_f1={epoch_row['val_macro_f1']:.4f}"
        )

        if validation["macro_f1"] > best_f1 + 1e-4:
            best_f1 = validation["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_metrics = evaluate_loader(model, loaders["test"], device)
    result = {
        **test_metrics,
        "validation_macro_f1": best_f1,
        "elapsed_seconds": elapsed,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "experiment": experiment.__dict__,
            "class_names": class_names,
            "metrics": result,
        },
        output_path,
    )
    return model, pd.DataFrame(history), result


def load_neural_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specification = checkpoint["experiment"]
    model = build_model(
        specification["architecture"],
        len(checkpoint["class_names"]),
        specification["dropout"],
        specification.get("width", 256),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
