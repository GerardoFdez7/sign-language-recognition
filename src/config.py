"""Configuración central del análisis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    images_per_class: int = 600
    image_height: int = 64
    image_width: int = 64
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    manifest_dir: Path = Path("data/manifests")

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not abs(total - 1.0) < 1e-9:
            raise ValueError("Las proporciones deben sumar 1.0")
        if self.images_per_class <= 0:
            raise ValueError("images_per_class debe ser positivo")


CONFIG = ProjectConfig()

