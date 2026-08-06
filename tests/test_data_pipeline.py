import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.config import ProjectConfig
from src.data import (
    assert_disjoint_splits,
    balanced_subsample,
    build_inventory,
    find_training_directory,
    inspect_image_headers,
    stratified_split,
)


def _make_tiny_dataset(root: Path, files_per_class: int = 10) -> Path:
    training = root / "asl_alphabet_train"
    labels = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "del", "nothing", "space"]
    for class_index, label in enumerate(labels):
        class_dir = training / label
        class_dir.mkdir(parents=True)
        for image_index in range(files_per_class):
            color = (class_index * 7 % 255, image_index * 20 % 255, 120)
            Image.new("RGB", (200, 200), color).save(
                class_dir / f"{label}_{image_index:03d}.jpg"
            )
    return training


class DataPipelineTest(unittest.TestCase):
    def test_inventory_headers_and_balanced_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training = _make_tiny_dataset(root)
            detected = find_training_directory(root)
            self.assertEqual(detected, training.resolve())

            inventory = build_inventory(detected)
            self.assertEqual(len(inventory), 290)
            self.assertEqual(inventory["label"].nunique(), 29)

            headers = inspect_image_headers(inventory, sample_size=None)
            self.assertTrue(headers["valid"].all())
            self.assertEqual(set(headers["width"]), {200})
            self.assertEqual(set(headers["height"]), {200})
            self.assertEqual(set(headers["mode"]), {"RGB"})

            sample = balanced_subsample(inventory, per_class=10, seed=42)
            config = ProjectConfig(images_per_class=10)
            split_data = stratified_split(sample, config=config)
            assert_disjoint_splits(split_data)

            self.assertEqual(len(split_data), len(sample))
            self.assertEqual(
                set(split_data["split"]),
                {"train", "validation", "test"},
            )
            self.assertTrue(
                split_data.groupby("split")["label"].nunique().eq(29).all()
            )


if __name__ == "__main__":
    unittest.main()
