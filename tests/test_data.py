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
from src.preprocessing import dataset_from_manifest, inspect_preprocessing


def make_dataset(root: Path, files_per_class: int = 10) -> Path:
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
    def test_inventory_split_and_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training = make_dataset(root)
            self.assertEqual(find_training_directory(root), training.resolve())
            inventory = build_inventory(training)
            self.assertEqual(len(inventory), 290)
            self.assertEqual(inventory["label"].nunique(), 29)

            headers = inspect_image_headers(inventory, sample_size=None)
            self.assertTrue(headers["valid"].all())
            self.assertEqual(set(headers["width"]), {200})
            self.assertEqual(set(headers["height"]), {200})

            sample = balanced_subsample(inventory, per_class=10)
            split_data = stratified_split(
                sample,
                config=ProjectConfig(images_per_class=10),
            )
            assert_disjoint_splits(split_data)
            self.assertEqual(len(split_data), 290)
            self.assertEqual(set(split_data["split"]), {"train", "validation", "test"})

            class_names = sorted(split_data["label"].unique())
            dataset = dataset_from_manifest(
                split_data[split_data["split"] == "train"],
                class_names,
                training=True,
                config=ProjectConfig(images_per_class=10),
            )
            check = inspect_preprocessing(dataset)
            self.assertEqual(check["image_shape"], (32, 64, 64, 3))
            self.assertEqual(check["dtype"], "float32")
            self.assertGreaterEqual(check["minimum"], 0.0)
            self.assertLessEqual(check["maximum"], 1.0)


if __name__ == "__main__":
    unittest.main()
