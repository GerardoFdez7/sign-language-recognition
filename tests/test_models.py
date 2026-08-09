import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluar_fotos import label_from_filename
from src.classical import hog_descriptor
from src.neural import build_model


class ModelTest(unittest.TestCase):
    def test_all_neural_models_produce_29_logits(self) -> None:
        values = torch.rand(2, 3, 64, 64)
        for architecture in ("cnn_baseline", "cnn_regularized", "mlp"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture, num_classes=29, dropout=0.3)
                self.assertEqual(tuple(model(values).shape), (2, 29))

    def test_hog_descriptor_is_finite_and_stable_size(self) -> None:
        image = Image.new("RGB", (200, 200), (20, 100, 180))
        descriptor = hog_descriptor(image)
        self.assertEqual(descriptor.shape, (1764,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_external_labels_are_read_from_filenames(self) -> None:
        self.assertEqual(label_from_filename(Path("A_01.jpg")), "A")
        self.assertEqual(label_from_filename(Path("space-demo.png")), "space")
        self.assertIsNone(label_from_filename(Path("unknown.jpg")))


if __name__ == "__main__":
    unittest.main()
