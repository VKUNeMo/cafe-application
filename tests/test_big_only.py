"""Big inference, Top-2 boundary, and single-model initialization."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from app.core.config import MODEL_FILENAMES
from app.inference.pipeline import CoffeePipeline


class BigOnlyTests(unittest.TestCase):
    def test_initialization_loads_only_big(self):
        with (
            patch("app.inference.pipeline.YOLO") as yolo,
            patch("app.inference.pipeline.load_classifier", return_value=(Mock(), ["Rust", "Mealy"], 224)) as load,
            patch("app.inference.pipeline.get_classifier_transforms"),
        ):
            CoffeePipeline(yolo_path="yolo.pt", big_path="big.pth", device="cpu")
        load.assert_called_once_with(Path("big.pth"), torch.device("cpu"))
        self.assertEqual(yolo.return_value.overrides.__setitem__.call_args.args, ("device", "cpu"))
        self.assertNotIn("best-small-last.pth", MODEL_FILENAMES)

    def test_top2_boundary_with_or_without_detections(self):
        pipeline = CoffeePipeline.__new__(CoffeePipeline)
        for confidence in (0.5, 0.749999, 0.75, 0.750001, 0.99):
            for boxes in ([], [{"bbox": [0, 0, 8, 8]}]):
                with self.subTest(confidence=confidence, boxes=bool(boxes)):
                    pipeline.detect_boxes = Mock(return_value=boxes)
                    pipeline.predict_big = Mock(return_value={
                        "pred_class": "Rust", "confidence": confidence,
                        "top2_class": "Mealy", "top2_confidence": 0.001,
                        "all_probs": {"Rust": confidence, "Mealy": 0.001},
                        "crop_rect": (0, 0, 16, 16),
                    })
                    with patch("app.inference.pipeline.draw_visual_overlay") as draw:
                        result = pipeline.predict(Image.new("RGB", (16, 16)))
                    self.assertEqual(result["confidence"], confidence)
                    self.assertEqual(result["show_top2"], confidence <= 0.75)
                    self.assertEqual(result["top2_class"], "Mealy")
                    self.assertEqual(draw.call_args.args[-1], confidence <= 0.75)
                    self.assertNotIn("small_branch", result)

    def test_big_ranks_model_probabilities(self):
        pipeline = CoffeePipeline.__new__(CoffeePipeline)
        pipeline.device = torch.device("cpu")
        pipeline.class_names = ["Mealy", "Healthy", "Rust"]
        pipeline.transform = Mock(return_value=torch.zeros(3, 16, 16))
        pipeline.big_model = Mock(return_value=torch.tensor([[0.2, 0.1, 0.7]]).log())
        result = pipeline.predict_big(Image.new("RGB", (16, 16)), [])
        self.assertEqual(result["pred_class"], "Rust")
        self.assertEqual(result["top2_class"], "Mealy")
        self.assertAlmostEqual(result["confidence"], 0.7)
        self.assertAlmostEqual(result["top2_confidence"], 0.2)
