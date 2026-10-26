"""Check that secondary scores stay internal while labels remain available."""

import sys
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pydantic import ValidationError
from app.schemas.prediction import PredictionDetails, PredictionResponse
from app.services import diagnosis, history_service, predictor_service


def mapped_prediction(show_top2=True):
    branch = {
        "pred_class": "Rust",
        "confidence": 0.8,
        "all_probs": {"Rust": 0.8, "Mealy": 0.2},
    }
    prediction = {
        "pred_class": "Rust",
        "confidence": 0.8,
        "top2_class": "Mealy",
        "top2_confidence": 0.9,
        "show_top2": show_top2,
        "is_uncertain": True,
        "annotated_image": None,
        "detected_boxes": [
            {"xyxy": [1.0, 2.0, 11.0, 12.0], "conf": 0.6, "class_name": "disease"}
        ],
        "big_branch": branch,
        "method": "big_only",
        "selected_branch": "big",
        "elapsed_seconds": 1.0,
    }
    pipeline = SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    with patch.dict(
        sys.modules, {"app.inference": SimpleNamespace(get_pipeline=lambda: pipeline)}
    ):
        return predictor_service._predict_sync(Image.new("RGB", (16, 16)))


class Top2ApiTests(unittest.IsolatedAsyncioTestCase):
    def test_documented_examples_match_public_schema(self):
        root = Path(__file__).resolve().parents[1]
        for path in [root / "API_DOCUMENTATION.md", root / "docs/API_DOCUMENTATION.md"]:
            text = path.read_text(encoding="utf-8")
            start = text.index(
                "```json\n", text.index("#### Phản hồi chẩn đoán thành công")
            ) + len("```json\n")
            end = text.index("\n```", start)
            example = json.loads(text[start:end])
            model = PredictionResponse.model_validate(example)
            self.assertEqual(model.model_dump(), example)

    def test_details_hide_secondary_score_and_preserve_label(self):
        for show in (True, False):
            response = mapped_prediction(show)
            self.assertNotIn("top2_confidence", response["details"])
            self.assertEqual(response["confidence"], 0.8)
            self.assertFalse(
                {
                    "pred_class",
                    "confidence",
                    "top2_class",
                    "is_uncertain",
                    "differential_note",
                    "precision_comparison",
                    "num_boxes",
                }
                & response["details"].keys()
            )
            self.assertNotIn("small_branch", response["details"])
            self.assertNotIn("precision", response["details"]["big_branch"])
            self.assertEqual(
                response["top2_result"]["name"] if show else response["top2_result"],
                "mealy" if show else None,
            )
        self.assertNotIn(
            "top2_confidence", PredictionResponse.model_json_schema()["properties"]
        )

    def test_box_coordinates_have_one_source_and_details_reject_duplicate_fields(self):
        details = mapped_prediction()["details"]
        self.assertEqual(details["detected_boxes"][0]["xyxy"], [1.0, 2.0, 11.0, 12.0])
        PredictionDetails.model_validate(details)
        for key, value in {
            "confidence": 0.8,
            "pred_class": "Rust",
            "num_boxes": 1,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValidationError):
                PredictionDetails.model_validate({**details, key: value})

    async def test_diagnosis_response_hides_score_but_internal_history_keeps_it(self):
        prediction = mapped_prediction()
        image = Image.new("RGB", (16, 16))
        file = SimpleNamespace(close=AsyncMock())
        with (
            patch.object(diagnosis, "read_image", AsyncMock(return_value=image)),
            patch.object(
                diagnosis, "validate_image_quality", return_value={"is_valid": True}
            ),
            patch.object(
                diagnosis, "predict_coffee_leaf", AsyncMock(return_value=prediction)
            ),
            patch.object(diagnosis, "save_image", return_value=Path("test.jpg")),
            patch.object(diagnosis, "add_history", AsyncMock()) as save,
        ):
            response = await diagnosis.diagnose(file, None, None, "default_user")
        self.assertNotIn("top2_confidence", response)
        self.assertNotIn("top2_confidence", response["details"])
        self.assertEqual(response["top2_result"]["name"], "mealy")
        self.assertEqual(response["confidence"], 0.8)
        self.assertEqual(save.await_args.args[0]["top2_confidence"], 0.9)
        self.assertNotIn(
            "top2_confidence", PredictionResponse.model_validate(response).model_dump()
        )

    async def test_legacy_history_scores_hidden_without_modifying_storage(self):
        old = {
            "id": "test",
            "user_id": "default_user",
            "image_url": "test.jpg",
            "result": 2,
            "confidence": 0.8,
            "top2_name": "mealy",
            "top2_confidence": 0.9,
            "created_at": "2026-10-26",
        }
        with patch.object(history_service, "load_histories", return_value=[old]):
            rows = await history_service.get_histories_by_user_id("default_user")
            detail = await history_service.get_history_by_id("test", "default_user")
        self.assertNotIn("top2_confidence", rows[0])
        self.assertNotIn("top2_confidence", detail)
        self.assertEqual(detail["top2_result"]["name"], "mealy")
        self.assertEqual(detail["confidence"], 0.8)
        self.assertEqual(old["top2_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
