"""YOLO detection followed by Big classification."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .classifiers import load_classifier
from .config import (
    BIG_MODEL_PATH, TOP2_CONF_THRESHOLD, YOLO_CONF, YOLO_IMGSZ,
    YOLO_IOU, YOLO_MODEL_PATH, YOLO_OVERLAP_RATIO,
)
from .detection import run_fast_square_slicing_yolo
from .preprocessing import find_best_square_crop_by_width, get_classifier_transforms
from .visualization import draw_visual_overlay


class CoffeePipeline:
    def __init__(self, yolo_path=None, big_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.yolo_model = YOLO(str(yolo_path or YOLO_MODEL_PATH))
        self.yolo_model.overrides["device"] = str(self.device)
        self.big_model, self.class_names, image_size = load_classifier(
            Path(big_path or BIG_MODEL_PATH), self.device
        )
        self.transform = get_classifier_transforms(image_size)

    def detect_boxes(self, image: Image.Image) -> list[dict]:
        return run_fast_square_slicing_yolo(
            model=self.yolo_model,
            img_bgr=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            iou=YOLO_IOU,
            overlap_ratio=YOLO_OVERLAP_RATIO,
        )

    def predict_big(self, image: Image.Image, boxes: list[dict]) -> dict:
        crop_rect = find_best_square_crop_by_width(*image.size, [box["bbox"] for box in boxes])
        tensor = self.transform(image.crop(crop_rect)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self.big_model(tensor).softmax(dim=1).cpu().numpy()[0]
        first, second = np.argsort(probs)[-2:][::-1]
        return {
            "pred_class": self.class_names[first],
            "confidence": float(probs[first]),
            "top2_class": self.class_names[second],
            "top2_confidence": float(probs[second]),
            "all_probs": dict(zip(self.class_names, map(float, probs))),
            "crop_rect": crop_rect,
        }

    def predict(self, image_input, return_visual: bool = True) -> dict:
        started = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as source:
                image = source.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert("RGB")

        boxes = self.detect_boxes(image)
        big = self.predict_big(image, boxes)
        show_top2 = big["confidence"] <= TOP2_CONF_THRESHOLD
        annotated = None
        if return_visual:
            annotated = draw_visual_overlay(
                image, boxes, big["crop_rect"], big["pred_class"], big["confidence"],
                big["top2_class"], show_top2,
            )
        return {
            "pred_class": big["pred_class"],
            "confidence": big["confidence"],
            "top2_class": big["top2_class"],
            "top2_confidence": big["top2_confidence"],
            "show_top2": show_top2,
            "detected_boxes": boxes,
            "big_branch": {
                "pred_class": big["pred_class"],
                "confidence": big["confidence"],
                "all_probs": big["all_probs"],
            },
            "annotated_image": annotated,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
