# -*- coding: utf-8 -*-
"""
COFFEE LEAF DISEASE DIAGNOSIS PIPELINE (Ensemble v5)
===================================================
Pipeline đa tầng hiện đại chẩn đoán 8 lớp bệnh lá cà phê:
1. Tầng 1: YOLOv8s Định vị vết bệnh (Square Slicing + NMS)
2. Tầng 2: ResNet Big Classifier (Width-based Square Crop 224x224)
3. Tầng 3: ResNet Small Classifier (BBox Crop 1:1, pad 25%, min 128px)
4. Tầng 4: ENSEMBLE v5 (Disease-Aware & Confidence-Gated Decision Engine)
   - Chẩn đoán phân biệt Top-2 (Top-2 Differential Diagnosis)
   - Cờ phát hiện ca phân vân lâm sàng (Uncertainty Flag & Note)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.inference.config import (
    ALWAYS_SHOW_TOP2,
    BIG_CONFIDENT_THRESHOLD,
    BIG_MARGIN_RATIO,
    BIG_MIN_LEAF_RATIO,
    BIG_MODEL_PATH,
    BIG_RESCUE_MAX_CONF,
    BLENDED_BIG_WEIGHT,
    BLENDED_SMALL_WEIGHT,
    CONF_BIG_MIN_NOBOX,
    CONF_GAP_DOMINANT,
    CONF_SMALL_DOMINANT,
    CONF_SMALL_NEMA_VETO,
    CROSS_CHECK_SMALL_CONF,
    HEALTHY_OVERRIDE_THRESHOLD,
    MACRO_AREA_THRESH,
    MACRO_SIDE_THRESH,
    MEALY_BIG_LOW_CONF,
    MEALY_OVERRIDE_THRESHOLD,
    MICRO_BOX_BIG_CONF,
    MICRO_BOX_GAP,
    MINER_PHOMA_SMALL_CONF,
    NEMA_MULTI_BOX_THRESH,
    NEMATODE_ANTHRACNOSE_BIG_CONF,
    SHOW_TOP2_MIN_CONF,
    SMALL_BATCH_SIZE,
    SMALL_MIN_SIZE,
    SMALL_MODEL_PATH,
    SMALL_PADDING_RATIO,
    SMALL_RESCUE_MIN_CONF,
    SMALL_RUST_OVERRIDE_CONF,
    UNCERTAINTY_CONF_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    YOLO_MODEL_PATH,
    YOLO_OVERLAP_RATIO,
)
from app.inference.preprocessing import (
    crop_single_bbox_square,
    find_best_square_crop_by_width,
    get_center_square_crop,
    get_resnet_transforms,
)

from .classifiers import load_classifier
from .detection import run_fast_square_slicing_yolo
from .visualization import draw_visual_overlay

logger = logging.getLogger(__name__)


class CoffeePipeline:
    """
    Hệ thống chẩn đoán đa tầng bệnh lá cà phê hoàn chỉnh (Pipeline Ensemble v5).
    Tích hợp YOLOv8s + ResNet Big + ResNet Small + Ensemble v5.
    """

    def __init__(
        self,
        config_or_yolo: Union[Any, str, Path] = None,
        big_path: Union[str, Path] = None,
        small_path: Union[str, Path] = None,
        device: Optional[torch.device] = None,
        config: Optional[Any] = None,
        big_resnet_path: Union[str, Path] = None,
        small_resnet_path: Union[str, Path] = None,
        **kwargs,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        cfg_obj = config if config is not None else config_or_yolo

        if hasattr(cfg_obj, "yolo_path"):
            self.yolo_path = Path(cfg_obj.yolo_path)
            self.big_path = Path(
                big_resnet_path or big_path or getattr(cfg_obj, "big_resnet_path", None) or cfg_obj.big_path
            )
            self.small_path = Path(
                small_resnet_path or small_path or getattr(cfg_obj, "small_resnet_path", None) or cfg_obj.small_path
            )
        elif config_or_yolo is not None:
            self.yolo_path = Path(config_or_yolo)
            self.big_path = Path(big_resnet_path or big_path or BIG_MODEL_PATH)
            self.small_path = Path(small_resnet_path or small_path or SMALL_MODEL_PATH)
        else:
            self.yolo_path = Path(YOLO_MODEL_PATH)
            self.big_path = Path(big_resnet_path or big_path or BIG_MODEL_PATH)
            self.small_path = Path(small_resnet_path or small_path or SMALL_MODEL_PATH)

        # 1. Nạp YOLOv8
        if YOLO is None:
            raise ImportError("Vui lòng cài đặt ultralytics để khởi tạo YOLO: pip install ultralytics")
        if not self.yolo_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model YOLO tại: {self.yolo_path}")
        self.yolo_model = YOLO(str(self.yolo_path))
        self.yolo_model.overrides["device"] = str(self.device)

        # 2. Nạp ResNet Models
        self.big_model, self.class_names, self.big_size = load_classifier(self.big_path, self.device)
        self.small_model, self.small_classes, self.small_size = load_classifier(self.small_path, self.device)
        if self.class_names != self.small_classes:
            raise ValueError("Thứ tự lớp của mô hình Big và Small không khớp.")

        # Aliases for compatibility
        self.model_big = self.big_model
        self.model_small = self.small_model
        self.diagnose = self.predict

        self.transform_big = get_resnet_transforms(self.big_size)
        self.transform_small = get_resnet_transforms(self.small_size)
        self.transform = self.transform_big
        self.yolo_conf = getattr(cfg_obj, "yolo_conf", YOLO_CONF)
        self.yolo_iou = getattr(cfg_obj, "yolo_iou", YOLO_IOU)
        self.yolo_imgsz = getattr(cfg_obj, "yolo_imgsz", YOLO_IMGSZ)
        self.yolo_overlap = getattr(cfg_obj, "yolo_overlap", YOLO_OVERLAP_RATIO)
        self.small_padding = getattr(cfg_obj, "small_padding", SMALL_PADDING_RATIO)
        self.small_min_size = getattr(cfg_obj, "small_min_size", SMALL_MIN_SIZE)
        self.big_margin_ratio = getattr(cfg_obj, "big_margin_ratio", BIG_MARGIN_RATIO)
        self.big_min_leaf_ratio = getattr(cfg_obj, "big_min_leaf_ratio", BIG_MIN_LEAF_RATIO)

        # Cấu hình Top-2 Smart Gating & Cờ Phân Vân
        self.always_show_top2 = getattr(cfg_obj, "always_show_top2", ALWAYS_SHOW_TOP2)
        self.show_top2_min_conf = getattr(cfg_obj, "show_top2_min_conf", SHOW_TOP2_MIN_CONF)
        self.uncertainty_conf = getattr(cfg_obj, "uncertainty_conf", UNCERTAINTY_CONF_THRESHOLD)
        self.uncertainty_margin = getattr(cfg_obj, "uncertainty_margin", UNCERTAINTY_MARGIN_THRESHOLD)

        # Ngưỡng điều phối Ensemble v5
        self.blended_big_weight = getattr(cfg_obj, "blended_big_weight", BLENDED_BIG_WEIGHT)
        self.blended_small_weight = getattr(cfg_obj, "blended_small_weight", BLENDED_SMALL_WEIGHT)
        self.miner_phoma_small_conf = getattr(cfg_obj, "miner_phoma_small_conf", MINER_PHOMA_SMALL_CONF)
        self.healthy_override_threshold = getattr(cfg_obj, "healthy_override_threshold", HEALTHY_OVERRIDE_THRESHOLD)
        self.small_rust_override_conf = getattr(cfg_obj, "small_rust_override_conf", SMALL_RUST_OVERRIDE_CONF)
        self.nematode_anthracnose_big_conf = getattr(
            cfg_obj, "nematode_anthracnose_big_conf", NEMATODE_ANTHRACNOSE_BIG_CONF
        )
        self.mealy_big_low_conf = getattr(cfg_obj, "mealy_big_low_conf", MEALY_BIG_LOW_CONF)
        self.mealy_override_threshold = getattr(cfg_obj, "mealy_override_threshold", MEALY_OVERRIDE_THRESHOLD)
        self.small_rescue_min_conf = getattr(cfg_obj, "small_rescue_min_conf", SMALL_RESCUE_MIN_CONF)
        self.big_rescue_max_conf = getattr(cfg_obj, "big_rescue_max_conf", BIG_RESCUE_MAX_CONF)
        self.big_confident_threshold = getattr(cfg_obj, "big_confident_threshold", BIG_CONFIDENT_THRESHOLD)
        self.cross_check_small_conf = getattr(cfg_obj, "cross_check_small_conf", CROSS_CHECK_SMALL_CONF)

        # Siêu tham số Ensemble Scale-Aware v2 / Fallback
        self.conf_big_min_nobox = getattr(cfg_obj, "conf_big_min_nobox", CONF_BIG_MIN_NOBOX)
        self.conf_small_dominant = getattr(cfg_obj, "conf_small_dominant", CONF_SMALL_DOMINANT)
        self.conf_gap_dominant = getattr(cfg_obj, "conf_gap_dominant", CONF_GAP_DOMINANT)
        self.nema_multi_box_thresh = getattr(cfg_obj, "nema_multi_box_thresh", NEMA_MULTI_BOX_THRESH)
        self.conf_small_nema_veto = getattr(cfg_obj, "conf_small_nema_veto", CONF_SMALL_NEMA_VETO)
        self.macro_side_thresh = getattr(cfg_obj, "macro_side_thresh", MACRO_SIDE_THRESH)
        self.macro_area_thresh = getattr(cfg_obj, "macro_area_thresh", MACRO_AREA_THRESH)
        self.micro_box_big_conf = getattr(cfg_obj, "micro_box_big_conf", MICRO_BOX_BIG_CONF)
        self.micro_box_gap = getattr(cfg_obj, "micro_box_gap", MICRO_BOX_GAP)

        logger.info(f"🌿 [CoffeePipeline] Khởi tạo thành công trên thiết bị: {self.device}")
        logger.info(f"   + YOLO:         {self.yolo_path.name}")
        logger.info(f"   + ResNet Big:   {self.big_path.name} (Lớp: {len(self.class_names)})")
        logger.info(f"   + ResNet Small: {self.small_path.name} (Lớp: {len(self.small_classes)})")
        logger.info(
            f"   + Top-2 Policy: {'Always Show' if self.always_show_top2 else 'Smart Gating (Chỉ hiện khi phân vân)'}"
        )

    def detect_boxes(
        self,
        img_pil_or_bgr: Union[Image.Image, np.ndarray],
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        iou: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Phát hiện Bounding Boxes các vết bệnh qua YOLOv8s Slicing."""
        if isinstance(img_pil_or_bgr, Image.Image):
            img_bgr = cv2.cvtColor(np.array(img_pil_or_bgr), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_pil_or_bgr

        return run_fast_square_slicing_yolo(
            model=self.yolo_model,
            img_bgr=img_bgr,
            conf=self.yolo_conf if conf is None else conf,
            imgsz=self.yolo_imgsz if imgsz is None else imgsz,
            iou=self.yolo_iou if iou is None else iou,
            overlap_ratio=self.yolo_overlap if overlap_ratio is None else overlap_ratio,
        )

    def predict_big(self, img_pil: Image.Image, detected_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Thực thi suy luận nhánh ResNet Big (Width-based Square Crop)."""
        w, h = img_pil.size
        raw_bboxes = [b["bbox"] for b in detected_boxes]

        crop_rect = find_best_square_crop_by_width(w, h, raw_bboxes)
        crop_pil = img_pil.crop(crop_rect)

        tensor_big = self.transform_big(crop_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.big_model(tensor_big)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        sorted_idx = np.argsort(probs)[::-1]
        top1_cls = self.class_names[sorted_idx[0]]
        top1_conf = float(probs[sorted_idx[0]])
        top2_cls = self.class_names[sorted_idx[1]]
        top2_conf = float(probs[sorted_idx[1]])

        return {
            "pred_class": top1_cls,
            "pred_conf": top1_conf,
            "pred_prob": top1_conf,
            "top2_class": top2_cls,
            "top2_conf": top2_conf,
            "probs": probs,
            "crop_rect": crop_rect,
            "all_probs": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
        }

    def predict_small(self, img_pil: Image.Image, detected_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Thực thi suy luận nhánh ResNet Small (BBox Crop 1:1, pad 25%, min 128px)."""
        w, h = img_pil.size
        has_detection = len(detected_boxes) > 0

        if has_detection:
            box_crops = []
            crop_boxes = []
            for b in detected_boxes:
                c = crop_single_bbox_square(
                    img_pil, b["bbox"], padding_ratio=self.small_padding, min_size=self.small_min_size
                )
                box_crops.append(self.transform_small(c))
                crop_boxes.append(b["bbox"])

            probability_batches = []
            with torch.no_grad():
                for start in range(0, len(box_crops), SMALL_BATCH_SIZE):
                    batch_small = torch.stack(box_crops[start : start + SMALL_BATCH_SIZE]).to(self.device)
                    logits_small = self.small_model(batch_small)
                    probability_batches.append(torch.softmax(logits_small, dim=1).cpu().numpy())
            probs_all = np.concatenate(probability_batches, axis=0)

            mean_probs = np.mean(probs_all, axis=0)

            # Phân loại cho từng box con
            box_preds = []
            for i, p_box in enumerate(probs_all):
                p_idx = int(np.argmax(p_box))
                box_preds.append(
                    {
                        "id": i + 1,
                        "crop_box": crop_boxes[i],
                        "pred_class": self.small_classes[p_idx],
                        "prob": float(p_box[p_idx]),
                    }
                )
        else:
            center_rect = get_center_square_crop(w, h)
            crop_center = img_pil.crop(center_rect)
            tensor_center = self.transform_small(crop_center).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.small_model(tensor_center)
                mean_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            box_preds = []

        sorted_idx = np.argsort(mean_probs)[::-1]
        top1_cls = self.small_classes[sorted_idx[0]]
        top1_conf = float(mean_probs[sorted_idx[0]])
        top2_cls = self.small_classes[sorted_idx[1]]
        top2_conf = float(mean_probs[sorted_idx[1]])

        return {
            "pred_class": top1_cls,
            "pred_conf": top1_conf,
            "pred_prob": top1_conf,
            "top2_class": top2_cls,
            "top2_conf": top2_conf,
            "probs": mean_probs,
            "box_preds": box_preds,
            "all_probs": {self.small_classes[i]: float(mean_probs[i]) for i in range(len(self.small_classes))},
        }

    def predict_ensemble(
        self,
        img_pil: Optional[Image.Image] = None,
        big_pred: Optional[Dict[str, Any]] = None,
        small_pred: Optional[Dict[str, Any]] = None,
        detected_boxes: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Scale-Aware Ensemble Decision Engine (v2 / v4 Compatible):
        Quy trình 5 bước:
        1. Đồng thuận -> 2. No-box safety -> 3. Nematode Guard (Small Veto) -> 4. Macro Box -> 5. Micro Box
        """
        boxes = detected_boxes if detected_boxes is not None else []
        num_boxes = len(boxes)

        if big_pred is None and img_pil is not None:
            big_pred = self.predict_big(img_pil, detected_boxes=boxes)
        if small_pred is None and img_pil is not None:
            small_pred = self.predict_small(img_pil, detected_boxes=boxes)

        b_class = big_pred.get("pred_class") if big_pred else "Unknown"
        b_conf = float(big_pred.get("pred_prob", big_pred.get("pred_conf", 0.0))) if big_pred else 0.0

        s_class = small_pred.get("pred_class") if small_pred else "Unknown"
        s_conf = float(small_pred.get("pred_prob", small_pred.get("pred_conf", 0.0))) if small_pred else 0.0

        max_side_ratio = 0.0
        max_area_ratio = 0.0

        if img_pil is not None:
            orig_w, orig_h = img_pil.size
        else:
            orig_w, orig_h = 1000, 1000

        min_dim = float(min(orig_w, orig_h))
        img_area = float(orig_w * orig_h)

        for b in boxes:
            bbox = b.get("bbox", None) if isinstance(b, dict) else b
            if bbox is not None and len(bbox) == 4:
                bx, by, bw, bh = bbox
                s_ratio = max(bw, bh) / max(1.0, min_dim)
                a_ratio = (bw * bh) / max(1.0, img_area)
                if s_ratio > max_side_ratio:
                    max_side_ratio = s_ratio
                if a_ratio > max_area_ratio:
                    max_area_ratio = a_ratio

        macro_side_thresh = getattr(self, "macro_side_thresh", MACRO_SIDE_THRESH)
        macro_area_thresh = getattr(self, "macro_area_thresh", MACRO_AREA_THRESH)
        is_macro_box = max_side_ratio >= macro_side_thresh or max_area_ratio >= macro_area_thresh

        conf_big_min_nobox = getattr(self, "conf_big_min_nobox", CONF_BIG_MIN_NOBOX)
        conf_small_dominant = getattr(self, "conf_small_dominant", CONF_SMALL_DOMINANT)
        conf_gap_dominant = getattr(self, "conf_gap_dominant", CONF_GAP_DOMINANT)
        nema_multi_box_thresh = getattr(self, "nema_multi_box_thresh", NEMA_MULTI_BOX_THRESH)
        conf_small_nema_veto = getattr(self, "conf_small_nema_veto", CONF_SMALL_NEMA_VETO)
        micro_box_big_conf = getattr(self, "micro_box_big_conf", MICRO_BOX_BIG_CONF)
        micro_box_gap = getattr(self, "micro_box_gap", MICRO_BOX_GAP)

        # 1. Bước 1: Đồng thuận tuyệt đối
        if b_class == s_class:
            pred_class = b_class
            pred_prob = max(b_conf, s_conf)
            method = "agreement"

        # 2. Bước 2: Không có BBox (No-Box Safety Valve)
        elif num_boxes == 0:
            if b_conf >= conf_big_min_nobox:
                pred_class = b_class
                pred_prob = b_conf
                method = "nobox_big_safe"
            else:
                if s_conf > b_conf:
                    pred_class = s_class
                    pred_prob = s_conf
                    method = "nobox_small_fallback"
                else:
                    pred_class = b_class
                    pred_prob = b_conf
                    method = "nobox_big_lowconf"

        # 3. Bước 3: Nematode Guard & Small Veto Power
        elif b_class == "Nematode" and s_class != "Nematode" and s_conf >= conf_small_nema_veto:
            pred_class = s_class
            pred_prob = s_conf
            method = f"nema_guard_veto_by_small_{s_class.lower()}"

        elif s_class == "Nematode" and b_class != "Nematode" and num_boxes >= nema_multi_box_thresh:
            pred_class = b_class
            pred_prob = b_conf
            method = "nema_guard_multi_box_reject_nema"

        # 4. Bước 4: Tổn thương diện rộng (Macro Box)
        elif is_macro_box:
            if s_conf >= conf_small_dominant and (s_conf - b_conf) >= conf_gap_dominant:
                pred_class = s_class
                pred_prob = s_conf
                method = "macro_box_small_dominant"
            else:
                pred_class = b_class
                pred_prob = b_conf
                method = "macro_box_big_context"

        # 5. Bước 5: Tổn thương vi mô (Micro Box)
        else:
            if b_conf >= micro_box_big_conf and (b_conf - s_conf) >= micro_box_gap:
                pred_class = b_class
                pred_prob = b_conf
                method = "micro_box_big_override"
            else:
                pred_class = s_class
                pred_prob = s_conf
                method = "micro_box_small_specialist"

        return {
            "pred_class": pred_class,
            "confidence": float(pred_prob),
            "pred_prob": float(pred_prob),
            "method": method,
            "is_macro_box": is_macro_box,
            "max_side_ratio": float(max_side_ratio),
            "max_area_ratio": float(max_area_ratio),
            "num_boxes": num_boxes,
            "big_class": b_class,
            "big_conf": b_conf,
            "small_class": s_class,
            "small_conf": s_conf,
        }

    def predict_ensemble_v4(
        self, res_big: Dict[str, Any], res_small: Dict[str, Any], has_detection: bool
    ) -> Dict[str, Any]:
        """
        Decision Engine Ensemble v5:
        Phân quyền theo bệnh + Chặn Mealy giả + Chẩn đoán phân biệt Top-2 & Cờ phân vân.
        """
        top1_b, conf1_b = res_big["pred_class"], res_big["pred_conf"]

        top1_s, conf1_s = res_small["pred_class"], res_small["pred_conf"]

        probs_big = res_big["probs"]
        probs_small = res_small["probs"]

        # Vector kết hợp chuẩn (0.55 Big + 0.45 Small)
        blended_big_w = getattr(self, "blended_big_weight", BLENDED_BIG_WEIGHT)
        blended_small_w = getattr(self, "blended_small_weight", BLENDED_SMALL_WEIGHT)
        probs_blend = blended_big_w * probs_big + blended_small_w * probs_small
        idx_blend_sorted = np.argsort(probs_blend)[::-1]
        top2_blend = self.class_names[idx_blend_sorted[1]]

        # --- 8 QUY TẮC ĐIỀU PHỐI DỰA TRÊN THẾ MẠNH TỪNG BỆNH (ENSEMBLE v5) ---
        # 1. Đồng thuận hoàn toàn
        if top1_b == top1_s:
            pred_label = top1_b
            conf = max(conf1_b, conf1_s)
            second_label = top2_blend if top2_blend != pred_label else self.class_names[idx_blend_sorted[2]]
            second_conf = float(probs_blend[self.class_names.index(second_label)])
            method = "agreement"

        # 2. Phân xử Miner vs Phoma tranh chấp (v5)
        elif top1_b in ["Miner", "Phoma"] and top1_s in ["Miner", "Phoma"]:
            miner_phoma_thresh = getattr(self, "miner_phoma_small_conf", MINER_PHOMA_SMALL_CONF)
            if conf1_s >= miner_phoma_thresh:
                pred_label = top1_s
                conf = conf1_s
                second_label = top1_b
                second_conf = conf1_b
                method = "small_specialist_miner_phoma"
            else:
                pred_label = top1_b
                conf = conf1_b
                second_label = top1_s
                second_conf = conf1_s
                method = "big_confident_miner_phoma"

        # 3. Phân xử Lá khoẻ (Healthy) -> Ưu tiên tuyệt đối Big (100% chính xác)
        elif top1_b == "Healthy":
            healthy_override_thresh = getattr(self, "healthy_override_threshold", HEALTHY_OVERRIDE_THRESHOLD)
            if not has_detection:
                pred_label = "Healthy"
                conf = conf1_b
                second_label = top1_s
                second_conf = conf1_s
                method = "big_healthy_no_box"
            elif top1_s != "Healthy" and conf1_s >= healthy_override_thresh:
                pred_label = top1_s
                conf = conf1_s
                second_label = "Healthy"
                second_conf = conf1_b
                method = "small_high_conf_disease_override"
            else:
                pred_label = "Healthy"
                conf = conf1_b
                second_label = top1_s
                second_conf = conf1_s
                method = "big_healthy_precedence"

        # 4a. NEW v5: Small Rust Override — Big nhận Nematode/Anthracnose nhưng Small nhận Rust
        elif (
            top1_b in ["Nematode", "Anthracnose"]
            and top1_s == "Rust"
            and conf1_s >= getattr(self, "small_rust_override_conf", SMALL_RUST_OVERRIDE_CONF)
        ):
            pred_label = top1_s
            conf = conf1_s
            second_label = top1_b
            second_conf = conf1_b
            method = "small_rust_override"

        # 4b. Phân xử Tuyến trùng (Nematode) và Thán thư (Anthracnose) -> Ưu tiên Big
        elif top1_b in ["Nematode", "Anthracnose"] and conf1_b >= getattr(
            self, "nematode_anthracnose_big_conf", NEMATODE_ANTHRACNOSE_BIG_CONF
        ):
            pred_label = top1_b
            conf = conf1_b
            second_label = top1_s
            second_conf = conf1_s
            method = "big_specialist_nematode_anthracnose"

        # 5. Cơ chế chặn báo động giả Rệp sáp (Mealy Guard)
        elif top1_s == "Mealy" and top1_b != "Mealy":
            mealy_low = getattr(self, "mealy_big_low_conf", MEALY_BIG_LOW_CONF)
            mealy_override = getattr(self, "mealy_override_threshold", MEALY_OVERRIDE_THRESHOLD)
            if conf1_b < mealy_low and conf1_s >= mealy_override and top1_b not in ["Corticium", "Nematode", "Rust"]:
                pred_label = "Mealy"
                conf = conf1_s
                second_label = top1_b
                second_conf = conf1_b
                method = "small_mealy_extreme_conf"
            else:
                pred_label = top1_b
                conf = conf1_b
                second_label = "Mealy"
                second_conf = conf1_s
                method = "reject_small_mealy_false_alarm"

        # 6. Nhánh Small tự tin cao cứu nguy cho Big
        elif conf1_s >= getattr(self, "small_rescue_min_conf", SMALL_RESCUE_MIN_CONF) and conf1_b < getattr(
            self, "big_rescue_max_conf", BIG_RESCUE_MAX_CONF
        ):
            pred_label = top1_s
            conf = conf1_s
            second_label = top1_b
            second_conf = conf1_b
            method = "small_confident_rescue"

        # 7. Nhánh Big tự tin cao (>= 60%) & Cross-check
        elif conf1_b >= getattr(self, "big_confident_threshold", BIG_CONFIDENT_THRESHOLD):
            if conf1_s >= getattr(self, "cross_check_small_conf", CROSS_CHECK_SMALL_CONF) and top1_b != top1_s:
                pred_label = top1_s
                conf = conf1_s
                second_label = top1_b
                second_conf = conf1_b
                method = "small_confident_cross_check"
            else:
                pred_label = top1_b
                conf = conf1_b
                second_label = top1_s
                second_conf = conf1_s
                method = "big_confident"

        # 8. Mặc định (Default Fallback)
        else:
            if conf1_b >= conf1_s:
                pred_label = top1_b
                conf = conf1_b
                second_label = top1_s
                second_conf = conf1_s
                method = "default_big"
            else:
                if top1_s == "Mealy":
                    pred_label = top1_b
                    conf = conf1_b
                    second_label = "Mealy"
                    second_conf = conf1_s
                    method = "default_big_over_small_mealy"
                else:
                    pred_label = top1_s
                    conf = conf1_s
                    second_label = top1_b
                    second_conf = conf1_b
                    method = "default_small"

        # --- CỜ PHÂN VÂN & CHÍNH SÁCH SMART GATING TOP-2 ---
        u_conf_thresh = getattr(self, "uncertainty_conf", UNCERTAINTY_CONF_THRESHOLD)
        u_margin_thresh = getattr(self, "uncertainty_margin", UNCERTAINTY_MARGIN_THRESHOLD)
        always_show = getattr(self, "always_show_top2", ALWAYS_SHOW_TOP2)
        top2_min_conf = getattr(self, "show_top2_min_conf", SHOW_TOP2_MIN_CONF)

        is_uncertain = bool(
            (conf < u_conf_thresh) or (second_label != "None" and abs(conf - second_conf) < u_margin_thresh)
        )

        # Smart Gating: Chỉ hiển thị Top-2 khi có cờ phân vân (hoặc khi bật ALWAYS_SHOW_TOP2)
        # Đồng thời yêu cầu độ tin cậy Top-2 phải đạt ngưỡng tối thiểu
        show_top2 = bool(always_show or (is_uncertain and second_conf >= top2_min_conf))

        diff_note = ""
        if is_uncertain and second_label != "None":
            diff_note = (
                f"Hệ thống nghiêng về {pred_label} ({conf * 100:.1f}%), nhưng nhận thấy đặc điểm "
                f"rất gần với {second_label} ({second_conf * 100:.1f}%). "
                f"Khuyến nghị kiểm tra thêm mặt dưới lá hoặc biểu hiện cành trước khi dùng thuốc."
            )

        status_str = "uncertain" if is_uncertain else "confident"
        display_diag = (
            f"{pred_label} ({conf * 100:.1f}%)"
            if not show_top2
            else f"{pred_label} ({conf * 100:.1f}%) [Phân vân: {second_label} ({second_conf * 100:.1f}%)]"
        )

        return {
            "pred_class": pred_label,
            "confidence": float(conf),
            "show_top2": show_top2,
            "status": status_str,
            "display_diagnosis": display_diag,
            "top2_class": second_label,
            "top2_confidence": float(second_conf),
            "top2_display_class": second_label if show_top2 else None,
            "top2_display_confidence": float(second_conf) if show_top2 else None,
            "is_uncertain": is_uncertain,
            "differential_note": diff_note,
            "method": method,
        }

    predict_ensemble_v5 = predict_ensemble_v4

    def draw_visual_overlay(
        self,
        img_pil,
        detected_boxes,
        crop_rect_big,
        pred_class,
        confidence,
        top2_class="None",
        top2_confidence=0.0,
        is_uncertain=False,
        show_top2=None,
    ):
        if show_top2 is None:
            show_top2 = self.always_show_top2 or is_uncertain
        return draw_visual_overlay(
            img_pil,
            detected_boxes,
            crop_rect_big,
            pred_class,
            confidence,
            top2_class,
            top2_confidence,
            is_uncertain,
            show_top2,
        )

    def predict(
        self, image_input: Union[str, Path, Image.Image, np.ndarray], return_visual: bool = True
    ) -> Dict[str, Any]:
        """
        Quy trình chẩn đoán toàn diện End-to-End:
        1. Tiền xử lý ảnh
        2. YOLOv8s Slicing
        3. ResNet Big (Width Crop)
        4. ResNet Small (BBox Crop)
        5. Ensemble v5 (Top-1 + Top-2 Smart Gating + Cờ phân vân)
        6. Vẽ overlay kết quả
        """
        start_time = time.time()

        # Đọc ảnh
        if isinstance(image_input, (str, Path)):
            img_pil = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
        else:
            raise ValueError("Định dạng ảnh đầu vào không hợp lệ.")

        # 1. YOLO Detection
        detected_boxes = self.detect_boxes(img_pil)
        has_detection = len(detected_boxes) > 0

        # 2. ResNet Big
        res_big = self.predict_big(img_pil, detected_boxes)

        # 3. ResNet Small
        res_small = self.predict_small(img_pil, detected_boxes)

        # 4. Ensemble v5 Decision Engine
        ens_res = self.predict_ensemble_v4(res_big, res_small, has_detection)

        elapsed = time.time() - start_time

        # 5. Vẽ ảnh chú thích
        annotated_pil = None
        if return_visual:
            annotated_pil = self.draw_visual_overlay(
                img_pil=img_pil,
                detected_boxes=detected_boxes,
                crop_rect_big=res_big["crop_rect"],
                pred_class=ens_res["pred_class"],
                confidence=ens_res["confidence"],
                top2_class=ens_res["top2_class"],
                top2_confidence=ens_res["top2_confidence"],
                is_uncertain=ens_res["is_uncertain"],
                show_top2=ens_res.get("show_top2", False),
            )

        return {
            "pred_class": ens_res["pred_class"],
            "confidence": ens_res["confidence"],
            "show_top2": ens_res.get("show_top2", False),
            "status": ens_res.get("status", "confident"),
            "display_diagnosis": ens_res.get(
                "display_diagnosis", f"{ens_res['pred_class']} ({ens_res['confidence'] * 100:.1f}%)"
            ),
            "top2_class": ens_res["top2_class"],
            "top2_confidence": ens_res["top2_confidence"],
            "top2_display_class": ens_res.get("top2_display_class"),
            "top2_display_confidence": ens_res.get("top2_display_confidence"),
            "is_uncertain": ens_res["is_uncertain"],
            "differential_note": ens_res["differential_note"],
            "method": ens_res["method"],
            "detected_boxes": detected_boxes,
            "has_disease_detected": has_detection,
            "elapsed_seconds": round(elapsed, 3),
            "annotated_image": annotated_pil,
            "original_image": img_pil,
            "big_branch": {
                "pred_class": res_big["pred_class"],
                "confidence": res_big["pred_conf"],
                "all_probs": res_big["all_probs"],
            },
            "small_branch": {
                "pred_class": res_small["pred_class"],
                "confidence": res_small["pred_conf"],
                "all_probs": res_small["all_probs"],
                "box_preds": res_small["box_preds"],
            },
        }
