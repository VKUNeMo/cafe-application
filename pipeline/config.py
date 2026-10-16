# -*- coding: utf-8 -*-
"""
PIPELINE CONFIGURATION
======================
Cấu hình tập trung chuẩn hóa cho toàn bộ hệ thống chẩn đoán đa tầng bệnh lá cà phê:
- YOLOv8s: Định vị vết bệnh (Square Slicing 640x640)
- ResNet Big: Width-based Square Crop 224x224
- ResNet Small: BBox Crop 1:1 (min_size=128, padding=25%)
- Ensemble v5: Disease-Aware & Confidence-Gated với Top-2 Smart Gating
"""

from pathlib import Path
from typing import Dict, Any, List
import torch

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

YOLO_MODEL_PATH = DEFAULT_MODEL_DIR / "best-yolov26.pt"
BIG_MODEL_PATH = DEFAULT_MODEL_DIR / "best-big-last.pth"
SMALL_MODEL_PATH = DEFAULT_MODEL_DIR / "best-small-last.pth"

DEFAULT_CLASS_NAMES = [
    'Anthracnose', 'Corticium', 'Mealy', 'Miner', 'Nematode', 'Phoma', 'Rust', 'Healthy'
]

# ==============================================================================
# 1. TIỀN XỬ LÝ & CHUẨN HÓA HÌNH ẢNH
# ==============================================================================
# ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Unsharp Mask Enhancer
UNSHARP_RADIUS = 1.5
UNSHARP_PERCENT = 130
UNSHARP_THRESHOLD = 3

# ==============================================================================
# 2. CẤU HÌNH YOLOv8s DETECTION & SQUARE SLICING
# ==============================================================================
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
YOLO_OVERLAP_RATIO = 0.20

# ==============================================================================
# 3. CẤU HÌNH CẮT ẢNH CHO 2 NHÁNH CLASSIFIER
# ==============================================================================
RESNET_IMG_SIZE = 224

# Nhánh Small (BBox Crop)
SMALL_PADDING_RATIO = 0.25
SMALL_MIN_SIZE = 128
SMALL_BATCH_SIZE = 16

# Nhánh Big (Width Crop)
BIG_MARGIN_RATIO = 0.35
BIG_MIN_LEAF_RATIO = 0.45

# ==============================================================================
# 4. CHÍNH SÁCH HIỂN THỊ TOP-2 & CỜ PHÂN VÂN (SMART GATING)
# ==============================================================================
# True: Luôn hiển thị Top-2 ở tất cả ảnh
# False: CHỈ hiển thị Top-2 khi mô hình rơi vào trạng thái phân vân (khuyến nghị cho thực tế)
ALWAYS_SHOW_TOP2: bool = False

# Ngưỡng kích hoạt cờ phân vân (Uncertainty Flag)
UNCERTAINTY_CONF_THRESHOLD: float = 0.65       # Nếu Conf Top-1 < 65% -> Kích hoạt phân vân
UNCERTAINTY_MARGIN_THRESHOLD: float = 0.20     # Nếu (Conf Top-1 - Conf Top-2) < 20% -> Kích hoạt phân vân

# Ngưỡng tin cậy tối thiểu để hiển thị Top-2 (tránh hiển thị các nhãn rác có conf quá bé)
SHOW_TOP2_MIN_CONF: float = 0.10

# ==============================================================================
# 5. CÁC NGƯỠNG ĐIỀU PHỐI ENSEMBLE v5 (DISEASE-AWARE & GATED)
# ==============================================================================
BLENDED_BIG_WEIGHT: float = 0.55               # Trọng số Big trong vector kết hợp
BLENDED_SMALL_WEIGHT: float = 0.45             # Trọng số Small trong vector kết hợp

# Quy tắc 2: Sâu vẽ bùa vs Đốm lá Phoma (Miner <-> Phoma)
MINER_PHOMA_SMALL_CONF: float = 0.70           # Small tự tin >= 70% thì ưu tiên Small, ngược lại Big

# Quy tắc 3: Phân xử Lá khỏe (Healthy)
HEALTHY_OVERRIDE_THRESHOLD: float = 0.90       # Small phải đạt >= 90% mới ghi đè Healthy của Big

# Quy tắc 4a: Phân xử Rỉ sắt (Rust Override)
SMALL_RUST_OVERRIDE_CONF: float = 0.55         # Small nhận Rust >= 55% thì override Big

# Quy tắc 4b: Tuyến trùng & Thán thư (Nematode & Anthracnose)
NEMATODE_ANTHRACNOSE_BIG_CONF: float = 0.50    # Big tự tin >= 50% ở 2 bệnh này thì ưu tiên Big

# Quy tắc 5: Chặn báo động giả Rệp sáp (Mealy Guard)
MEALY_BIG_LOW_CONF: float = 0.40              # Ngưỡng Big chưa chắc chắn (< 40%)
MEALY_OVERRIDE_THRESHOLD: float = 0.95         # Small phải cực kỳ tự tin (>= 95%) mới được đoán Mealy

# Quy tắc 6: Small tự tin cao cứu nguy cho Big
SMALL_RESCUE_MIN_CONF: float = 0.70            # Small tự tin cứu nguy khi >= 70%
BIG_RESCUE_MAX_CONF: float = 0.60              # Big do dự khi < 60%

# Quy tắc 7: Big tự tin cao & Cross-check
BIG_CONFIDENT_THRESHOLD: float = 0.60          # Big tự tin khi >= 60%
CROSS_CHECK_SMALL_CONF: float = 0.65           # Small tự tin >= 65% khi Big tự tin nhưng nhận khác bệnh

# ==============================================================================
# 6. CÁC NGƯỠNG CHO ENSEMBLE SCALE-AWARE (LEGACY / FALLBACK)
# ==============================================================================
CONF_BIG_MIN_NOBOX: float = 0.50
CONF_SMALL_DOMINANT: float = 0.85
CONF_GAP_DOMINANT: float = 0.15
NEMA_MULTI_BOX_THRESH: int = 5
CONF_SMALL_NEMA_VETO: float = 0.45
MACRO_SIDE_THRESH: float = 0.45
MACRO_AREA_THRESH: float = 0.15
MICRO_BOX_BIG_CONF: float = 0.90
MICRO_BOX_GAP: float = 0.30

# ==============================================================================
# 7. CẤU HÌNH CÁC PROFILE MÔ HÌNH (SETTINGS)
# ==============================================================================
AVAILABLE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "thucte-ptn": {
        "name": "Thực Tế + Phòng Thí Nghiệm (8 Lớp)",
        "yolo_path": YOLO_MODEL_PATH,
        "big_path": BIG_MODEL_PATH,
        "small_path": SMALL_MODEL_PATH,
        "classes": DEFAULT_CLASS_NAMES
    },
    "setting_rust_only_real": {
        "name": "Thực Tế + Phòng Thí Nghiệm (8 Lớp - big_best_model-rust-only-real.pth)",
        "yolo_path": YOLO_MODEL_PATH,
        "big_path": BIG_MODEL_PATH,
        "small_path": SMALL_MODEL_PATH,
        "classes": DEFAULT_CLASS_NAMES
    },
    "setting_healthy": {
        "name": "Thực Tế + Phòng Thí Nghiệm (8 Lớp)",
        "yolo_path": YOLO_MODEL_PATH,
        "big_path": BIG_MODEL_PATH,
        "small_path": SMALL_MODEL_PATH,
        "classes": DEFAULT_CLASS_NAMES
    },
    "thucte": {
        "name": "Thực Tế (8 Lớp)",
        "yolo_path": YOLO_MODEL_PATH,
        "big_path": BIG_MODEL_PATH,
        "small_path": SMALL_MODEL_PATH,
        "classes": DEFAULT_CLASS_NAMES
    }
}

BASE_DIR = PROJECT_ROOT


class PipelineConfig:
    """Đối tượng cấu hình tổng hợp cho toàn bộ Pipeline."""
    def __init__(self, setting_name: str = "thucte-ptn"):
        self.setting = setting_name
        setting = AVAILABLE_SETTINGS.get(setting_name, AVAILABLE_SETTINGS["thucte-ptn"])
        self.name = setting["name"]
        self.yolo_path = setting["yolo_path"]
        self.big_path = setting["big_path"]
        self.small_path = setting["small_path"]
        self.big_resnet_path = self.big_path
        self.small_resnet_path = self.small_path
        self.classes = setting["classes"]
        self.class_names = self.classes
        self.num_classes = len(self.classes)

        # YOLO params
        self.yolo_conf = YOLO_CONF
        self.yolo_iou = YOLO_IOU
        self.yolo_imgsz = YOLO_IMGSZ
        self.yolo_overlap = YOLO_OVERLAP_RATIO
        self.yolo_overlap_ratio = YOLO_OVERLAP_RATIO

        # Crop params
        self.resnet_img_size = RESNET_IMG_SIZE
        self.small_padding = SMALL_PADDING_RATIO
        self.small_padding_ratio = SMALL_PADDING_RATIO
        self.small_min_size = SMALL_MIN_SIZE
        self.big_margin_ratio = BIG_MARGIN_RATIO
        self.big_min_leaf_ratio = BIG_MIN_LEAF_RATIO

        # Smart Gating Top-2 & Uncertainty params
        self.always_show_top2 = ALWAYS_SHOW_TOP2
        self.show_top2_min_conf = SHOW_TOP2_MIN_CONF
        self.uncertainty_conf = UNCERTAINTY_CONF_THRESHOLD
        self.uncertainty_margin = UNCERTAINTY_MARGIN_THRESHOLD

        # Ensemble v5 params
        self.blended_big_weight = BLENDED_BIG_WEIGHT
        self.blended_small_weight = BLENDED_SMALL_WEIGHT
        self.miner_phoma_small_conf = MINER_PHOMA_SMALL_CONF
        self.healthy_override_threshold = HEALTHY_OVERRIDE_THRESHOLD
        self.small_rust_override_conf = SMALL_RUST_OVERRIDE_CONF
        self.nematode_anthracnose_big_conf = NEMATODE_ANTHRACNOSE_BIG_CONF
        self.mealy_big_low_conf = MEALY_BIG_LOW_CONF
        self.mealy_override_threshold = MEALY_OVERRIDE_THRESHOLD
        self.small_rescue_min_conf = SMALL_RESCUE_MIN_CONF
        self.big_rescue_max_conf = BIG_RESCUE_MAX_CONF
        self.big_confident_threshold = BIG_CONFIDENT_THRESHOLD
        self.cross_check_small_conf = CROSS_CHECK_SMALL_CONF

        # Compatibility aliases for app.py
        self.ens_big_confident = BIG_CONFIDENT_THRESHOLD
        self.ens_small_dominant = CONF_SMALL_DOMINANT

        # Test dataset config
        self.test_dir = PROJECT_ROOT / "Data" / "anh-train-thuc-te-va-thi-nghiem" / "test"
        self.coco_json_path = self.test_dir / "_annotations.coco.json"
        self.include_healthy_test = True
        self.use_color_norm = False
        self.color_norm_method = "reinhard"


def get_config(setting_name: str = "thucte-ptn") -> PipelineConfig:
    """Hàm lấy cấu hình pipeline theo tên setting."""
    return PipelineConfig(setting_name)
