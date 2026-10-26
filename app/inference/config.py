# -*- coding: utf-8 -*-
"""
PIPELINE CONFIGURATION
======================
Cấu hình tập trung chuẩn hóa cho toàn bộ hệ thống chẩn đoán đa tầng bệnh lá cà phê:
- YOLOv8s: Định vị vết bệnh (Square Slicing 640x640)
- ResNet Big: Width-based Square Crop 224x224
- ResNet Small: BBox Crop 1:1 (min_size=128, padding=25%)
- Ensemble Precision: đồng thuận hoặc chọn nhánh theo precision từng lớp
"""
from app.core.config import get_settings

PROJECT_ROOT = get_settings().project_dir
DEFAULT_MODEL_DIR = get_settings().model_dir

YOLO_MODEL_PATH = DEFAULT_MODEL_DIR / "best-yolov26.pt"
BIG_MODEL_PATH = DEFAULT_MODEL_DIR / "best-big-last.pth"
SMALL_MODEL_PATH = DEFAULT_MODEL_DIR / "best-small-last.pth"

DEFAULT_CLASS_NAMES = ["Anthracnose", "Corticium", "Mealy", "Miner", "Nematode", "Phoma", "Rust", "Healthy"]

# ==============================================================================
# 1. TIỀN XỬ LÝ & CHUẨN HÓA HÌNH ẢNH
# ==============================================================================
# ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
UNCERTAINTY_CONF_THRESHOLD: float = 0.65  # Nếu Conf Top-1 < 65% -> Kích hoạt phân vân
UNCERTAINTY_MARGIN_THRESHOLD: float = 0.20  # Nếu (Conf Top-1 - Conf Top-2) < 20% -> Kích hoạt phân vân

# Ngưỡng tin cậy tối thiểu để hiển thị Top-2 (tránh hiển thị các nhãn rác có conf quá bé)
SHOW_TOP2_MIN_CONF: float = 0.10

# ==============================================================================
# 5. BẢNG PRECISION THEO LỚP TỪ BỘ ĐÁNH GIÁ
# ==============================================================================
PRECISION_TABLE_PATH = get_settings().precision_table_path
