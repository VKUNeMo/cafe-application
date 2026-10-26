"""Settings used by YOLO and the Big classifier."""

from app.core.config import get_settings

DEFAULT_MODEL_DIR = get_settings().model_dir
YOLO_MODEL_PATH = DEFAULT_MODEL_DIR / "best-yolov26.pt"
BIG_MODEL_PATH = DEFAULT_MODEL_DIR / "best-big-last.pth"
DEFAULT_CLASS_NAMES = ["Anthracnose", "Corticium", "Mealy", "Miner", "Nematode", "Phoma", "Rust", "Healthy"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSIFIER_IMG_SIZE = 224
UNSHARP_RADIUS = 1.5
UNSHARP_PERCENT = 130
UNSHARP_THRESHOLD = 3

YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
YOLO_OVERLAP_RATIO = 0.20
TOP2_CONF_THRESHOLD = 0.75
