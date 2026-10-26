# -*- coding: utf-8 -*-
"""
PIPELINE PREPROCESSING & CROPPING
=================================
Các thuật toán tiền xử lý hình ảnh và cắt trích xuất vùng đặc trưng:
- ApplyUnsharpMask: Tăng độ nét vi mô và giữ rõ đường biên đốm bệnh
- find_best_square_crop_by_width: Cắt vùng vuông kích thước theo cạnh ngắn bao bọc cụm vết bệnh cho nhánh ResNet Big
- crop_single_bbox_square: Cắt vùng vuông quanh từng BBox cho nhánh ResNet Small (kèm padding 25% và đệm viền đen)
- get_center_square_crop: Cắt vùng vuông trung tâm khi không có BBox
- get_resnet_transforms: Chuỗi biến đổi ảnh ResNet (Resize Bilinear, Unsharp Mask, Normalize)
"""

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from app.inference.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESNET_IMG_SIZE,
    SMALL_MIN_SIZE,
    SMALL_PADDING_RATIO,
    UNSHARP_PERCENT,
    UNSHARP_RADIUS,
    UNSHARP_THRESHOLD,
)


class ApplyUnsharpMask:
    """Làm nét giữ rõ biên của đốm bệnh và chi tiết vi mô qua Unsharp Mask."""

    def __init__(
        self, radius: float = UNSHARP_RADIUS, percent: int = UNSHARP_PERCENT, threshold: int = UNSHARP_THRESHOLD
    ):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        return img.filter(ImageFilter.UnsharpMask(radius=self.radius, percent=self.percent, threshold=self.threshold))


def get_resnet_transforms(img_size: int = RESNET_IMG_SIZE) -> transforms.Compose:
    """Transform cho ResNet: Resize Bilinear -> UnsharpMask -> ToTensor -> Normalize."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            ApplyUnsharpMask(radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=UNSHARP_THRESHOLD),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_center_square_crop(img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Cắt vùng vuông chính giữa ảnh khi YOLO không tìm thấy vết bệnh."""
    side = min(img_w, img_h)
    x1 = (img_w - side) // 2
    y1 = (img_h - side) // 2
    return (x1, y1, x1 + side, y1 + side)


def find_best_square_crop_by_width(
    img_w: int, img_h: int, bboxes: List[Tuple[float, float, float, float]]
) -> Tuple[int, int, int, int]:
    """
    Thuật toán cắt hình vuông bao bọc tối đa cụm vết bệnh cho nhánh ResNet Big.
    - Cạnh hình vuông = min(img_w, img_h)
    - Trượt dọc theo chiều dài hơn để tối đa hóa số lượng vết bệnh nằm trong vùng cắt
    - Ưu tiên vùng có tâm gần tâm cụm vết bệnh nhất nếu số lượng vết bệnh bằng nhau
    """
    side = min(img_w, img_h)

    if not bboxes:
        return get_center_square_crop(img_w, img_h)

    # Tính tâm cụm vết bệnh (Centroid)
    cluster_cx = np.mean([b[0] + b[2] / 2.0 for b in bboxes])
    cluster_cy = np.mean([b[1] + b[3] / 2.0 for b in bboxes])

    if img_h > img_w:
        max_shift = img_h - side
        if max_shift <= 0:
            return (0, 0, side, side)

        candidates = {0, max_shift}
        for bx, by, bw, bh in bboxes:
            candidates.add(int(round(by)))
            candidates.add(int(round(by + bh - side)))
            candidates.add(int(round(by + bh / 2.0 - side / 2.0)))
        candidates.add(int(round(cluster_cy - side / 2.0)))

        step = max(10, max_shift // 100)
        for s in range(0, max_shift + 1, step):
            candidates.add(s)

        valid_shifts = [max(0, min(s, max_shift)) for s in candidates]

        best_y1 = None
        best_count = -1
        min_dist = float("inf")

        for y1_cand in valid_shifts:
            y2_cand = y1_cand + side
            cnt = sum(1 for (bx, by, bw, bh) in bboxes if y1_cand <= (by + bh / 2.0) <= y2_cand)
            center_dist = abs((y1_cand + side / 2.0) - cluster_cy)

            if (cnt > best_count) or (cnt == best_count and center_dist < min_dist):
                best_count = cnt
                min_dist = center_dist
                best_y1 = y1_cand

        return (0, best_y1, side, best_y1 + side)
    else:
        max_shift = img_w - side
        if max_shift <= 0:
            return (0, 0, side, side)

        candidates = {0, max_shift}
        for bx, by, bw, bh in bboxes:
            candidates.add(int(round(bx)))
            candidates.add(int(round(bx + bw - side)))
            candidates.add(int(round(bx + bw / 2.0 - side / 2.0)))
        candidates.add(int(round(cluster_cx - side / 2.0)))

        step = max(10, max_shift // 100)
        for s in range(0, max_shift + 1, step):
            candidates.add(s)

        valid_shifts = [max(0, min(s, max_shift)) for s in candidates]

        best_x1 = None
        best_count = -1
        min_dist = float("inf")

        for x1_cand in valid_shifts:
            x2_cand = x1_cand + side
            cnt = sum(1 for (bx, by, bw, bh) in bboxes if x1_cand <= (bx + bw / 2.0) <= x2_cand)
            center_dist = abs((x1_cand + side / 2.0) - cluster_cx)

            if (cnt > best_count) or (cnt == best_count and center_dist < min_dist):
                best_count = cnt
                min_dist = center_dist
                best_x1 = x1_cand

        return (best_x1, 0, best_x1 + side, side)


def crop_single_bbox_square(
    img: Image.Image, bbox: List[float], padding_ratio: float = SMALL_PADDING_RATIO, min_size: int = SMALL_MIN_SIZE
) -> Image.Image:
    """
    Cắt vùng vuông 1:1 bao quanh một BBox cụ thể cho nhánh ResNet Small.
    - Kích thước = max(bw, bh) * (1 + padding_ratio)
    - Đảm bảo >= min_size (128px)
    - Nếu tràn viền ảnh, đệm đen (zero-padding canvas) để giữ trọn vẹn tỷ lệ vết bệnh
    """
    img_w, img_h = img.size
    bx, by, bw, bh = bbox
    cx = bx + bw / 2.0
    cy = by + bh / 2.0

    side = max(bw, bh) * (1.0 + padding_ratio)
    side = max(side, min_size)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(x1 + side))
    y2 = int(round(y1 + side))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    rx1 = max(0, x1)
    ry1 = max(0, y1)
    rx2 = min(img_w, x2)
    ry2 = min(img_h, y2)

    crop = img.crop((rx1, ry1, rx2, ry2))

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        new_w = crop.width + pad_left + pad_right
        new_h = crop.height + pad_top + pad_bottom
        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        canvas.paste(crop, (pad_left, pad_top))
        crop = canvas
    return crop


# Quality Control Configuration & Functions (Chuẩn hóa cho app.py & services)
CONFIG = {"min_width": 200, "min_height": 200, "brightness_min": 25.0, "brightness_max": 235.0}


def is_valid_image(img_pil: Image.Image) -> Tuple[bool, str]:
    """Kiểm tra ảnh hợp lệ về kích thước tối thiểu."""
    w, h = img_pil.size
    if w < CONFIG["min_width"] or h < CONFIG["min_height"]:
        return (
            False,
            f"Kích thước ảnh quá nhỏ ({w}x{h}). Yêu cầu tối thiểu {CONFIG['min_width']}x{CONFIG['min_height']}.",
        )
    return True, "Hợp lệ"


def run_quality_control(img_pil: Image.Image) -> dict:
    """Kiểm tra kích thước và ghi nhận độ sáng; không kiểm tra độ mờ."""
    valid, msg = is_valid_image(img_pil)
    img_gray = np.array(img_pil.convert("L"))
    brightness = float(np.mean(img_gray))

    is_extreme_lighting = brightness < CONFIG["brightness_min"] or brightness > CONFIG["brightness_max"]

    return {
        "is_valid": valid,
        "message": msg if not valid else "Đạt tiêu chuẩn",
        "resolution": f"{img_pil.width}x{img_pil.height}",
        "brightness": round(brightness, 2),
        "is_extreme_lighting": is_extreme_lighting,
    }
