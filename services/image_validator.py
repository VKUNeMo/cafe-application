# -*- coding: utf-8 -*-
"""
IMAGE QUALITY VALIDATOR SERVICE
===============================
Module tiền xử lý kiểm định chất lượng ảnh đầu vào:
- Độ phân giải tối thiểu (Resolution Check)
- Độ sắc nét / Độ mờ (Blur Detection qua Laplacian Variance)
- Độ sáng tối / Cháy sáng (Underexposure & Overexposure Check)
- Nhận diện vùng mô lá cà phê (Foliage / Plant Coverage Ratio)
Tự động từ chối ảnh không đạt chuẩn và trả về lý do cụ thể.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image


def calculate_blur_score(img_bgr: np.ndarray) -> float:
    """Tính toán độ sắc nét bằng phương sai Laplacian (Laplacian Variance)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness_metrics(img_bgr: np.ndarray) -> Tuple[float, float]:
    """
    Tính độ sáng trung bình và tỷ lệ điểm cháy sáng (Overexposed pixels).
    - Độ sáng thang [0, 255]
    - Tỷ lệ điểm cháy sáng: V > 240 và S < 25 (màu trắng gắt lóa)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    mean_v = float(np.mean(v_channel))

    # Điểm trắng lóa cháy sáng
    overexposed_mask = (v_channel > 240) & (s_channel < 25)
    overexposed_ratio = float(np.count_nonzero(overexposed_mask) / (img_bgr.shape[0] * img_bgr.shape[1]))

    return mean_v, overexposed_ratio


def calculate_plant_leaf_ratio(img_bgr: np.ndarray) -> float:
    """
    Tính tỷ lệ diện tích mang màu sắc của mô lá cà phê trên ảnh:
    - Màu xanh lục tự nhiên của phiến lá khỏe: H in [25, 90], S > 20, V > 20
    - Màu vàng, cam của bệnh tuyến trùng hoặc bào tử gỉ sắt: H in [10, 25], S > 30, V > 30
    - Màu nâu hoại tử của vết cháy nấm Phoma / sâu đục lá: H in [5, 20], S > 35, V in [20, 160]
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # 1. Lá xanh
    mask_green = cv2.inRange(hsv, (25, 20, 20), (90, 255, 255))

    # 2. Vàng/cam bệnh lý (Nematode / Rust)
    mask_yellow_orange = cv2.inRange(hsv, (10, 30, 30), (25, 255, 255))

    # 3. Nâu hoại tử (Phoma / Miner necrotic)
    mask_brown = cv2.inRange(hsv, (5, 35, 20), (20, 255, 160))

    plant_mask = mask_green | mask_yellow_orange | mask_brown
    plant_pixels = np.count_nonzero(plant_mask)

    return float(plant_pixels / total_pixels)


def validate_image_quality(
    img_pil: Image.Image,
    min_resolution: int = 180,
    min_total_pixels: int = 40000,
    min_laplacian_var: float = 15.0,
    min_brightness: float = 35.0,
    max_brightness: float = 230.0,
    max_overexposed_ratio: float = 0.40,
    min_leaf_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Kiểm định toàn diện chất lượng ảnh đầu vào:
    1. Kiểm tra kích thước và độ phân giải
    2. Kiểm tra độ mờ / mất nét
    3. Kiểm tra ảnh quá tối hoặc cháy sáng
    4. Kiểm tra sự hiện diện của phiến lá thực vật
    """
    w, h = img_pil.size
    total_pixels = w * h

    # Chuyển đổi sang BGR cho OpenCV xử lý
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 1. Kiểm tra độ phân giải tối thiểu
    if w < min_resolution or h < min_resolution or total_pixels < min_total_pixels:
        return {
            "is_valid": False,
            "error_code": "RESOLUTION_TOO_LOW",
            "message": f"Độ phân giải ảnh quá thấp ({w}x{h} px). Vui lòng chụp hoặc tải ảnh có kích thước tối thiểu 200x200 px để nhìn rõ đốm bệnh.",
            "details": {
                "width": w,
                "height": h,
                "total_pixels": total_pixels
            }
        }

    # 2. Kiểm tra độ sắc nét / độ mờ
    lap_var = calculate_blur_score(img_bgr)
    if lap_var < min_laplacian_var:
        return {
            "is_valid": False,
            "error_code": "IMAGE_BLURRY",
            "message": f"Ảnh bị nhòe/mờ nghiêm trọng (độ nét {lap_var:.1f} < {min_laplacian_var}). Vui lòng giữ chắc tay máy và lấy nét lại vào bề mặt lá cà phê.",
            "details": {
                "width": w,
                "height": h,
                "laplacian_var": round(lap_var, 2),
                "threshold": min_laplacian_var
            }
        }

    # 3. Kiểm tra độ sáng (Quá tối hoặc Quá chói sáng)
    mean_v, overexposed_ratio = calculate_brightness_metrics(img_bgr)
    if mean_v < min_brightness:
        return {
            "is_valid": False,
            "error_code": "IMAGE_TOO_DARK",
            "message": f"Ảnh quá tối (độ sáng {mean_v:.1f}/255 < {min_brightness}). Vui lòng bật đèn flash hoặc chụp ở nơi có đủ ánh sáng ban ngày.",
            "details": {
                "mean_brightness": round(mean_v, 2),
                "threshold": min_brightness
            }
        }

    if mean_v > max_brightness or overexposed_ratio > max_overexposed_ratio:
        return {
            "is_valid": False,
            "error_code": "IMAGE_OVEREXPOSED",
            "message": f"Ảnh bị chói sáng/cháy sáng nghiêm trọng (độ sáng {mean_v:.1f}/255, diện tích cháy lóa {overexposed_ratio*100:.1f}%). Vui lòng tránh chụp ngược sáng hoặc ánh nắng gắt chói trực diện.",
            "details": {
                "mean_brightness": round(mean_v, 2),
                "overexposed_ratio": round(overexposed_ratio, 3),
                "threshold": max_brightness
            }
        }

    # 4. Kiểm tra sự hiện diện của phiến lá cà phê (màu sắc thực vật)
    leaf_ratio = calculate_plant_leaf_ratio(img_bgr)
    if leaf_ratio < min_leaf_ratio:
        return {
            "is_valid": False,
            "error_code": "NOT_A_COFFEE_LEAF",
            "message": f"Không nhận diện được phiến lá cà phê trong ảnh (tỷ lệ bề mặt lá chỉ đạt {leaf_ratio*100:.1f}% < {min_leaf_ratio*100:.0f}%). Vui lòng chụp rõ bề mặt lá cà phê thay vì nền đất, sàn nhà hoặc vật thể khác.",
            "details": {
                "leaf_ratio": round(leaf_ratio, 3),
                "threshold": min_leaf_ratio
            }
        }

    # Ảnh đạt chuẩn chất lượng
    return {
        "is_valid": True,
        "error_code": None,
        "message": "Ảnh đạt chuẩn chất lượng chẩn đoán.",
        "details": {
            "width": w,
            "height": h,
            "laplacian_var": round(lap_var, 2),
            "mean_brightness": round(mean_v, 2),
            "leaf_ratio": round(leaf_ratio, 3)
        }
    }
