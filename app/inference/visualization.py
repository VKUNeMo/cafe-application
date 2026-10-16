"""Prediction overlay rendering."""

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .config import ALWAYS_SHOW_TOP2


def draw_visual_overlay(
    img_pil: Image.Image,
    detected_boxes: List[Dict[str, Any]],
    crop_rect_big: Tuple[int, int, int, int],
    pred_class: str,
    confidence: float,
    top2_class: str = "None",
    top2_confidence: float = 0.0,
    is_uncertain: bool = False,
    show_top2: Optional[bool] = None,
) -> Image.Image:
    """Vẽ chú thích trực quan BBoxes, vùng crop Big và banner kết quả Top-1 / Top-2."""
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img_pil.size

    # 1. Vẽ vùng crop Big (Cyan)
    bx1, by1, bx2, by2 = crop_rect_big
    draw.rectangle([bx1, by1, bx2, by2], outline="#00e5ff", width=max(4, int(w * 0.004)))

    # 2. Vẽ YOLO BBoxes (Đỏ)
    for i, b in enumerate(detected_boxes):
        dx1, dy1, dx2, dy2 = map(int, b["xyxy"])
        draw.rectangle([dx1, dy1, dx2, dy2], outline="#ff1744", width=max(3, int(w * 0.003)))

    # 3. Vẽ Banner kết quả phía trên ảnh
    banner_h = max(55, int(h * 0.08))
    banner_color = (183, 28, 28) if is_uncertain else (27, 67, 50)  # Đỏ trầm nếu phân vân, Xanh đậm nếu chắc chắn

    # Vẽ banner overlay
    overlay = Image.new("RGB", (w, banner_h), banner_color)
    annotated.paste(overlay, (0, 0))

    draw = ImageDraw.Draw(annotated)
    if show_top2 is None:
        show_top2 = ALWAYS_SHOW_TOP2 or is_uncertain

    # Trạng thái tự tin cao (mặc định): Chỉ hiển thị 1 bệnh duy nhất!
    if not show_top2:
        title_text = f"Chan doan: {pred_class} ({confidence * 100:.1f}%) [XAC NHAN]"
    else:
        # Trạng thái phân vân hoặc khi bật hiển thị Top-2
        title_text = f"Top-1: {pred_class} ({confidence * 100:.1f}%)"
        if top2_class != "None" and top2_confidence > 0:
            title_text += f"  |  Phan van Top-2: {top2_class} ({top2_confidence * 100:.1f}%)"
        if is_uncertain:
            title_text += " [CANH BAO PHAN VAN]"

    draw.text((15, int(banner_h * 0.25)), title_text, fill="white")
    return annotated
