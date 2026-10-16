"""Square slicing detection and non-maximum suppression."""

from typing import Any, Dict, List

import numpy as np

from .config import YOLO_CONF, YOLO_IMGSZ, YOLO_IOU, YOLO_OVERLAP_RATIO


def nms_bboxes(bboxes: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Non-Maximum Suppression (NMS) gộp các Bounding Box chồng lấn."""
    if not bboxes:
        return []

    boxes = np.array([b["xyxy"] for b in bboxes], dtype=float)
    scores = np.array([b["conf"] for b in bboxes], dtype=float)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return [bboxes[k] for k in keep]


def run_fast_square_slicing_yolo(
    model: Any,
    img_bgr: np.ndarray,
    conf: float = YOLO_CONF,
    imgsz: int = YOLO_IMGSZ,
    iou: float = YOLO_IOU,
    overlap_ratio: float = YOLO_OVERLAP_RATIO,
) -> List[Dict[str, Any]]:
    """Phát hiện vết bệnh với YOLOv8s kết hợp kỹ thuật cắt lát vuông thích ứng (Square Slicing)."""
    h_orig, w_orig = img_bgr.shape[:2]
    side = min(w_orig, h_orig)

    # 1. Dự đoán toàn cảnh (Full Image)
    all_boxes = []
    full_res = model.predict(source=img_bgr, conf=conf, imgsz=imgsz, iou=iou, verbose=False)[0]
    for box in full_res.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        c = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = full_res.names.get(cls_id, str(cls_id))
        all_boxes.append(
            {
                "xyxy": xyxy,
                "bbox": (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
                "conf": c,
                "class_id": cls_id,
                "class_name": cls_name,
            }
        )

    # 2. Cắt lát vuông (Square Slices) theo chiều dài hơn nếu cần
    if max(w_orig, h_orig) > 1.35 * side:
        step = int(side * (1.0 - overlap_ratio))
        slices = []
        if w_orig > h_orig:
            x = 0
            while x < w_orig:
                x_end = min(x + side, w_orig)
                x_start = max(0, x_end - side)
                slices.append((x_start, 0, x_end, side))
                if x_end >= w_orig:
                    break
                x += step
        else:
            y = 0
            while y < h_orig:
                y_end = min(y + side, h_orig)
                y_start = max(0, y_end - side)
                slices.append((0, y_start, side, y_end))
                if y_end >= h_orig:
                    break
                y += step

        for sx1, sy1, sx2, sy2 in slices:
            slice_bgr = img_bgr[sy1:sy2, sx1:sx2]
            res_slice = model.predict(source=slice_bgr, conf=conf, imgsz=imgsz, iou=iou, verbose=False)[0]
            for box in res_slice.boxes:
                b_xyxy = box.xyxy[0].cpu().numpy().tolist()
                c = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = res_slice.names.get(cls_id, str(cls_id))
                gx1 = b_xyxy[0] + sx1
                gy1 = b_xyxy[1] + sy1
                gx2 = b_xyxy[2] + sx1
                gy2 = b_xyxy[3] + sy1
                all_boxes.append(
                    {
                        "xyxy": [gx1, gy1, gx2, gy2],
                        "bbox": (gx1, gy1, gx2 - gx1, gy2 - gy1),
                        "conf": c,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

    return nms_bboxes(all_boxes, iou_thresh=iou)
