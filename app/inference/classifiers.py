"""Classifier architectures and checkpoint loading."""

from pathlib import Path
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
from torchvision import models

from .config import DEFAULT_CLASS_NAMES, RESNET_IMG_SIZE


class ConvNeXtCoffeeClassifier(nn.Module):
    """Mô hình ConvNeXt V2-T (convnextv2_tiny) Classifier cho chẩn đoán bệnh lá cà phê (8 lớp)."""

    def __init__(self, num_classes: int = 8, dropout: float = 0.5):
        super().__init__()
        if timm is None:
            raise ImportError("Thư viện `timm` chưa được cài đặt. Vui lòng cài `pip install timm` để chạy ConvNeXt V2.")
        self.backbone = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=False, num_classes=0)
        in_features = self.backbone.num_features  # 768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out


class CoffeeClassifier(nn.Module):
    """Mô hình ResNet50 Classifier cho chẩn đoán bệnh lá cà phê (8 lớp)."""

    def __init__(self, num_classes: int = 8, dropout: float = 0.5):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_classifier(model_path: Path, device: torch.device) -> Tuple[nn.Module, List[str], int]:
    """Nạp trọng số checkpoint của ResNet hoặc ConvNeXt V2 model (tự động nhận diện kiến trúc)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model checkpoint tại: {model_path}")

    ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    num_classes = 8
    if "classifier.6.weight" in state_dict:
        num_classes = state_dict["classifier.6.weight"].shape[0]
    elif "classifier.weight" in state_dict:
        num_classes = state_dict["classifier.weight"].shape[0]

    if isinstance(ckpt, dict) and "class_names" in ckpt and len(ckpt["class_names"]) == num_classes:
        class_names = ckpt["class_names"]
    else:
        class_names = DEFAULT_CLASS_NAMES if num_classes == 8 else [f"class_{i}" for i in range(num_classes)]

    is_convnext = (
        (isinstance(ckpt, dict) and ckpt.get("model_name") == "convnextv2_tiny")
        or any(k.startswith("backbone.") for k in state_dict.keys())
        or ("convnext" in str(model_path).lower())
    )

    if is_convnext:
        model = ConvNeXtCoffeeClassifier(num_classes=num_classes).to(device)
    else:
        model = CoffeeClassifier(num_classes=num_classes).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    is_448 = "448" in str(model_path.name)
    img_size = 448 if is_448 else RESNET_IMG_SIZE

    return model, class_names, img_size
