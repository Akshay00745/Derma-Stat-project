"""
EfficientNetB0 Transfer Learning Architecture for 7-Class Skin Lesion Classification.

Uses torchvision's pretrained EfficientNetB0 backbone:
- Pretrained on ImageNet (1000 classes)
- Feature extractor: Compound scaled convolutional blocks (output: 1280 channels)
- Classifier replacement: Dropout (0.3) + Linear(1280 -> 7 classes)
- Supports feature extraction and fine-tuning
- Exposes final conv stage for Grad-CAM explainability
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional

try:
    from . import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class SkinLesionEfficientNet(nn.Module):
    """
    7-Class Skin Lesion Classifier based on EfficientNetB0.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # Optional backbone freezing for feature extraction
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # In EfficientNetB0, classifier is Sequential(Dropout, Linear(1280, 1000))
        in_features = self.model.classifier[1].in_features  # 1280
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the final convolutional block of EfficientNetB0
        for Grad-CAM feature map and gradient extraction.
        In EfficientNetB0: self.model.features[-1] (Conv2dNormActivation, 1280 channels).
        """
        return self.model.features[-1]


def build_model(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> SkinLesionEfficientNet:
    """
    Factory function to construct the model and optionally load saved weights.
    """
    model = SkinLesionEfficientNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
        # Handle cases where checkpoint is full dict or just state_dict
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded weights from checkpoint: {checkpoint_path}")

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    net = build_model(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = net(dummy_input)
    print("Model Architecture Initialized Successfully.")
    print(f"Input shape: {dummy_input.shape} -> Output logits shape: {out.shape}")
    print(f"Final conv layer for Grad-CAM: {net.get_last_conv_layer()}")
