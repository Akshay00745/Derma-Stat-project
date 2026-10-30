"""
Native PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation.

Generates visual heatmaps highlighting the influential dermoscopic image regions
that drove the model's diagnostic classification for any chosen class.

Compatible with EfficientNetB0 and standard convolutional architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM explanation generator for PyTorch vision models.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # If model implements get_last_conv_layer
            if hasattr(model, "get_last_conv_layer"):
                self.target_layer = model.get_last_conv_layer()
            elif hasattr(model, "model") and hasattr(model.model, "features"):
                self.target_layer = model.model.features[-1]
            else:
                raise ValueError("Target convolutional layer must be specified for Grad-CAM.")
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates the raw Grad-CAM heatmap array [H, W] normalized in [0, 1].

        Args:
            input_tensor: [1, 3, H, W] normalized input tensor
            target_class: Target class index. If None, uses model's top predicted class.
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        # Backward pass for the target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global Average Pooling on gradients
        # gradients shape: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to only capture features that positively contribute to the target class
        cam = F.relu(cam)

        # Interpolate heatmap to input image dimensions
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def create_gradcam_overlay(
    original_pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a Grad-CAM heatmap onto the original PIL image.

    Returns:
        (heatmap_rgb_uint8, overlay_rgb_uint8)
    """
    orig_w, orig_h = original_pil_image.size

    # Resize heatmap to match the original image aspect ratio and resolution
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heatmap_arr = np.array(heatmap_resized) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(heatmap_arr)[:, :, :3]  # RGBA -> RGB [0, 1]
    colored_cam_uint8 = (colored_cam * 255).astype(np.uint8)

    # Convert original to RGB numpy array
    orig_np = np.array(original_pil_image.convert("RGB"))

    # Blend original image and colored heatmap
    overlay = (alpha * colored_cam_uint8 + (1.0 - alpha) * orig_np).astype(np.uint8)

    return colored_cam_uint8, overlay
