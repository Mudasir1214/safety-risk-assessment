"""
DINOv2 Processing Module
------------------------
Provides a PyTorch-based classifier using a DINOv2 Vision Transformer (ViT)
for multi-class crane safety risk assessment.
"""

# pylint: disable=too-few-public-methods, import-error, no-member, invalid-name

import sys
import warnings
from typing import Tuple, Union

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

warnings.filterwarnings("ignore", message="xFormers is not available")


class DinoVisionTransformerClassifier(nn.Module):
    """
    Vision Transformer classifier built on top of a DINOv2 backbone.
    """

    def __init__(self, backbone: nn.Module, n_classes: int = 1):
        """Initialize the ViT model with a custom classification head."""
        super().__init__()
        self.transformer = backbone
        self.classifier = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer + classifier head."""
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


class DinoClassifier:
    """
    Wrapper class for loading the DINOv2 model and performing inference.
    """

    def __init__(self, weight_path: str):
        """Initialize the DINO classifier and load weights."""
        self.class_names = [
            "Safe Operations",
            "Unsafe Operations",
            "Near-Miss Incidents",
            "Incidents",
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sys.modules[
            "__main__"
        ].DinoVisionTransformerClassifier = DinoVisionTransformerClassifier

        # Load DINOv2 Vision Transformer backbone
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dinov2_vits14 = dinov2_vits14.to(self.device)

        num_classes = len(self.class_names)
        model = DinoVisionTransformerClassifier(
            backbone=dinov2_vits14, n_classes=num_classes
        )
        model = model.to(self.device)

        # Load trained weights
        model = torch.load(weight_path, map_location=self.device)
        model.eval()
        self.model = model

        print("DINO model loaded successfully!")

    @staticmethod
    def preprocess_image(
        image_input: Union[str, Image.Image, "np.ndarray"]
    ) -> torch.Tensor:
        """
        Preprocess an image (path, PIL, or OpenCV array) into a model-ready tensor.

        Args:
            image_input: Path, PIL Image, or numpy array in BGR format.

        Returns:
            torch.Tensor: Preprocessed image tensor with batch dimension.
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            arr_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr_rgb)

        return preprocess(img).unsqueeze(0)

    def predict(
        self, image: Union[str, Image.Image, "np.ndarray"]
    ) -> Tuple[str, float]:
        """
        Run inference on an input image and return the predicted label and probability.

        Args:
            image: Path, PIL Image, or numpy array.

        Returns:
            Tuple[str, float]: (predicted_label, confidence_score)
        """
        img_tensor = self.preprocess_image(image)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)

        predicted = torch.argmax(probabilities, dim=1)
        label = self.class_names[predicted.item()]
        prob = probabilities[0][predicted].item()

        return label, prob
