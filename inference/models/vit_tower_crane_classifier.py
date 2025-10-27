"""Module for Tower Crane risk classification using a Vision Transformer (ViT)."""

import cv2
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms


class CustomHead(nn.Module):
    """Custom classification head for the Vision Transformer model."""

    def __init__(self, in_features: int, num_classes: int):
        """Initialize the classification head with two linear layers."""
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_tensor):
        """Forward pass through the custom head."""
        x = self.fc1(input_tensor)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class TowerCraneRiskModel:
    """Model wrapper for tower crane operation risk classification."""

    def __init__(self, weight_path: str):
        """Initialize model, load pretrained weights, and define preprocessing."""
        classes_with_score = [
            # (class_name, consequence_score)
            ("b41_Collapse", 80),
            ("b42_Tipping over", 80),
            ("b43_Fall of crane jib, boom, other parts", 80),
            ("b44_Fall of load", 80),
            ("b45_Collision", 80),
            ("b46_Struck-by", 80),
            ("b47_Struck by objects", 80),
            ("b31_Potential collision", 60),
            ("b32_Potential struck-by", 60),
            ("b33_Load tipping or shifting", 60),
            ("b11_Assembling, dismantling", 20),
            ("b12_Erection", 20),
            ("b13_Lifting operation", 20),
            ("b14_Safe access to the deck", 20),
            ("b21_Inadequate use of PPE", 40),
            ("b22_Unstable lifting operation", 40),
            ("b23_Unclear division of work area", 40),
            ("b24_Unsafe access to a working crane", 40),
            ("b25_Operating near power lines", 40),
        ]

        # Extract into separate lists (maintaining your original structure)
        self.class_names = [item[0] for item in classes_with_score]
        self.consequence_scores = [item[1] for item in classes_with_score]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ViT model
        model = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )

        # Freeze base parameters
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.heads[0].in_features
        num_classes = len(self.class_names)
        model.heads[0] = CustomHead(num_features, num_classes)

        # Load trained weights
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = model.to(self.device)
        print("ViT model loaded successfully for tower crane...")

        # Freeze all parameters (inference mode)
        for param in self.model.parameters():
            param.requires_grad = False

        # Define preprocessing transformations
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_frame(self, frame):
        """Convert an OpenCV frame (BGR) to a normalized tensor."""
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # pylint: disable=no-member
        image = self.preprocess(image).unsqueeze(0)
        return image.to(self.device)

    def predict(self, frame):
        """Predict class, consequence level, and score for a given frame."""
        image_tensor = self.preprocess_frame(frame)

        with torch.no_grad():
            output = self.model(image_tensor)  # pylint: disable=not-callable
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()

        predicted_class = self.class_names[output.argmax().item()]

        consequence_score = float(
            sum(
                prob * self.consequence_scores[i]
                for i, prob in enumerate(probabilities)
            )
        )

        if consequence_score <= 20:
            consequence_level = "Negligible Consequences (Safety Classification)"
        elif 20 < consequence_score <= 40:
            consequence_level = "Mild Consequences (Unsafe Classification)"
        elif 40 < consequence_score <= 60:
            consequence_level = "Moderate Consequences (Near Miss Incident)"
        else:
            consequence_level = "Severe Consequences (Incident)"

        return predicted_class, consequence_level, round(consequence_score, 2)
