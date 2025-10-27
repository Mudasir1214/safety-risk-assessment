"""Module for Mobile Crane risk classification using a Vision Transformer (ViT)."""

import cv2
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms


class CustomHead(nn.Module):
    """Custom classification head for Vision Transformer model."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_tensor):
        """Forward pass through the custom head."""
        x = self.fc1(input_tensor)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class MobileCraneRiskModel:
    """Model wrapper for classifying mobile crane operation risk categories."""

    def __init__(self, weight_path: str):
        """Initialize the model, load weights, and define preprocessing pipeline."""
        # Define risk data as tuples for clarity and maintainability
        classes_with_score = [
            # (class_name, consequence_score)
            ("a41_Collapse", 80),
            ("a42_Tipping over", 80),
            ("a43_Fall of crane jib, boom, other parts", 80),
            ("a44_Fall of loads", 80),
            ("a45_Collision", 80),
            ("a46_Struck-by", 80),
            ("a47_Struck-by objects", 80),
            ("a31_Potential colision", 60),
            ("a32_Potential struck-by", 60),
            ("a33_Load tipping or shifting", 60),
            ("a11_Outriggers", 20),
            ("a12_Movement", 20),
            ("a13_Lifting operation", 20),
            ("a14_Safe access to the deck", 20),
            ("a21_Operating on unsafe terrian", 40),
            ("a22_Unstable lifting operation", 40),
            ("a23_Unclear division of work area", 40),
            ("a24_Unsafe access to a working crane", 40),
            ("a25_Operating near power lines", 40),
        ]

        # Extract into separate lists (maintaining your original structure)
        self.class_names = [item[0] for item in classes_with_score]
        self.consequence_scores = [item[1] for item in classes_with_score]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ViT model and customize
        model = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.heads[0].in_features
        num_classes = len(self.class_names)
        model.heads[0] = CustomHead(num_features, num_classes)

        # Load trained weights
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = model.to(self.device)
        print("ViT model loaded successfully for mobile crane...")

        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_frame(self, frame):
        """Convert OpenCV frame (BGR) to tensor."""
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # pylint: disable=no-member
        image = self.preprocess(image).unsqueeze(0)
        return image.to(self.device)

    def predict(self, frame):
        """Predict class label, consequence level, and score for a given frame."""
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
