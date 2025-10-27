"""
Model Weight Auto-Downloader
----------------------------
Checks if all required model weight files exist in MODEL_DIR.
If missing, automatically downloads them from Google Drive using gdown.

Usage:
    python download_models.py
"""

import os
import gdown
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Base model directory
MODEL_DIR = os.getenv("MODEL_DIR", "./all_saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# List of model files from .env
MODEL_FILES = {
    "RTDETR_WEIGHTS": os.getenv("RTDETR_WEIGHTS", "detr_detection_best_model.pt"),
    "DINO_WEIGHTS": os.getenv("DINO_WEIGHTS", "dino_model_epoch20.pth"),
    "VIT_MOBILE_CRANE_WEIGHT": os.getenv("VIT_MOBILE_CRANE_WEIGHT", "vit_mobile_crane_activity_model_epoch10.pth"),
    "VIT_TOWER_CRANE_WEIGHT": os.getenv("VIT_TOWER_CRANE_WEIGHT", "vit_tower_crane_activity_model_epoch10.pth"),
}
# Corresponding Google Drive file IDs (replace these with your actual IDs)
MODEL_FILE_IDS = {
    "RTDETR_WEIGHTS": "1Odv2qBCuY2mK0lvycsXxnRkj1_P-7OB8",
    "DINO_WEIGHTS": "1C9uuT6Ex-f-2PPFpPmqjKDEluoxREj4K",
    "VIT_MOBILE_CRANE_WEIGHT": "1EBDqUXN2AMI1EZmWTNuXzVgIMOk0wH-T",
    "VIT_TOWER_CRANE_WEIGHT": "1hg2BQEUuP5jn0XQcvEOVKzT8PdBDXCdm",
}


def download_from_drive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {os.path.basename(output_path)} ...")
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded successfully: {output_path}\n")
    except Exception as e:
        print(f"Failed to download {output_path}: {e}\n")


def ensure_models_exist():
    """Ensure all required model weights are available."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Checking model directory: {MODEL_DIR}\n")

    for key, filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, os.path.basename(filename))
        if not os.path.exists(model_path):
            print(f"Missing file: {filename}")
            file_id = MODEL_FILE_IDS.get(key)
            if file_id and file_id != "YOUR_RTDETR_FILE_ID":
                download_from_drive(file_id, model_path)
            else:
                print(f"No valid file ID provided for {key}\n")
        else:
            print(f"Found file: {filename}")
    print("\nAll model files checked.")


if __name__ == "__main__":
    ensure_models_exist()
