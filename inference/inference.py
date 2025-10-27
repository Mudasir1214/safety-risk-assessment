"""
Safety Risk Assessment - Multi-Model Video Analysis
---------------------------------------------------
Command-line tool for real-time or batch video safety risk assessment
using RT-DETR (detection), DINO (classification), and MobileViT (risk prediction).

Usage:
    python inference.py --input_video ./videos/sample.mp4 --save_results True --show_results False
"""

import os
from utils.args_parser import parse_arguments
from core.pipeline import SafetyRiskPipeline
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR","./all_saved_models")
RTDETR_WEIGHTS = os.path.join(MODEL_DIR, os.getenv('RTDETR_WEIGHTS'))
DINO_WEIGHTS = os.path.join(MODEL_DIR, os.getenv('DINO_WEIGHTS'))
VIT_MOBILE_CRANE_WEIGHT = os.path.join(MODEL_DIR, os.getenv('VIT_MOBILE_CRANE_WEIGHT'))
VIT_TOWER_CRANE_WEIGHT = os.path.join(MODEL_DIR, os.getenv('VIT_TOWER_CRANE_WEIGHT'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR','./output_results')
TEMPORAL_WINDOW = int(os.getenv('TEMPORAL_WINDOW','5'))


def main():
    # --- Parse command-line arguments ---
    args = parse_arguments()

    # --- Initialize the pipeline ---
    pipeline = SafetyRiskPipeline(
        RTDETR_WEIGHTS=RTDETR_WEIGHTS,
        DINO_WEIGHTS=DINO_WEIGHTS,
        VIT_MOBILE_CRANE_WEIGHT=VIT_MOBILE_CRANE_WEIGHT,
        VIT_TOWER_CRANE_WEIGHT=VIT_TOWER_CRANE_WEIGHT,
        TEMPORAL_WINDOW=TEMPORAL_WINDOW,
        OUTPUT_DIR=OUTPUT_DIR,
        input_video=args.input_video,
        save_results=args.save_results,
        show_results=args.show_results,
    )

    # --- Run inference ---
    pipeline.run()


if __name__ == "__main__":
    main()
