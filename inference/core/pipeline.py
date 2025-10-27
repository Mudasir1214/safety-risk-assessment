import os
import cv2
from models.rtdetr_detector import RTDetrDetector
from models.dino_classifier import DinoClassifier
from models.vit_mobile_crane_classifier import MobileCraneRiskModel
from models.vit_tower_crane_classifier import TowerCraneRiskModel
from core.temporal_fusion_module import TemporalContextFusion
from utils.drawing import draw_text_box


def _init_video_writer(output_path, fps, width, height):
    """Create a video writer for saving annotated video results."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


class SafetyRiskPipeline:
    """Handles full video inference using multi-model integration."""

    def __init__(self, input_video,RTDETR_WEIGHTS, DINO_WEIGHTS, VIT_MOBILE_CRANE_WEIGHT, 
                 VIT_TOWER_CRANE_WEIGHT, TEMPORAL_WINDOW, OUTPUT_DIR, save_results=False, 
                 show_results=True):
        
        self.input_video = input_video
        self.save_results = save_results
        self.show_results = show_results
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)


        # Initialize models
        self.detector = RTDetrDetector(weight_model=RTDETR_WEIGHTS)
        self.dino = DinoClassifier(weight_path=DINO_WEIGHTS)
        self.vit_mobile_crane = MobileCraneRiskModel(weight_path=VIT_MOBILE_CRANE_WEIGHT)
        self.vit_tower_crane = TowerCraneRiskModel(weight_path=VIT_TOWER_CRANE_WEIGHT)
        self.tcf = TemporalContextFusion(window_size=TEMPORAL_WINDOW)

        # Setup output writer path
        base_name = os.path.basename(input_video)
        name, ext = os.path.splitext(base_name)
        self.output_path = os.path.join(OUTPUT_DIR, f"{name}_results{ext}")
        self.video_writer = None

    def run(self):
        """Run the inference pipeline."""
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.input_video}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.save_results:
            self.video_writer = _init_video_writer(self.output_path, fps, width, height)

        print(f"Processing video: {self.input_video}")
        print("Press 'q' to exit detection windows.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, det_results = self.detector.detect_objects(frame, conf_threshold=0.80)

            if not det_results:
                info_lines = ["RT-DETR: No object detected"]
            else:
                det_results = self.tcf.update(det_results)
                det_label, det_conf = det_results[0][0], det_results[0][1]
                dino_label, dino_prob = self.dino.predict(frame)
                if det_label=='mobile crane':
                    vit_label, vit_consequence, vit_score = self.vit_mobile_crane.predict(frame)
                else:
                    vit_label, vit_consequence, vit_score = self.vit_tower_crane.predict(frame)

                alert_text = "ALERT" if vit_score > 40 else "NORMAL"
                alert_color = (0, 0, 255) if vit_score > 40 else (255, 255, 255)

                
                info_lines = [
                    f"RT-DETR: {det_label} Conf:{det_conf:.2f}",
                    f"DINO: {dino_label} ({dino_prob:.2f})",
                    f"{det_label} ViT:",
                    f"  vit classified: {vit_label}",
                    f"  consequences_level: {vit_consequence}",
                    f"  consequences_score: {vit_score:.2f}",
                    f"  Warning Signal: {alert_text}",
            ]

            y_offset = 40
            for line in info_lines:
                color = (0, 0, 255) if "Warning Signal" in line else (255, 255, 255)
                draw_text_box(annotated_frame, line, (20, y_offset), color=color)
                y_offset += 30

            if self.save_results:
                self.video_writer.write(annotated_frame)

            if self.show_results:
                cv2.imshow("Safety Risk Assessment - Multi-Models Analysis", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n Processing complete. Output saved at: {self.output_path}")
