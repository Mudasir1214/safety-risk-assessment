"""
RT-DETR Stream Processing Module
--------------------------------
Provides a simple wrapper for running inference using the RT-DETR model
from the Ultralytics framework. Handles detection and model initialization.
"""

from ultralytics import RTDETR
import cv2


class RTDetrDetector:
    """RT-DETR detector class for performing object detection on video frames."""

    def __init__(self, weight_model: str):
        """
        Initialize the RT-DETR model for inference.

        Args:
            weight_model (str): Path to the RT-DETR weight file.
        """
        self.model = RTDETR(model=weight_model)
        print("RT-DETR model loaded successfully!")

    def detect_objects(self, frame, conf_threshold: float = 0.25):
        """
        Run inference on a single frame and return detection results.

        Args:
            frame: The image frame for inference.
            confidence_threshold (float): Minimum confidence score for valid detections.

        Returns:
            results: RT-DETR model prediction results.
        """

        detections_results = self.model.predict(frame, conf=conf_threshold)
        det_results = []

        for result in detections_results:
            boxes = result.boxes  # Boxes object
            
            if len(boxes) == 0:
                return frame, det_results
            else:
                det_results = []
                for box in boxes:
                    # Extract coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls_id]
                    
                    det_results.append((label, round(conf, 2), [x1, y1, x2, y2]))


                    # Draw bounding box on frame
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )

            return frame, det_results

       