"""
Temporal Context Fusion (TCF)
-----------------------------
Lightweight module for enhancing temporal consistency in object detection.
It maintains a sliding window of previous frame detections and smooths
bounding box positions and confidence scores.
"""

from collections import deque
import numpy as np


class TemporalContextFusion:
    def __init__(self, window_size=5):
        """
        Args:
            window_size (int): Number of previous frames to use for temporal fusion.
        """
        self.buffer = deque(maxlen=window_size)

    def update(self, detections):
        """
        Smooths detection results using a temporal buffer.
        Args:
            detections (list): List of [label, confidence, bbox] or similar tuples.

        Returns:
            fused_detections (list): Temporally smoothed detections.
        """
        if not detections or len(detections) == 0:
            return detections

        # Store current detections
        self.buffer.append(detections)

        # Not enough history yet
        if len(self.buffer) < 2:
            return detections

        # --- Temporal fusion: smooth confidences and boxes ---
        fused = []
        num_objects = len(detections)

        for obj_idx in range(num_objects):
            label, conf, bbox = detections[obj_idx]

            # Collect past confs and boxes
            conf_history = []
            bbox_history = []
            for frame_dets in self.buffer:
                if len(frame_dets) > obj_idx:
                    conf_history.append(frame_dets[obj_idx][1])
                    bbox_history.append(frame_dets[obj_idx][2])

            avg_conf = float(np.mean(conf_history))
            avg_bbox = np.mean(np.array(bbox_history), axis=0).tolist()

            fused.append((label, avg_conf, avg_bbox))

        return fused
