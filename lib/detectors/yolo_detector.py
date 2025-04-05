import cv2
import os
import torch
import numpy as np
import logging

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. Please install with:")
    print("pip install ultralytics")
    import sys
    sys.exit(1)


class YOLODetector:
    def __init__(self, model_path, confidence_threshold=0.3):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.logger = logging.getLogger("YOLOv11ZoneDetector")

    def setup_model(self):
        """Initialize YOLOv11 model"""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False

            # Load YOLOv11 model using ultralytics
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLOv11 model loaded from {self.model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing YOLOv11 model: {e}")
            return False

    def detect_objects(self, frame, roi_mask=None):
        """Detect objects in frame using YOLOv11"""
        if not self.model:
            return []

        # If ROI mask is provided, only detect within that region
        if roi_mask is not None:
            # Create a masked frame for detection (only ROI area)
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            # Run YOLOv11 inference on masked region
            results = self.model(
                masked_frame, conf=self.confidence_threshold, verbose=False)
        else:
            # Run YOLOv11 inference on whole frame
            results = self.model(
                frame, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process results
        if results[0].boxes and len(results[0].boxes) > 0:
            for i, detection in enumerate(results[0].boxes):
                # Get bounding box in (x1, y1, x2, y2) format
                box = detection.xyxy[0].cpu().numpy()
                confidence = detection.conf[0].item()
                class_id = int(detection.cls[0].item())
                label = results[0].names[class_id]

                # Convert to our format
                x1, y1, x2, y2 = box
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                center_x, center_y = int(x + w/2), int(y + h/2)

                detections.append({
                    'box': (x, y, w, h),
                    'center': (center_x, center_y),
                    'label': label,
                    'confidence': confidence,
                    'id': None  # Will be assigned by tracker
                })

        return detections
