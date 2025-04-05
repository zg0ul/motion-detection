import cv2
import numpy as np
import logging


class MotionDetector:
    def __init__(self, history=500, threshold=25, detect_shadows=True):
        self.history = history
        self.threshold = threshold
        self.detect_shadows = detect_shadows
        self.background_subtractor = None
        self.logger = logging.getLogger("YOLOv11ZoneDetector")
        self.initialize()

    def initialize(self):
        """Initialize the background subtractor"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.threshold,
            detectShadows=self.detect_shadows
        )
        self.logger.info("Motion detector initialized")

    def reset(self):
        """Reset the background subtractor"""
        self.initialize()
        self.logger.info("Motion detector reset")

    def detect_motion(self, frame, min_area=500):
        """Detect motion in the frame"""
        if self.background_subtractor is None:
            self.initialize()

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Remove shadows (gray pixels)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Noise removal
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                detection = {
                    'box': (x, y, w, h),
                    'center': (center_x, center_y),
                    'label': 'Motion',
                    'confidence': area / 5000,  # Normalize area as confidence
                    'id': None  # Will be assigned by tracker
                }
                detections.append(detection)

        return detections
