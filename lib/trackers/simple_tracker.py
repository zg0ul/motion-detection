import numpy as np
import logging


class SimpleTracker:
    def __init__(self):
        self.next_track_id = 0
        self.prev_detections = []
        self.logger = logging.getLogger("YOLOv11ZoneDetector")

    def update(self, detections, distance_threshold=50):
        """Update tracking IDs for new detections"""
        # If this is the first frame with detections, assign new IDs to all
        if not self.prev_detections:
            for detection in detections:
                detection['id'] = self.next_track_id
                self.next_track_id += 1
            self.prev_detections = detections.copy()
            return detections

        # For subsequent frames, try to match detections with previous ones
        # based on centroids distance
        for detection in detections:
            min_dist = float('inf')
            matching_id = None

            # Current detection center
            cx, cy = detection['center']

            # Find the closest previous detection
            for prev_detection in self.prev_detections:
                prev_cx, prev_cy = prev_detection['center']
                # Calculate Euclidean distance
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

                # If the distance is below threshold and smaller than previous matches
                if dist < distance_threshold and dist < min_dist:
                    min_dist = dist
                    matching_id = prev_detection['id']

            # Assign ID: either matched or new
            if matching_id is not None:
                detection['id'] = matching_id
            else:
                detection['id'] = self.next_track_id
                self.next_track_id += 1

        # Update previous detections for next frame
        self.prev_detections = detections.copy()

        return detections
