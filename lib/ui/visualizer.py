import cv2
import numpy as np
import datetime


class Visualizer:
    def __init__(self):
        self.frame = None

    def draw_detections(self, frame, detections, intruders=None, highlight_color=(0, 0, 255)):
        """Draw detection boxes and labels on the frame"""
        self.frame = frame.copy()

        if intruders is None:
            intruders = []

        # Create a set of intruder IDs for faster lookup
        intruder_ids = {det['id']
                        for det in intruders if 'id' in det and det['id'] is not None}

        # Draw detection boxes
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            detection_id = detection.get('id')

            # Default color for regular detections
            color = (0, 255, 0)  # Green

            # Highlight intruders
            if detection_id in intruder_ids:
                color = highlight_color  # Red for intruders

            # Draw bounding box
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)

            # Draw center point (used for zone checking)
            cv2.circle(self.frame, detection['center'], 4, (255, 0, 0), -1)

            # Draw label with ID if available
            if detection_id is not None:
                label_text = f"{label} #{detection_id}: {confidence:.2f}"
            else:
                label_text = f"{label}: {confidence:.2f}"

            cv2.putText(self.frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self.frame

    def draw_zone(self, frame, points, is_secure=True, fill_alpha=0.3):
        """Draw the restricted zone on the frame"""
        self.frame = frame.copy()

        if not points or len(points) < 3:
            return self.frame

        # Draw polygon outline
        points_array = np.array(points, np.int32)
        points_array = points_array.reshape((-1, 1, 2))
        cv2.polylines(self.frame, [points_array], True, (0, 255, 0), 2)

        # Fill polygon with semi-transparent color based on security status
        overlay = self.frame.copy()
        if is_secure:
            fill_color = (0, 255, 0)  # Green for secure
        else:
            fill_color = (0, 0, 255)  # Red for intrusion

        cv2.fillPoly(overlay, [points_array], fill_color)
        cv2.addWeighted(overlay, fill_alpha, self.frame,
                        1 - fill_alpha, 0, self.frame)

        return self.frame

    def add_status_info(self, frame, status_text, is_secure=True, detector_status=""):
        """Add status information to the frame"""
        self.frame = frame.copy()

        # Combine status text with detector status
        combined_status = f"{status_text}{detector_status}"

        # Set color based on security status
        if is_secure:
            status_color = (0, 255, 0)  # Green for secure
        else:
            status_color = (0, 0, 255)  # Red for intrusion

        # Add status text
        cv2.putText(self.frame, combined_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return self.frame

    def add_timestamp(self, frame):
        """Add current timestamp to the frame"""
        self.frame = frame.copy()

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(self.frame, timestamp, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return self.frame

    def add_help_text(self, frame, text):
        """Add help text to the frame"""
        self.frame = frame.copy()

        # Add help text
        cv2.putText(self.frame, text,
                    (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return self.frame
