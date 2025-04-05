import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import logging


class ZoneManager:
    def __init__(self):
        self.points = []
        self.drawing = False
        self.is_zone_defined = False
        self.restricted_zone = None
        self.current_intruders = set()
        self.logger = logging.getLogger("YOLOv11ZoneDetector")

    def create_roi_mask(self, frame):
        """Create a binary mask for the region of interest"""
        if not self.is_zone_defined:
            return None

        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        points_array = np.array(self.points, np.int32)
        points_array = points_array.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points_array], 255)

        return mask

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone definition"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.points = []
                self.drawing = True
            self.points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.drawing and len(self.points) > 2:
                self.drawing = False
                self.restricted_zone = Polygon(self.points)
                self.is_zone_defined = True
                self.logger.info(
                    f"Zone defined with {len(self.points)} points")

    def define_zone(self, cap):
        """Interactive zone definition mode"""
        if not cap:
            self.logger.error("Camera not initialized")
            return False

        window_name = "Define Restricted Zone - Right-click to finish"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.logger.info(
            "Zone definition mode - Left-click to add points, Right-click to finish")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw points and polygon
            if len(self.points) > 0:
                for i in range(len(self.points)):
                    cv2.circle(frame, self.points[i], 5, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(
                            frame, self.points[i-1], self.points[i], (0, 255, 0), 2)

                if self.is_zone_defined:
                    cv2.line(frame, self.points[-1],
                             self.points[0], (0, 255, 0), 2)

                    # Fill polygon with semi-transparent color
                    overlay = frame.copy()
                    points_array = np.array(self.points, np.int32)
                    points_array = points_array.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [points_array], (0, 0, 255, 64))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            cv2.putText(frame, "Left-click: Add point, Right-click: Finish", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or (self.is_zone_defined and key == ord('q')):  # ESC or Q when done
                break

        cv2.destroyWindow(window_name)

        if self.is_zone_defined:
            self.logger.info(
                f"Restricted zone defined with vertices: {self.points}")
            return True
        else:
            self.logger.warning("Zone definition canceled or incomplete")
            return False

    def check_zone_intrusion(self, detections):
        """Check if any detected objects are in the restricted zone"""
        if not self.is_zone_defined or not self.restricted_zone:
            return []

        intruders = []
        current_ids = set()

        for detection in detections:
            center_point = Point(detection['center'])

            # Check if object center is inside the polygon
            if self.restricted_zone.contains(center_point):
                intruders.append(detection)
                if detection['id'] is not None:
                    current_ids.add(detection['id'])

        # Find new intruders
        new_intruders = current_ids - self.current_intruders

        # Update current intruders
        self.current_intruders = current_ids

        return intruders, new_intruders
