import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# Global variables for ROI selection
roi_points = []
roi_selection_complete = False
temp_frame = None


def click_and_crop(event, x, y, flags, param):
    """Mouse callback function for ROI selection"""
    global roi_points, roi_selection_complete, temp_frame

    if roi_selection_complete:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

        if len(roi_points) > 0:
            cv2.circle(temp_frame, roi_points[-1], 3, (0, 255, 0), -1)

        if len(roi_points) > 1:
            cv2.line(temp_frame, roi_points[-2],
                     roi_points[-1], (0, 255, 0), 2)

        cv2.imshow("Select ROI", temp_frame)

    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:
        cv2.line(temp_frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
        roi_selection_complete = True
        cv2.imshow("Select ROI", temp_frame)
        print("ROI selection complete. Press any key to continue...")


def select_roi(frame):
    """Function to allow the user to select a polygon ROI"""
    global roi_points, roi_selection_complete, temp_frame

    roi_points = []
    roi_selection_complete = False

    temp_frame = frame.copy()

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", click_and_crop)

    print("Select ROI by clicking points. Right-click to complete the selection.")
    instruction_text = "LEFT CLICK: add point, RIGHT CLICK: complete polygon (min 3 points)"
    cv2.putText(temp_frame, instruction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Select ROI", temp_frame)

    while not roi_selection_complete:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            return None

    cv2.waitKey(0)
    cv2.destroyWindow("Select ROI")

    return np.array(roi_points, dtype=np.int32)


def create_roi_mask(frame, roi_polygon_points):
    """Create a binary mask for the ROI polygon"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon_points], 255)
    return mask


def point_in_polygon(x, y, polygon):
    """Check if a point is inside a polygon using a simple ray casting algorithm"""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def setup_csv_logger(filename="vehicle_detection_log.csv"):
    """Set up CSV logger for vehicle detection events"""
    # Open in 'w' mode instead of 'a' to overwrite existing file
    csv_file = open(filename, 'w', newline='')
    fieldnames = ['timestamp', 'track_id', 'class',
                  'confidence', 'x_center', 'y_center', 'width', 'height']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Always write the header since we're creating a new file
    writer.writeheader()

    return csv_file, writer


# Simple tracker class
class SimpleTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        # How many frames a track can exist without being matched
        self.max_age = max_age
        # Minimum number of matches needed before a track is considered valid
        self.min_hits = min_hits
        # Minimum overlap required to match a detection to a track
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}  # id -> {box, class_id, confidence, age, hits, last_pos}

    def _calc_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Check if there's no intersection
        if x2 < x1 or y2 < y1:
            return 0.0

        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # IoU = intersection / union
        return intersection / float(box1_area + box2_area - intersection)

    def update(self, detections):
        """Update tracks with new detections
        detections: List of [x1, y1, x2, y2, confidence, class_id]
        Returns: List of [x1, y1, x2, y2, track_id]
        """
        # First, increment age for all tracks
        for track_id in self.tracks:
            self.tracks[track_id]['age'] += 1

        # If no detections, return active tracks and remove expired ones
        if len(detections) == 0:
            result = []
            for track_id, track in list(self.tracks.items()):
                if track['age'] <= self.max_age and track['hits'] >= self.min_hits:
                    # Return predicted track
                    result.append([*track['box'], track_id])

                # Remove old tracks
                if track['age'] > self.max_age:
                    del self.tracks[track_id]

            return np.array(result) if result else np.empty((0, 5))

        # Match detections to existing tracks
        det_boxes = np.array([d[:4] for d in detections])
        det_scores = np.array([d[4] for d in detections])
        det_classes = np.array([d[5] for d in detections])

        # Calculate IoU for all combinations of tracks and detections
        matched_indices = []
        unmatched_detections = list(range(len(detections)))

        # For each track, find the best matching detection
        for track_id, track in list(self.tracks.items()):
            if track['age'] > self.max_age:
                del self.tracks[track_id]
                continue

            track_box = track['box']
            max_iou = self.iou_threshold
            match_idx = -1

            for i, det_idx in enumerate(unmatched_detections):
                iou = self._calc_iou(track_box, det_boxes[det_idx])
                if iou > max_iou:
                    max_iou = iou
                    match_idx = i

            if match_idx >= 0:
                # Found a match
                det_idx = unmatched_detections[match_idx]
                matched_indices.append((track_id, det_idx))

                # Update track
                self.tracks[track_id].update({
                    'box': det_boxes[det_idx],
                    'class_id': det_classes[det_idx],
                    'confidence': det_scores[det_idx],
                    'age': 0,
                    'hits': track['hits'] + 1
                })

                # Remove this detection from unmatched list
                unmatched_detections.pop(match_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks[self.next_id] = {
                'box': det_boxes[det_idx],
                'class_id': det_classes[det_idx],
                'confidence': det_scores[det_idx],
                'age': 0,
                'hits': 1,
                'last_pos': (
                    (det_boxes[det_idx][0] + det_boxes[det_idx][2]) // 2,
                    (det_boxes[det_idx][1] + det_boxes[det_idx][3]) // 2
                )
            }
            self.next_id += 1

        # Prepare result: active tracks
        result = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                result.append([*track['box'], track_id])

        return np.array(result) if result else np.empty((0, 5))


def main():
    video_path = 'motion_detection/vid.mp4'
    video_cap = cv2.VideoCapture(video_path)

    # Parameters
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for YOLO detections

    # Object classes we're interested in (from COCO dataset)
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    try:
        # Set up CSV logger
        csv_file, csv_writer = setup_csv_logger()

        # Read first frame to get dimensions
        success, frame = video_cap.read()
        if not success:
            print("Failed to read from video source")
            exit()

        # Resize if needed
        frame = cv2.resize(frame, (1280, 720))

        # Let the user select a polygon ROI
        roi_polygon = select_roi(frame)
        if roi_polygon is None or len(roi_polygon) < 3:
            print("Invalid ROI selection. Exiting...")
            exit()

        # Create ROI mask for visualization
        roi_mask = create_roi_mask(frame, roi_polygon)

        # Initialize YOLO model
        print("Loading YOLO model...")
        model = YOLO('yolo11n.pt')

        # Initialize our simple tracker
        tracker = SimpleTracker(max_age=15, min_hits=3, iou_threshold=0.3)

        # For stats
        vehicle_count = 0
        counted_ids = set()  # Keep track of IDs that have been counted
        fps = 0
        fps_start_time = time.time()
        fps_frame_count = 0

        # Store class information for each track
        track_metadata = defaultdict(dict)

        print("Starting detection and tracking...")
        while True:
            success, frame = video_cap.read()
            if not success:
                print("End of video or failed to read frame")
                break

            # Resize if needed
            frame = cv2.resize(frame, (1280, 720))

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:  # Update FPS every 10 frames
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Draw ROI polygon
            cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 2)

            # Run YOLO detection on the entire frame
            results = model(frame, verbose=False)

            # Extract detections only inside ROI
            roi_detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # Calculate center of the detection
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Check if center is inside ROI and class is a vehicle
                    if (point_in_polygon(center_x, center_y, roi_polygon) and 
                        class_id in VEHICLE_CLASSES and 
                        confidence > CONFIDENCE_THRESHOLD):
                        roi_detections.append([x1, y1, x2, y2, confidence, class_id])

            # Update tracker with ROI detections only
            tracks = tracker.update(roi_detections)

            # Process tracking results
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Find the matching detection to get class and confidence
                matched_detection = None
                for det in roi_detections:
                    det_x1, det_y1, det_x2, det_y2 = det[:4]
                    iou_val = tracker._calc_iou(
                        [x1, y1, x2, y2], [det_x1, det_y1, det_x2, det_y2])
                    if iou_val > 0.5:  # If IoU is high enough, consider it a match
                        matched_detection = det
                        break

                # Update metadata if we have a match
                if matched_detection is not None:
                    class_id = int(matched_detection[5])
                    confidence = matched_detection[4]
                    track_metadata[track_id] = {
                        'class_id': class_id,
                        'confidence': confidence
                    }

                # Get class info from metadata or use defaults
                class_id = track_metadata.get(track_id, {}).get('class_id', -1)
                confidence = track_metadata.get(track_id, {}).get('confidence', 0.0)

                # Get class name for display
                class_names = {2: 'Car', 3: 'Motorcycle',
                               5: 'Bus', 7: 'Truck', -1: 'Vehicle'}
                class_name = class_names.get(class_id, 'Vehicle')

                # Draw bounding box and ID (all tracks are in ROI by definition now)
                color = (0, 255, 0)  # Green for all objects in ROI
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw ID and class
                label = f"ID:{track_id} {class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check if this is a new vehicle to count
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_count += 1

                    # Log to CSV
                    csv_writer.writerow({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'track_id': int(track_id),
                        'class': class_name,
                        'confidence': float(confidence),
                        'x_center': center_x,
                        'y_center': center_y,
                        'width': (x2 - x1),
                        'height': (y2 - y1)
                    })

            # Show vehicle count and FPS
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Visualize ROI mask (optional)
            roi_vis = cv2.bitwise_and(frame, frame, mask=roi_mask)
            cv2.imshow("ROI Only", roi_vis)

            # Show main frame
            cv2.imshow("ROI Vehicle Detection", frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' key
                print("User requested exit")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up resources")
        if 'csv_file' in locals():
            csv_file.close()
        video_cap.release()
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()