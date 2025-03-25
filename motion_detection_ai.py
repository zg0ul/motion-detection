import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

# Global variables for ROI selection
roi_points = []
roi_selection_complete = False
temp_frame = None

def click_and_crop(event, x, y, flags, param):
    """Mouse callback function for ROI selection"""
    global roi_points, roi_selection_complete, temp_frame
    
    # If ROI selection is already complete, do nothing
    if roi_selection_complete:
        return
        
    # If left mouse button clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        
        # Draw circles at the clicked points
        if len(roi_points) > 0:
            cv2.circle(temp_frame, roi_points[-1], 5, (0, 255, 0), -1)
            
        # Draw lines between consecutive points
        if len(roi_points) > 1:
            cv2.line(temp_frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            
        # Update display
        cv2.imshow("Select ROI", temp_frame)
    
    # If right mouse button clicked, close the polygon
    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:
        # Close the polygon by connecting the last point to the first
        cv2.line(temp_frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
        roi_selection_complete = True
        cv2.imshow("Select ROI", temp_frame)
        print("ROI selection complete. Press any key to continue...")

def select_roi(frame):
    """Function to allow the user to select a polygon ROI"""
    global roi_points, roi_selection_complete, temp_frame
    
    # Reset ROI selection variables
    roi_points = []
    roi_selection_complete = False
    
    # Create a copy of the frame to draw on
    temp_frame = frame.copy()
    
    # Create a window and set the mouse callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", click_and_crop)
    
    # Display instructions
    print("Select ROI by clicking points. Right-click to complete the selection.")
    instruction_text = "LEFT CLICK: add point, RIGHT CLICK: complete polygon (min 3 points)"
    cv2.putText(temp_frame, instruction_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame and wait for ROI selection
    cv2.imshow("Select ROI", temp_frame)
    
    # Keep window open until ROI selection is complete
    while not roi_selection_complete:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            return None
    
    # Wait for a key press before continuing
    cv2.waitKey(0)
    cv2.destroyWindow("Select ROI")
    
    return np.array(roi_points, dtype=np.int32)

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def calculate_center(bbox):
    """Calculate center point of a bounding box [x, y, w, h]"""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def main():
    video_path = 'motion_detection/test_cars.mp4'
    video_cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_cap.isOpened():
        print("Error: Could not open video source")
        return

    # Configure YOLO11 model
    # Load YOLO11 model - you can choose different sizes: n, s, m, l, x
    try:
        model = YOLO("yolo11n.pt")  # Load the model (will download if not present)
        print("YOLO11n model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO11 model: {e}")
        print("Make sure you have installed ultralytics: pip install ultralytics")
        return
        
    # Set classes to detect - focusing only on vehicles
    # Class list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco128.yaml
    # 2: car, 5: bus, 7: truck
    vehicle_classes = [2, 5, 7]
    
    # Detection parameters
    conf_threshold = 0.4  # Confidence threshold
    
    # Get first frame for ROI selection
    success, frame = video_cap.read()
    if not success:
        print("Failed to read from video source")
        return
    
    if frame.shape[1] > 1280:  # Resize if too large
        frame = cv2.resize(frame, (1280, 720))
    
    # Let the user select a polygon ROI
    roi_polygon = select_roi(frame)
    if roi_polygon is None or len(roi_polygon) < 3:
        print("Invalid ROI selection. Exiting...")
        return
    
    # Vehicle tracking variables
    tracked_vehicles = {}  # Dictionary to store tracked vehicles
    next_vehicle_id = 0    # Counter for vehicle IDs
    vehicle_count = 0      # Total count of vehicles
    
    # Set up display window
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            start_time = time.time()  # For FPS calculation
            
            # Read a frame
            success, frame = video_cap.read()
            if not success:
                print("End of video or failed to read frame")
                break

            # Resize frame if needed
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, 720))
            
            # Draw the ROI polygon
            cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 2)
            
            # Run YOLOv8 inference
            results = model(frame, conf=conf_threshold, classes=vehicle_classes)
            
            # Process detections
            current_time = time.time()
            current_active_vehicles = set()
            detections_in_roi = 0
            
            # Extract detection results
            if len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Calculate center point
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Check if detection is in ROI
                    if point_in_polygon((center_x, center_y), roi_polygon):
                        # Get class name - just use "Vehicle" for all types to avoid classification changes
                        class_name = "Vehicle"
                        # Store the actual class for debugging but don't show it
                        actual_class = "Car" if cls_id == 2 else "Bus" if cls_id == 5 else "Truck" if cls_id == 7 else "Vehicle"
                            
                        # Draw bounding box with label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                        
                        detections_in_roi += 1
                        
                        # Vehicle tracking logic
                        matched = False
                        for v_id, vehicle_data in list(tracked_vehicles.items()):
                            prev_x, prev_y, last_seen, counted = vehicle_data
                            
                            # Calculate distance between centers
                            distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            
                            # If close enough, consider it the same vehicle
                            if distance < 50:  # Adjust threshold as needed
                                # Update tracking information
                                tracked_vehicles[v_id] = (center_x, center_y, current_time, counted)
                                current_active_vehicles.add(v_id)
                                matched = True
                                break
                        
                        # If no match found, create new vehicle ID
                        if not matched:
                            tracked_vehicles[next_vehicle_id] = (center_x, center_y, current_time, False)
                            current_active_vehicles.add(next_vehicle_id)
                            next_vehicle_id += 1
            
            # Process tracked vehicles for counting
            roi_height = np.max(roi_polygon[:, 1]) - np.min(roi_polygon[:, 1])
            count_line_y = np.min(roi_polygon[:, 1]) + roi_height // 2
            
            # Draw counting line
            roi_min_x = np.min(roi_polygon[:, 0])
            roi_max_x = np.max(roi_polygon[:, 0])
            cv2.line(frame, (roi_min_x, count_line_y), (roi_max_x, count_line_y), (255, 0, 0), 2)
            
            # Initialize the previous position dictionary if it doesn't exist
            if not hasattr(main, 'prev_position'):
                main.prev_position = {}
                
            # Update vehicle count for vehicles crossing the line
            for v_id in list(tracked_vehicles.keys()):
                if v_id in current_active_vehicles:
                    # Vehicle is still active
                    center_x, center_y, last_seen, counted = tracked_vehicles[v_id]
                    
                    # If vehicle has crossed the counting line and hasn't been counted yet
                    if v_id in main.prev_position:
                        prev_x, prev_y = main.prev_position[v_id]
                        
                        # Draw tracking line for debugging
                        cv2.line(frame, (prev_x, prev_y), (center_x, center_y), (0, 255, 255), 1)
                        
                        # Check if it crossed the line (from top to bottom or bottom to top)
                        if not counted and ((prev_y < count_line_y and center_y >= count_line_y) or 
                            (prev_y > count_line_y and center_y <= count_line_y)):
                            tracked_vehicles[v_id] = (center_x, center_y, last_seen, True)
                            vehicle_count += 1
                            print(f"Vehicle {v_id} counted! Total: {vehicle_count}")
                    
                    # Store current position for next frame comparison
                    main.prev_position[v_id] = (center_x, center_y)
                else:
                    # Vehicle is no longer visible
                    center_x, center_y, last_seen, counted = tracked_vehicles[v_id]
                    if current_time - last_seen > 1.0:  # Remove after 1 second of not seeing
                        del tracked_vehicles[v_id]
                        if hasattr(main, 'prev_position') and v_id in main.prev_position:
                            del main.prev_position[v_id]
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Display info on frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Detections in ROI: {detections_in_roi}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow("YOLOv8 Detection", frame)
            
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
        video_cap.release()
        cv2.destroyAllWindows()
        # Ensure all windows close properly on some systems
        for i in range(1, 5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()