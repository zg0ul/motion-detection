import cv2
import imutils
import time
import numpy as np

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

def create_roi_mask(frame, roi_polygon):
    """Create a binary mask for the ROI polygon"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    return mask

def main():
    video_path = 'motion_detection/vid.mp4'
    video_cap = cv2.VideoCapture(video_path)

    # Motion detection parameters - easily adjustable
    THRESHOLD_VALUE = 25       # Lower = more sensitive, Higher = less sensitive
    MIN_AREA = 700             # Minimum contour area to consider
    BLUR_SIZE = (7, 7)         # Blur kernel size for noise reduction
    HISTORY_FRAMES = 100       # For background subtractor - more history for stability
    LEARNING_RATE = 0.005      # How quickly background model adapts to changes (smaller = slower)

    try:
        # Read first frame to get dimensions
        success, frame = video_cap.read()
        if not success:
            print("Failed to read from video source")
            exit()
            
        frame = cv2.resize(frame, (1920, 1080))
        
        # Let the user select a polygon ROI
        roi_polygon = select_roi(frame)
        if roi_polygon is None or len(roi_polygon) < 3:
            print("Invalid ROI selection. Exiting...")
            exit()
        
        # Create ROI mask
        roi_mask = create_roi_mask(frame, roi_polygon)
        
        # Get ROI bounding box for visualization
        x, y, w, h = cv2.boundingRect(roi_polygon)
        
        # Create a background subtractor with shadow detection
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=HISTORY_FRAMES, 
            varThreshold=50, 
            detectShadows=True  # Enable shadow detection
        )
        
        # Create kernels for morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))  # Horizontally biased for vehicles
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Larger kernel to merge nearby detections
        
        # Skip some initial frames to let background model stabilize
        for _ in range(30):  # Process more frames for better initialization
            success, frame = video_cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (1920, 1080))
            
            # Apply mask to get just the ROI
            roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            # Apply background subtraction only to the masked region
            bg_subtractor.apply(roi_frame, learningRate=LEARNING_RATE)
        
        # Counter for vehicle detection
        vehicle_count = 0
        last_detection_time = time.time()
        
        # For vehicle tracking
        tracked_vehicles = {}
        next_vehicle_id = 0
        
        while True:
            success, frame = video_cap.read()
            if not success:
                print("End of video or failed to read frame")
                break

            frame = cv2.resize(frame, (1920, 1080))
            
            # Apply the ROI mask
            roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            # Apply background subtraction to masked region
            fg_mask = bg_subtractor.apply(roi_frame, learningRate=LEARNING_RATE)
            
            # Remove shadows (they are marked as gray (127) in the mask)
            shadow_mask = (fg_mask == 127)
            fg_mask[shadow_mask] = 0  # Set shadows to black (background)
            
            # Convert to binary mask (foreground only)
            _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            # Apply noise reduction with Gaussian blur
            blurred_mask = cv2.GaussianBlur(binary_mask, BLUR_SIZE, 0)
            _, binary_mask = cv2.threshold(blurred_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            
            # Apply advanced morphological operations
            # 1. Opening to remove small noise (erosion followed by dilation)
            morph_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
            
            # 2. Closing to fill small holes inside detected objects - use larger kernel to merge nearby detections
            morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            
            # 3. Another opening with larger kernel to better separate objects
            morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel_large)
            
            # Find contours in the processed mask
            cnts = cv2.findContours(morph_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            motion_detected = False
            current_time = time.time()
            current_active_vehicles = set()
            
            # Draw the ROI polygon on the main frame
            cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 2)

            # Merge overlapping bounding boxes
            merged_contours = []
            for c in cnts:
                # Filter contours by area
                contour_area = cv2.contourArea(c)
                if contour_area < MIN_AREA:
                    continue
                    
                # Add to merged contours list for further processing
                merged_contours.append(c)
                
            # Apply non-maximum suppression to remove duplicate detections
            # Convert contours to bounding boxes for NMS
            boxes = [cv2.boundingRect(c) for c in merged_contours]
            if boxes:
                # Convert to format needed for NMS: [x, y, x+w, y+h]
                nms_boxes = [[x, y, x+w, y+h] for (x, y, w, h) in boxes]
                # Calculate areas for each box
                areas = [w*h for (_, _, w, h) in boxes]
                
                # Apply NMS
                indices = []
                while len(nms_boxes) > 0:
                    # Get index of largest area
                    idx = areas.index(max(areas))
                    indices.append(idx)
                    
                    # Remove overlapping boxes
                    new_nms_boxes = []
                    new_areas = []
                    
                    for i, box in enumerate(nms_boxes):
                        if i == idx:
                            continue
                        
                        # Calculate IoU
                        xx1 = max(nms_boxes[idx][0], box[0])
                        yy1 = max(nms_boxes[idx][1], box[1])
                        xx2 = min(nms_boxes[idx][2], box[2])
                        yy2 = min(nms_boxes[idx][3], box[3])
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        
                        overlap = float(w * h) / (areas[i])
                        
                        # If overlap is less than threshold, keep the box
                        if overlap < 0.3:
                            new_nms_boxes.append(box)
                            new_areas.append(areas[i])
                    
                    nms_boxes = new_nms_boxes
                    areas = new_areas
                
                # Keep only the selected contours
                merged_contours = [merged_contours[i] for i in indices]
                
            # Process filtered contours
            for c in merged_contours:
                    
                # Get bounding box
                (x_cnt, y_cnt, w_cnt, h_cnt) = cv2.boundingRect(c)
                
                # Additional filtering - aspect ratio check for vehicles
                aspect_ratio = float(w_cnt) / h_cnt
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue  # Skip objects with unusual aspect ratios
                
                # Get the center of the contour
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    # Fallback to bounding box center if moments calculation fails
                    center_x = x_cnt + w_cnt // 2
                    center_y = y_cnt + h_cnt // 2
                
                # Draw bounding box directly on the frame (not just in ROI)
                cv2.rectangle(frame, (x_cnt, y_cnt), (x_cnt+w_cnt, y_cnt+h_cnt), (0, 255, 0), 2)
                
                # Add contour area as text
                cv2.putText(frame, f"Area: {contour_area:.0f}", (x_cnt, y_cnt-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Check if this matches a vehicle we're already tracking
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
                    # Create a new vehicle ID (no edge condition since we have a custom ROI)
                    tracked_vehicles[next_vehicle_id] = (center_x, center_y, current_time, False)
                    current_active_vehicles.add(next_vehicle_id)
                    next_vehicle_id += 1
                
                motion_detected = True
            
            # Count vehicles that have moved significantly through the ROI
            for v_id in list(tracked_vehicles.keys()):
                if v_id in current_active_vehicles:
                    # Vehicle is still active
                    center_x, center_y, last_seen, counted = tracked_vehicles[v_id]
                    
                    # If vehicle is in the central region of the polygon and hasn't been counted yet
                    # This is a simplified approach - you may need to refine the counting logic for your specific ROI
                    if point_in_polygon_center(center_x, center_y, roi_polygon) and not counted:
                        tracked_vehicles[v_id] = (center_x, center_y, last_seen, True)
                        vehicle_count += 1
                else:
                    # Vehicle is no longer visible
                    center_x, center_y, last_seen, counted = tracked_vehicles[v_id]
                    if current_time - last_seen > 0.5:  # Remove after 0.5 seconds of not seeing
                        del tracked_vehicles[v_id]
            
            # Display detection status
            if motion_detected:
                cv2.putText(frame, "Motion DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Show vehicle count
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
            
            # Show frames
            cv2.imshow("Motion Detection", frame)
            cv2.imshow("ROI Threshold", morph_mask)  # Show processed mask for debugging
            
            # Optionally show shadow mask for debugging
            shadow_visible = shadow_mask.astype(np.uint8) * 255
            cv2.imshow("Shadow Mask", shadow_visible)
            
            # Key handling
            key = cv2.waitKey(100) & 0xFF
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
        for i in range(1, 5):
            cv2.waitKey(1)

def point_in_polygon_center(x, y, polygon, center_percentage=0.6):
    """Check if a point is in the central region of the polygon"""
    # First check if point is in polygon at all
    if cv2.pointPolygonTest(polygon, (x, y), False) < 0:
        return False
    
    # Get bounding box of polygon
    x_min, y_min, w, h = cv2.boundingRect(polygon)
    
    # Calculate center region
    margin_x = w * (1 - center_percentage) / 2
    margin_y = h * (1 - center_percentage) / 2
    
    # Check if point is in center region
    if (x_min + margin_x <= x <= x_min + w - margin_x and
        y_min + margin_y <= y <= y_min + h - margin_y):
        return True
    
    return False

if __name__ == "__main__":
    main()