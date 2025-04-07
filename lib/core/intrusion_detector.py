from shapely.geometry import Polygon
import cv2
import os
import time
import datetime
import yaml
import csv
import numpy as np

# Import our modules
from lib.utils.logger import setup_logging, get_log_manager
from lib.utils.config import load_config, save_config
from lib.detectors.yolo_detector import YOLODetector
from lib.trackers.simple_tracker import SimpleTracker
from lib.core.zone_manager import ZoneManager
from lib.ui.visualizer import Visualizer


class IntrusionDetector:
    def __init__(self, config_path=None):
        # Initialize logger and log manager
        self.logger = setup_logging()
        self.log_manager = get_log_manager()
        self.log_dir = self.log_manager.get_log_dir()
        self.run_id = self.log_manager.get_run_id()

        self.logger.info("YOLOv9 Intrusion Detector initializing...")

        # Configuration
        self.confidence_threshold = 0.4  # confidence threshold for detections
        self.alert_cooldown = 5  # seconds
        self.last_alert_time = 0

        # Camera
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0

        # Initialize components
        self.zone_manager = ZoneManager()
        self.yolo_detector = None
        self.tracker = SimpleTracker()
        self.visualizer = Visualizer()

        # Intrusion logging
        self.intrusion_log = []

        # Create CSV file for intrusion data
        self.intrusion_csv_file = os.path.join(self.log_dir, "intrusions.csv")
        self._initialize_csv()

        # Create directory for intrusion images
        self.intrusion_images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.intrusion_images_dir, exist_ok=True)

        # Load configuration
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        try:
            with open(self.intrusion_csv_file, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'id', 'label',
                              'confidence', 'x', 'y', 'width', 'height']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            self.logger.info(
                f"Initialized intrusion CSV log: {self.intrusion_csv_file}")
        except Exception as e:
            self.logger.error(f"Error initializing CSV log: {e}")

    def load_config(self, config_path):
        """Load configuration from file"""
        config = load_config(config_path)

        # If config is empty (file not found or invalid), create default config
        if not config:
            self.logger.warning(
                f"Could not load config from {config_path}, creating default configuration")
            new_config_path = os.path.join(self.log_dir, "zone_config.json")
            from lib.utils.config import create_default_config
            create_default_config(new_config_path)
            config = load_config(new_config_path)
            self.logger.info(
                f"Created default configuration at {new_config_path}")

        if config:
            self.confidence_threshold = config.get('confidence_threshold', 0.4)
            self.alert_cooldown = config.get('alert_cooldown', 5)

            # Load predefined zone if available
            if 'zone_points' in config and config['zone_points']:
                self.zone_manager.points = config['zone_points']
                self.zone_manager.restricted_zone = Polygon(
                    self.zone_manager.points)
                self.zone_manager.is_zone_defined = True

            self.logger.info(
                f"Configuration loaded with threshold={self.confidence_threshold}, cooldown={self.alert_cooldown}")
            if self.zone_manager.is_zone_defined:
                self.logger.info(
                    f"Zone loaded with {len(self.zone_manager.points)} points")

    def save_config(self, config_path):
        """Save current configuration to file"""
        # Save to the logs directory instead of root
        config_path = os.path.join(self.log_dir, os.path.basename(config_path))

        config = {
            'confidence_threshold': self.confidence_threshold,
            'alert_cooldown': self.alert_cooldown,
            'zone_points': self.zone_manager.points if self.zone_manager.is_zone_defined else []
        }

        save_config(config_path, config)
        self.logger.info(f"Configuration saved to {config_path}")

    def setup_camera(self, source=0):
        """Initialize camera or video source"""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.logger.error(f"Error opening video source {source}")
            return False

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(
            f"Camera initialized: {self.frame_width}x{self.frame_height}")
        return True

    def setup_detector(self, model_path):
        """Initialize the YOLO detector"""
        self.yolo_detector = YOLODetector(
            model_path, self.confidence_threshold)
        return self.yolo_detector.setup_model()

    def log_intrusion(self, detection):
        """Log intrusion events"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        intrusion_data = {
            'timestamp': timestamp,
            'object_type': detection['label'],
            'confidence': detection['confidence'],
            'position': detection['center'],
            'id': detection['id']
        }

        self.intrusion_log.append(intrusion_data)
        self.logger.warning(
            f"INTRUSION: {intrusion_data['object_type']} #{intrusion_data['id']} detected in restricted zone at {timestamp}"
        )

        # Trigger alert if cooldown has passed
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.trigger_alert(detection)
            self.last_alert_time = current_time

        # Store intrusion data in CSV
        self.save_intrusion_to_csv(detection)

    def save_intrusion_to_csv(self, detection):
        """Save intrusion data to CSV file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self.intrusion_csv_file, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'id', 'label',
                              'confidence', 'x', 'y', 'width', 'height']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                x, y, w, h = detection['box']
                center_x, center_y = detection['center']

                writer.writerow({
                    'timestamp': timestamp,
                    'id': detection['id'],
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h
                })

                self.logger.debug(
                    f"Intrusion recorded in CSV: {detection['label']} #{detection['id']}")
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")

    def trigger_alert(self, detection):
        """Trigger intrusion alert"""
        self.logger.critical(
            f"⚠️ ALERT: {detection['label']} #{detection['id']} intrusion detected! Position: {detection['center']}"
        )

        # Save intrusion image if we have a current frame
        if hasattr(self, 'current_frame'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.intrusion_images_dir, f"intrusion_{detection['id']}_{timestamp}.jpg")
            cv2.imwrite(filename, self.current_frame)
            self.logger.info(f"Intrusion image saved: {filename}")

    def run(self):
        """Main detection loop"""
        if not self.cap:
            self.logger.error("Camera not initialized")
            return

        if not self.zone_manager.is_zone_defined:
            self.logger.warning("No restricted zone defined")
            success = self.zone_manager.define_zone(self.cap)
            if not success:
                return

        self.logger.info("Starting YOLOv9 zone intrusion detection...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read frame from camera")
                break

            self.current_frame = frame.copy()

            # Create ROI mask for detection
            roi_mask = self.zone_manager.create_roi_mask(frame)

            # Detect objects using YOLOv9
            detections = []
            if self.yolo_detector and self.yolo_detector.model is not None:
                try:
                    # Only detect within the ROI
                    detections = self.yolo_detector.detect_objects(
                        frame, roi_mask)
                except Exception as e:
                    self.logger.error(f"YOLOv9 detection error: {e}")

            # Update tracking
            detections = self.tracker.update(detections)

            # Check for zone intrusions
            intruders, new_intruder_ids = self.zone_manager.check_zone_intrusion(
                detections)

            # Log new intrusions
            for detection in intruders:
                if detection['id'] in new_intruder_ids:
                    self.log_intrusion(detection)

            # Visualization - with each step modifying the frame
            frame_vis = frame.copy()

            # Draw the zone
            is_secure = len(intruders) == 0
            frame_vis = self.visualizer.draw_zone(
                frame_vis, self.zone_manager.points, is_secure)

            # Draw detections and intruders
            frame_vis = self.visualizer.draw_detections(
                frame_vis, detections, intruders)

            # Add status info
            status_text = "Status: SECURE" if is_secure else f"Status: INTRUSION DETECTED ({len(intruders)})"
            frame_vis = self.visualizer.add_status_info(
                frame_vis, status_text, is_secure, " - YOLOv9 DETECTION")

            # Add timestamp and help text
            frame_vis = self.visualizer.add_timestamp(frame_vis)
            help_text = "Press 'q' to quit, 's' to save config"
            frame_vis = self.visualizer.add_help_text(frame_vis, help_text)

            # Display the frame
            cv2.imshow("YOLOv9 Zone Intrusion Detection", frame_vis)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Save config with 's'
            if key == ord('s'):
                self.save_config("zone_config.yaml")
                self.logger.info("Configuration saved")

            # Quit with 'q' or ESC
            elif key == ord('q') or key == 27:
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Save intrusion log to YAML file in the log directory
        if self.intrusion_log:
            log_file = os.path.join(self.log_dir, "intrusions.yaml")
            try:
                with open(log_file, 'w') as file:
                    yaml.dump(self.intrusion_log, file)
                self.logger.info(f"Intrusion log saved to {log_file}")
            except Exception as e:
                self.logger.error(f"Error saving intrusion log: {e}")
