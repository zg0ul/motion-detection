import cv2
import os
import torch
import numpy as np
import logging
import sys
import traceback


class YOLODetector:
    def __init__(self, model_path, confidence_threshold=0.3):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.yolov9_modules = {}
        self.logger = logging.getLogger("YOLOv9Detector")

    def _import_yolov9_modules(self):
        """Import required YOLOv9 modules dynamically"""
        # Determine the path to the YOLOv9 repo
        yolov9_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "yolov9")

        if not os.path.exists(yolov9_dir):
            self.logger.error(f"YOLOv9 directory not found at {yolov9_dir}")
            return False

        # Add YOLOv9 directory to system path
        if yolov9_dir not in sys.path:
            sys.path.append(yolov9_dir)

        try:
            # Import required modules
            from yolov9.models.common import DetectMultiBackend # type: ignore
            from yolov9.utils.general import non_max_suppression, scale_boxes # type: ignore
            from yolov9.utils.augmentations import letterbox # type: ignore

            # Store modules for later use
            self.yolov9_modules = {
                'DetectMultiBackend': DetectMultiBackend,
                'non_max_suppression': non_max_suppression,
                'scale_boxes': scale_boxes,
                'letterbox': letterbox
            }
            return True
        except ImportError as e:
            self.logger.error(f"Error importing YOLOv9 modules: {e}")
            return False

    def setup_model(self):
        """Initialize YOLOv9 model"""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False

            # Import required YOLOv9 modules
            if not self._import_yolov9_modules():
                self.logger.error("Failed to import YOLOv9 modules")
                return False

            # Store original torch.load to restore it later
            original_torch_load = torch.load

            # Patch torch.load to use weights_only=False (required for PyTorch 2.6+)
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)

            # Apply the patch
            torch.load = patched_torch_load

            try:
                # Simply load the model using DetectMultiBackend
                self.model = self.yolov9_modules['DetectMultiBackend'](
                    self.model_path)
                self.logger.info(
                    f"YOLOv9 model loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading YOLOv9 model: {e}")
                self.logger.error(traceback.format_exc())
                return False
            finally:
                # Always restore the original torch.load
                torch.load = original_torch_load

        except Exception as e:
            self.logger.error(f"Error initializing YOLOv9 model: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def detect_objects(self, frame, roi_mask=None):
        """Detect objects in frame using YOLOv9"""
        if not self.model:
            return []

        # Apply ROI mask if provided
        if roi_mask is not None:
            # Create a masked frame for detection (only ROI area)
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            detection_frame = masked_frame
        else:
            detection_frame = frame

        detections = []

        try:
            # Preserve original for drawing
            im0 = detection_frame.copy()

            # Preprocess image according to YOLOv9 requirements
            # Use letterbox function from YOLOv9 to properly resize the image
            img = self.yolov9_modules['letterbox'](
                im0, 640, stride=32, auto=True)[0]

            # Convert to the correct format
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.model.device)
            img = img.float() / 255.0

            if len(img.shape) == 3:
                img = img.unsqueeze(0)

            # Run inference
            pred = self.model(img)

            # Apply non-maximum suppression
            pred = self.yolov9_modules['non_max_suppression'](
                pred, self.confidence_threshold, 0.45, classes=None, agnostic=False, max_det=1000)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = self.yolov9_modules['scale_boxes'](
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Process detections
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        confidence = float(conf)
                        class_id = int(cls)

                        # Get the class label
                        if hasattr(self.model, 'names') and class_id in self.model.names:
                            label = self.model.names[class_id]
                        else:
                            label = f"class_{class_id}"

                        # Convert to our format
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                        center_x, center_y = int(x + w/2), int(y + h/2)

                        detections.append({
                            'box': (x, y, w, h),
                            'center': (center_x, center_y),
                            'label': label,
                            'confidence': confidence,
                            'id': None  # Will be assigned by tracker
                        })

        except Exception as e:
            self.logger.error(f"Error in YOLOv9 detection: {e}")
            self.logger.error(traceback.format_exc())
            return []

        return detections
