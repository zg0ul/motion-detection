#!/usr/bin/env python3
from lib.core.intrusion_detector import IntrusionDetector
import argparse
import os
import sys

# Ensure lib directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point for YOLOv9 Zone Intrusion Detection"""
    parser = argparse.ArgumentParser(
        description="YOLOv9 Zone Intrusion Detection System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument(
        "--model", type=str, default="models/yolov9-s-converted.pt",
        help="Path to YOLOv9 model file")
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold for detections (0-1)")

    args = parser.parse_args()

    # Create detector instance
    detector = IntrusionDetector(config_path=args.config)

    # Set confidence threshold if specified
    if args.conf != 0.4:  # Only if different from default
        detector.confidence_threshold = args.conf

    # Setup camera
    source = 0 if args.source == "0" else args.source
    if not detector.setup_camera(source):
        return

    # Setup YOLOv9 model
    model_path = args.model
    if not detector.setup_detector(model_path):
        print(f"Error: Failed to load YOLOv9 model from {model_path}")
        return

    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
