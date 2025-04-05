#!/usr/bin/env python3
from lib.core.intrusion_detector import IntrusionDetector
import argparse
import os
import sys

# Ensure lib directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our main class


def main():
    """Main entry point for YOLOv11 Zone Intrusion Detection"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Zone Intrusion Detection")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument(
        "--model", type=str, default="models/yolo11n.pt", help="Path to YOLOv11 model file")

    args = parser.parse_args()

    # Create detector instance
    detector = IntrusionDetector(config_path=args.config)

    # Setup camera
    source = 0 if args.source == "0" else args.source
    if not detector.setup_camera(source):
        return

    # Setup YOLOv11 model
    model_path = args.model
    detector.setup_detector(model_path)

    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
