import yaml
import logging
import os
import json
import re


def _convert_yaml_tuples_to_lists(yaml_content):
    """Convert Python tuple representation in YAML to standard list format"""
    # Replace !!python/tuple tags with empty strings
    modified_content = re.sub(r'!!python/tuple\s*\n', '', yaml_content)
    return modified_content


def load_config(config_path):
    """Load configuration from file"""
    try:
        # Check if file exists and has content
        if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
            logger = logging.getLogger("YOLOv11ZoneDetector")
            logger.warning(
                f"Configuration file {config_path} is empty or does not exist")
            return {}

        # Determine file type by extension
        is_json = config_path.lower().endswith('.json')

        with open(config_path, 'r') as file:
            raw_config = file.read()

            # Ensure we have content
            if not raw_config.strip():
                logger = logging.getLogger("YOLOv11ZoneDetector")
                logger.warning(f"Configuration file {config_path} is empty")
                return {}

            config = None

            # Try to load as JSON first if it's a JSON file
            if is_json:
                try:
                    config = json.loads(raw_config)
                except Exception as json_error:
                    logger = logging.getLogger("YOLOv11ZoneDetector")
                    logger.error(f"Failed to parse JSON config: {json_error}")
                    return {}
            else:
                # For YAML files, preprocess to handle Python tuples
                try:
                    # Check if this is a YAML file with Python tuples
                    if '!!python/tuple' in raw_config:
                        # This is a YAML file with Python tuples, convert them to lists
                        logger.info(
                            "Converting YAML with Python tuples to standard format")
                        modified_yaml = _convert_yaml_tuples_to_lists(
                            raw_config)
                        config = yaml.safe_load(modified_yaml)

                        # Save as JSON for future use
                        json_path = os.path.splitext(config_path)[0] + '.json'
                        with open(json_path, 'w') as json_file:
                            json.dump(config, json_file, indent=2)
                        logger.info(
                            f"Converted YAML config to JSON and saved at {json_path}")
                    else:
                        # Regular YAML file
                        config = yaml.safe_load(raw_config)
                except Exception as yaml_error:
                    logger = logging.getLogger("YOLOv11ZoneDetector")
                    logger.error(f"Failed to parse YAML config: {yaml_error}")
                    return {}

            # Convert zone points from lists back to tuples if they exist
            if config and 'zone_points' in config and isinstance(config['zone_points'], list):
                config['zone_points'] = [tuple(point) if isinstance(point, list) else point
                                         for point in config['zone_points']]

        logger = logging.getLogger("YOLOv11ZoneDetector")
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except Exception as e:
        logger = logging.getLogger("YOLOv11ZoneDetector")
        logger.error(f"Error loading configuration: {e}")
        return {}


def save_config(config_path, config):
    """Save current configuration to file"""
    try:
        # Prepare a copy of the config to modify
        config_copy = config.copy()

        # If zone_points exist in config, convert tuples to lists for safe serialization
        if 'zone_points' in config_copy and isinstance(config_copy['zone_points'], list):
            config_copy['zone_points'] = [list(point) if isinstance(point, tuple) else point
                                          for point in config_copy['zone_points']]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(
            os.path.abspath(config_path)), exist_ok=True)

        # Always save as JSON which handles list serialization well
        # Force .json extension
        json_path = os.path.splitext(config_path)[0] + '.json'
        with open(json_path, 'w') as file:
            json.dump(config_copy, file, indent=2)

        logger = logging.getLogger("YOLOv11ZoneDetector")
        logger.info(f"Configuration saved to {json_path}")
        return True
    except Exception as e:
        logger = logging.getLogger("YOLOv11ZoneDetector")
        logger.error(f"Error saving configuration: {e}")
        return False


def create_default_config(config_path):
    """Create a default configuration file if one doesn't exist"""
    default_config = {
        'confidence_threshold': 0.3,
        'alert_cooldown': 5,
        'zone_points': []
    }

    return save_config(config_path, default_config)
