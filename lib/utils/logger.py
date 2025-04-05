import logging
import os
import datetime
import time


class LogManager:
    """Centralized logging management"""

    def __init__(self):
        # Create a unique run ID based on timestamp
        self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join("logs", f"run_{self.run_id}")

        # Create log directory for this run
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Set up the main log file
        self.main_log_file = os.path.join(self.log_dir, "application.log")

        # Initialize logger
        self._setup_logger()

        # Log startup
        self.logger.info(f"Started new session with ID: {self.run_id}")

    def _setup_logger(self):
        """Set up logging configuration"""
        # Create logger
        self.logger = logging.getLogger("YOLOv11ZoneDetector")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create handlers
        file_handler = logging.FileHandler(self.main_log_file)
        console_handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # Set formatter
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """Return the configured logger"""
        return self.logger

    def get_run_id(self):
        """Return the current run ID"""
        return self.run_id

    def get_log_dir(self):
        """Return the log directory for this run"""
        return self.log_dir


# Singleton instance
_log_manager = None


def setup_logging():
    """Set up and return the logger"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager.get_logger()


def get_log_manager():
    """Get the log manager instance"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager
