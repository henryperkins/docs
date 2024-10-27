import logging
import logging.handlers

def setup_logging(log_file: str, log_level: str = "INFO", formatter_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """Sets up logging configuration."""
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.hasHandlers():
        # Create a rotating file handler
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)  # Log all levels to file

            # Create a formatter
            formatter = logging.Formatter(formatter_str)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except (IOError, OSError) as e:
            print(f"Failed to set up file handler for logging: {e}")
            return False  # Indicate failure

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Log INFO and above to console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return True  # Indicate success