import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "docuscan") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create logs directory if not exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exports")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "docuscan.log")

    # File Handler
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file logging: {e}")

    return logger
