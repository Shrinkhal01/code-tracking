import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    log_dir = "logs"
    log_file = os.path.join(log_dir, "backup.log")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("BackupLogger")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
