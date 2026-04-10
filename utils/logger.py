import csv
import logging
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
_log_dir = "artifacts/logs"
_file_handler = None


def set_log_dir(path):
    global _log_dir, _file_handler
    old_log_dir = _log_dir
    _log_dir = path
    os.makedirs(_log_dir, exist_ok=True)

    if _file_handler is not None:
        old_path = _file_handler.baseFilename
        logger.removeHandler(_file_handler)
        _file_handler.close()
        if os.path.exists(old_path) and os.path.getsize(old_path) == 0:
            os.remove(old_path)

    log_path = os.path.join(_log_dir, f"run_{TIMESTAMP}.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    _file_handler = logging.FileHandler(log_path)
    _file_handler.setFormatter(formatter)
    logger.addHandler(_file_handler)


def setup_logger(name="nn_project"):
    global _file_handler
    os.makedirs(_log_dir, exist_ok=True)
    log_path = os.path.join(_log_dir, f"run_{TIMESTAMP}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    _file_handler = logging.FileHandler(log_path)
    _file_handler.setFormatter(formatter)
    logger.addHandler(_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class CSVLogger:
    def __init__(self, name, columns):
        os.makedirs(_log_dir, exist_ok=True)
        path = os.path.join(_log_dir, f"{name}_{TIMESTAMP}.csv")
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)

    def log(self, *values):
        self.writer.writerow(values)
        self.file.flush()

    def close(self):
        self.file.close()


logger = setup_logger()
