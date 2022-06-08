import logging
from logging.handlers import TimedRotatingFileHandler
import sys

def init_logger(name: str = None) -> None:
    """
    Initializes root logger for package.

    Args:
        name: package name.
        log_format: logging format.
    """

    if not name:
        name = __name__
    FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = 'log.txt'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)
