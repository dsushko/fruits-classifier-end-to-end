import logging
from logging.handlers import TimedRotatingFileHandler
import sys

def _set_logging():
    FORMATTER = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    LOG_FILE = 'log.txt'
    logging.root.setLevel(logging.DEBUG)
    console_handler = logging.root.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logging.root.addHandler(console_handler)
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    logging.root.addHandler(file_handler)

