import logging
from logging.handlers import TimedRotatingFileHandler
import sys

loggers = {}

def get_logger(name):
    if name not in loggers:
        FORMATTER = logging.Formatter("%(asctime)s- %(name)s - %(levelname)s - %(message)s")
        LOG_FILE = 'log.txt'
        logger = logging.getLogger('fruits-classifier')
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.addHandler(console_handler)
        file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)
        loggers[name] = logger
    return loggers[name]

