import os
import sys
import logging
import functools
import time

import colorlog
from termcolor import colored


def create_logger(log_level=logging.INFO):
    """
    Initialize internal logger.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)-5.5s]  %(message)s")

    logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    logger.setLevel(log_level)

    log_colors_config = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    color_fmt = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %('
            'message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )
    # Prevent duplicate log printing.
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setFormatter(log_formatter)
        console_handler.setFormatter(color_fmt)
        logger.addHandler(console_handler)
        file_path = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger
