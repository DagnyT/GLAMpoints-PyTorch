import os
from .loggers import FileLogger
from .loggers import TensorboardLogger

def build_logger(cfg):

    LOG_DIR = os.path.join(cfg['LOGGING']['LOG_DIR'])

    return FileLogger(LOG_DIR), TensorboardLogger(LOG_DIR)