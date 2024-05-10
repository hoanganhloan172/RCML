import os
import logging
from pathlib import Path

def setup_logger(name, log_dir, file_name, level=logging.INFO):
    '''
    Setup logger
    '''

    # Create the path if it does not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_dir, file_name)

    handler = logging.FileHandler(log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
