import os
import logging.config

def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Only add handlers if they are not already added
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(os.getcwd(), 'logfile.log'))

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
        f_format = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
