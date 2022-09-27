#cython: language_level=3
"""Manage log

Set log level

Args:
    str: set_level
    Chose from bellow
    - 'debug'
    - 'info'
    You can pass value as CONFIG["set_level"]

Example:
    >>> # Setup logger: common way
    >>> import os.path
    >>> from .Initialize import Initialize
    >>> name: str = __name__
    >>> dir: str = os.path.dirname(__file__)
    >>> parent_dir, _ = os.path.split(dir)
    >>> CONFIG = Initialize().initialize()
    >>> 
    >>> logger = Logger(CONFIG["log_level"]).logger(name, parent_dir)
"""        

import logging
import sys
from typing import Dict

class Logger:

    def __init__(self, log_level: str = 'info') -> None:
        self.log_level: str = log_level


    def logger(
            self,
            name: str,
            dir: str
        ):
        """Manage log

        Args:
            name (str): File name
            dir (str): Directory name

        Returns:
            Logger object: logger

        Example:
            >>> # Setup logger: common way
            >>> import os.path
            >>> from .Initialize import Initialize
            >>> name: str = __name__
            >>> dir: str = os.path.dirname(__file__)
            >>> parent_dir, _ = os.path.split(dir)
            >>> CONFIG = Initialize().initialize()
            >>> 
            >>> logger = Logger(CONFIG["log_level"]).logger(name, parent_dir)

        """        
        self.name = name
        self.dir = dir

        logger = logging.getLogger(self.name)

        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')
        
        log_file = dir + 'face01.log'
        file_handler = logging.FileHandler(log_file, mode='a')
        if self.log_level == 'debug':
            logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        if self.log_level == 'debug':
            logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger
