#cython: language_level=3
"""Manage log

"""
import logging
import sys

class Logger:

    def __init__(self) -> None:
        # TODO: #28 ここでsetlevelを変更するようにする
        self.setlevel: None = None


    def logger(
            self,
            name: str,
            dir: str,
            setlevel: str
        ):
        """Manage log

        Args:
            name (str): File name
            dir (str): Directory name
            setlevel (str): Set level (ex. debug)

        Returns:
            Logger object: logger

        Example:
            >>> name: str = __name__
            >>> dir: str = dirname(__file__)
            >>> logger = Logger().logger(name, dir, 'info')

        """        
        self.name = name
        self.dir = dir
        self.setlevel = setlevel

        logger = logging.getLogger(self.name)

        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')
        
        log_file = dir + 'face01.log'
        file_handler = logging.FileHandler(log_file, mode='a')
        if self.setlevel == 'debug':
            logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        if self.setlevel == 'debug':
            logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger
