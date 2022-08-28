#cython: language_level=3
import logging

class Logger:
    def __init__(self) -> None:
        self.setlevel = None

    def logger(self, name, dir, setlevel):
        self.name = name
        self.dir = dir
        self.setlevel = setlevel

        logger = logging.getLogger(self.name)

        # logger.setLevel(logging.INFO)
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

        stream_handler = logging.StreamHandler()
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
