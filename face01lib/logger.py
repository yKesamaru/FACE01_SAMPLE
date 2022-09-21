#cython: language_level=3
import logging
# from memory_profiler import profile
import sys

class Logger:
    def __init__(self) -> None:
        # TODO: #28 ここでsetlevelを変更するようにする
        self.setlevel: None = None


    # @profile(precision=4)
    def logger(self, name, dir, setlevel):
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
