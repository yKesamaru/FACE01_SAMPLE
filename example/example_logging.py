"""Example of benchmark for face recognition

Summary:
    In this example, you can learn about log functions.


Usage:
    >>> python3 logging.py

NOTE:
    Befor running, change False to True 'output_debug_log = False' in config.ini
"""

# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)


from typing import Dict

from face01lib.Core import Core
from face01lib.Initialize import Initialize
from face01lib.logger import Logger


def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()


    if CONFIG["headless"] == False:
        print("""
        For this example, set config.ini as follows.
            > [MAIN] 
            > headless = True 
        """)
        exit()


    # Setup logger
    log_level: str = 'debug'
    import os.path
    name: str = __name__
    dir: str = os.path.dirname(__file__)
    parent_dir, _ = os.path.split(dir)

    logger = Logger(log_level).logger(name, parent_dir)


    gen = Core().common_process(CONFIG)
    

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    logger.debug(person_data['name'])
                    logger.debug(person_data['percentage_and_symbol'])
                    logger.debug(person_data['location'])
                    logger.debug("-----------------")
            
    
if __name__ == '__main__':
    main(exec_times = 50)
