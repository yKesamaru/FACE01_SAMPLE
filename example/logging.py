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

from face01lib.Calc import Cal
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


    name = __name__
    if CONFIG["output_debug_log"] == True:
        logger = Logger().logger(name, dir, 'debug')
    else:
        logger = Logger().logger(name, dir, 'info')


    gen = Core().common_process(CONFIG)
    

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    logger.
                    print(
                        person_data['name'], "\n",
                        "\t", "similarity\t\t", person_data['percentage_and_symbol'], "\n",
                        "\t", "coordinate\t\t", person_data['location'], "\n",
                        "\t", "time\t\t\t", person_data['date'], "\n",
                        "\t", "output\t\t\t", person_data['pict'], "\n",
                        "-------\n"
                    )
            
    
    END = Cal.Measure_processing_time_backward()

    print(f'Total processing time: {round(Cal.Measure_processing_time(START, END) , 3)}[seconds]')
    print(f'Per frame: {round(Cal.Measure_processing_time(START, END) / ( exec_times), 3)}[seconds]')


if __name__ == '__main__':
    pr.run('main(exec_times = 50)', 'restats')
    subprocess.run(["snakeviz", "restats"])
