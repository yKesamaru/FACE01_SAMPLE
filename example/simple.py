"""Example of simple face recognition script.

Summary:

Usage:
    >>> python3 simple.py
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


def main(exec_times: int = 50):
    """Simple example.

    This simple example script prints out results of face recognition process.

    Args:
        exec_times (int, optional): Number of frames for process. Defaults to 50.
    """    
    # Initialize
    CONFIG: Dict =  Initialize('DEFAULT', 'info').initialize()


    # Make generator
    gen = Core().common_process(CONFIG)
    

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    print(
                        person_data['name'], "\n",
                        "\t", "similarity\t\t", person_data['percentage_and_symbol'], "\n",
                        "\t", "coordinate\t\t", person_data['location'], "\n",
                        "\t", "time\t\t\t", person_data['date'], "\n",
                        "\t", "output\t\t\t", person_data['pict'], "\n",
                        "-------\n"
                    )


if __name__ == '__main__':
    # Call main function. Pass 5.
    main(exec_times = 5)