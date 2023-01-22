"""Example of LIGHTWEIGHT GUI window.

Summary:
    In this example, you can learn how to make LIGHTWEIGHT GUI application.
    PySimpleGUI is used for GUI display. 
    See below for how to use PySimpleGUI.
    https://www.pysimplegui.org/en/latest/

Usage:
    >>> python3 lightweight_GUI.py
"""
# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)


import os
from configparser import ConfigParser
from datetime import datetime
from typing import Dict

import cv2
import PySimpleGUI as sg

from face01lib.Core import Core
from face01lib.Initialize import Initialize
from face01lib.logger import Logger

# Initialize
CONFIG: Dict =  Initialize('LIGHTWEIGHT_GUI', 'info').initialize()
# Set up logger
logger = Logger(CONFIG['log_level']).logger(__file__, CONFIG['RootDir'])
"""Initialize and Setup logger.
When coding a program that uses FACE01, code `initialize` and `logger` first.
This will read the configuration file `config.ini` and log errors etc.
"""


def main(exec_times: int = 500) -> None:
    """LIGHTWEIGHT GUI application example.

    Args:
        exec_times (int, optional): Receive value of number which is processed. Defaults to 500.
    """
    # Make PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Text('LIGHTWEIGHT GUI app sample')],
        [sg.Image(key='display')],
        [sg.Button('CAPTURE', key='capture_button', expand_x = True)],
        [sg.Button('TERMINATE', key='terminate', button_color='red', expand_x = True)]
    ]

    window = sg.Window('LIGHTWEIGHT GUI APP', layout)

    # Make generator
    gen = Core().common_process(CONFIG)

    # Repeat 'exec_times' times
    for i in range(0, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        event, _ = window.read(timeout = 1)  # type: ignore

        for frame_datas in frame_datas_array:
            if event == sg.WIN_CLOSED:
                logger.info("The window was closed manually")
                exit(0)
            
            if event=='terminate':
                exit(0)
            
            imgbytes = cv2.imencode(".png", frame_datas['img'])[1].tobytes()
            window["display"].update(data = imgbytes)
            
            if event=='capture_button':
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
    # Call main function.
    main(exec_times = 500)