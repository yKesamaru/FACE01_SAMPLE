"""Example of GUI display and face recognition data output

Summary:
    In this example you can learn how to display GUI and output 
    face recognition.
    PySimpleGUI is used for GUI display. 
    See below for how to use PySimpleGUI.
    https://www.pysimplegui.org/en/latest/

Usage:
    >>> python3 this_example.py
"""

from os.path import dirname
from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt
import PySimpleGUI as sg

from face01lib.api import Dlib_api
from face01lib.Core import Core
from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.video_capture import VidCap


# Make instances
Dlib_api_obj: Dlib_api = Dlib_api()
Core_obj: Core = Core()
VidCap_obj: VidCap = VidCap()


def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()

    # Override config.ini for example
    # This method is experimental
    CONFIG = \
        Core_obj.override_args_dict(
            CONFIG,
            [
                ('anti_spoof', False),
                ('frame_skip', 10)
            ]
        )


    # Make PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
        location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
    )


    # gen = VidCap_obj.frame_generator(CONFIG)
    gen = Core_obj.common_process(
        CONFIG
    )
    
    # Repeat 'exec_times' times
    for i in range(1, exec_times):
        frame_datas_array = gen.__next__()
        print(f'exec_times: {i}')

        event, _ = window.read(timeout = 1)

        if event == sg.WIN_CLOSED:
            print("The window was closed manually")
            break

        for frame_datas in frame_datas_array:
            img: npt.NDArray[np.uint8] = frame_datas['img']
            person_data_list: List[Dict] = frame_datas['person_data_list']
            
            for person_data in person_data_list:
                if person_data == {}:
                    continue
                
                name = person_data['name']
                pict = person_data['pict']
                date = person_data['date']
                location = person_data['location']
                percentage_and_symbol = person_data['percentage_and_symbol']

                if not name == 'Unknown':
                    print(
                        name, "\n",
                        "\t", "similarity\t\t", percentage_and_symbol, "\n",
                        "\t", "coordinate\t\t", location, "\n",
                        "\t", "time\t\t\t", date, "\n",
                        "\t", "output\t\t\t", pict, "\n",
                        "-------\n"
                    )
            
                imgbytes = cv2.imencode(".png", img)[1].tobytes()
                window["display"].update(data = imgbytes)
            
        if event =='terminate':
            break
    window.close()


if __name__ == '__main__':
    main(exec_times = 50)
