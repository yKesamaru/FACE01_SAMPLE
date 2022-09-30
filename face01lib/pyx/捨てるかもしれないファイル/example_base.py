import inspect
import traceback
from configparser import ConfigParser
from os import chdir
from os.path import dirname, exists
from sys import exit, getsizeof, version, version_info
from time import perf_counter
from traceback import format_exc
from typing import Dict, Generator, List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt  # See [](https://discuss.python.org/t/how-to-type-annotate-mathematical-operations-that-supports-built-in-numerics-collections-and-numpy-arrays/13509)
import PySimpleGUI as sg

# from memory_profiler import profile  # @profile()
from face01lib.api import Dlib_api

Dlib_api_obj: Dlib_api = Dlib_api()
from face01lib.Core import Core

Core_obj: Core = Core()
from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.system_check import System_check
from face01lib.video_capture import VidCap
VidCap_obj: VidCap = VidCap()


def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()

    # Logging
    name: str = __name__
    dir: str = dirname(__file__)
    logger = Logger().logger(name, dir, 'info')

    ALL_FRAME: int = exec_times

    # Override config.ini for example
    # This method is experimental
    CONFIG = \
        Core_obj.override_args_dict(
            CONFIG,
            [
                ('anti_spoof', False),
                ('frame_skip', 20)
            ]
        )


    # PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
        location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
    )

    # obj = VidCap_obj.frame_generator(CONFIG)
    obj = Core_obj.common_process(
        CONFIG
    )
    
    for i in range(1, exec_times):
        frame_datas_array = obj.__next__()
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

    logger.info(f'Predetermined number of frames: {ALL_FRAME}')


# main =================================================================
if __name__ == '__main__':
    main(exec_times = 50)
