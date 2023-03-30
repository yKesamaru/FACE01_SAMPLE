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

from typing import Dict

import cv2
import PySimpleGUI as sg

from face01lib.Core import Core
from face01lib.Initialize import Initialize


def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()


    # Make PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE',
        layout, alpha_channel = 1,
        margins=(10, 10),
        location=(0, 0),
        modal = True,
        titlebar_icon="./images/g1320.png",
        icon="./images/g1320.png"
    )


    gen = Core().common_process(CONFIG)
    

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()
        print(f'exec_times: {i}')

        event, _ = window.read(timeout = 1)

        if event == sg.WIN_CLOSED:
            print("The window was closed manually")
            break

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
            
                imgbytes = cv2.imencode(".png", frame_datas['img'])[1].tobytes()
                window["display"].update(data = imgbytes)
            
        if event =='terminate':
            break
    window.close()


if __name__ == '__main__':
    main(exec_times = 50)
