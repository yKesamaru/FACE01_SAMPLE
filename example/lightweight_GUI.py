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
# import FACE01IMAGER129 as fi
# from face01lib129.video_capture import video_capture


def main(exec_times: int = 500) -> None:
    """LIGHTWEIGHT GUI application example.

    Args:
        exec_times (int, optional): Receive value of number which is processed. Defaults to 500.
    """
    # Initialize
    CONFIG: Dict =  Initialize('LIGHTWEIGHT_GUI').initialize()


    # Make PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Text('LIGHTWEIGHT GUI app sample')],
        [sg.Image(key='display')],
        [sg.Button('CAPTURE', key='capture_button', expand_x = True)],
        [sg.Button('TERMINATE', key='terminate', button_color='red')]
    ]

    window = sg.Window('LIGHTWEIGHT GUI APP', layout)

    gen = Core().common_process(CONFIG)

# Load ini file
conf = ConfigParser()
conf.read('config_FACE01GRAPHICS129.ini', 'utf-8')
similar_percentage= float(conf.get('DEFAULT','similar_percentage'))
jitters = int(conf.get('DEFAULT','jitters'))
upsampling = int(conf.get('DEFAULT','upsampling'))
mode = 'hog'
movie = conf.get('DEFAULT','movie')

kaoninshoDir, priset_face_imagesDir, check_images = fi.home()

known_face_encodings, known_face_names = fi.load_priset_image.load_priset_image(
    kaoninshoDir,
    priset_face_imagesDir, 
    jitters
)

sg.theme('Reddit')

layout = [
    [sg.Text('軽量アプリケーションのサンプル')],
    [sg.Image(key='display')],
    [sg.Button('キャプチャ', key='capture_button', expand_x = True)],
    [sg.Button('終了', key='terminate', button_color='red')]
]

window = sg.Window('lightweight_sample', layout)

vcap = video_capture(kaoninshoDir, movie)

while True:
    ret, frame = vcap.read()
    if ret == False:
        print('入力動画の信号がないため終了します')
        break

    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = int(height * (500 / width))
    small_frame = cv2.resize(frame, (500, HEIGHT))

    event, _ = window.read(timeout = 1)

    imgbytes = cv2.imencode(".png", small_frame)[1].tobytes()
    window["display"].update(data = imgbytes)

    if event=='terminate':
        break

    if event=='capture_button':
        date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")

        filename = 'check_images/lightweight_sample' + date + '.png'
        
        os.chdir(kaoninshoDir)

        cv2.imwrite(filename, small_frame)

        con = fi.Face_attestation

        xs = con.face_attestation(
            kaoninshoDir,
            check_images, 
            known_face_encodings, 
            known_face_names, 
            similar_percentage = similar_percentage,
            jitters = jitters,
            upsampling = upsampling,
            mode = mode
        )

        for x in xs:
            name, date, percentage = x['name'], x['date'], x['percentage']
            print(
                name, "\n",
                "\t", percentage, "\n",
                "\t", date, "\n",
                '-------------', "\n",
            )
            break


if __name__ == '__main__':
    # Call main function.
    main(exec_times = 500)