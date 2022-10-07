import os
from configparser import ConfigParser
from datetime import datetime

import cv2
import PySimpleGUI as sg

import FACE01IMAGER129 as fi
from face01lib129.video_capture import video_capture

# iniファイル読み込み
conf=ConfigParser()
conf.read('config_FACE01GRAPHICS129.ini', 'utf-8')
similar_percentage= float(conf.get('DEFAULT','similar_percentage'))
jitters=            int(conf.get('DEFAULT','jitters'))
upsampling=         int(conf.get('DEFAULT','upsampling'))
mode=               'hog'
movie=              conf.get('DEFAULT','movie')

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
    [sg.Button('キャプチャ', key='capture_button', expand_x=True)],
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

    event, _ = window.read(timeout=1)

    imgbytes=cv2.imencode(".png", small_frame)[1].tobytes()
    window["display"].update(data=imgbytes)

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
            similar_percentage=similar_percentage,
            jitters=jitters,
            upsampling=upsampling,
            mode=mode
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
