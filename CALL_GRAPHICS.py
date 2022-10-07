import configparser
from concurrent.futures import ThreadPoolExecutor

import cv2
import PySimpleGUI as sg

import FACE01GRAPHICS129 as fg

# configファイル読み込み
conf=configparser.ConfigParser()
conf.read('config_FACE01GRAPHICS129.ini', 'utf-8')

kaoninshoDir, priset_face_imagesDir=fg.home()

known_face_encodings, known_face_names=fg.load_priset_image(
    kaoninshoDir,
    priset_face_imagesDir, 
    # jitters=                        int(conf.get('DEFAULT','priset_face_images_jitters')) 
)

xs=fg.face_attestation(
    kaoninshoDir, 
    known_face_encodings, 
    known_face_names, 
    similar_percentage=             float(conf.get('DEFAULT','similar_percentage')), 
    jitters=                        int(conf.get('DEFAULT','jitters')), 
    upsampling=                     int(conf.get('DEFAULT','upsampling')),
    mode=                           conf.get('DEFAULT','mode'), 
    model=                          'small', 
    frame_skip=                     int(conf.get('DEFAULT','frame_skip')), 
    movie=                          conf.get('DEFAULT','movie'), 
    rectangle=                      conf.getboolean('DEFAULT','rectangle'), 
    target_rectangle=               conf.getboolean('DEFAULT','target_rectangle'), 
    show_video=                     conf.getboolean('DEFAULT','show_video'),
    frequency_crop_image=           int(conf.get('DEFAULT','frequency_crop_image')), 
    set_area=                       conf.get('DEFAULT','set_area'),
    print_property=                 conf.getboolean('DEFAULT','print_property'),
    calculate_time=                 conf.getboolean('DEFAULT','calculate_time'),
    SET_WIDTH=                      int(conf.get('DEFAULT','SET_WIDTH')),
    default_face_image_draw=        conf.getboolean('DEFAULT', 'default_face_image_draw'),
    show_overlay=                   conf.getboolean('DEFAULT', 'show_overlay'),
    show_percentage=                conf.getboolean('DEFAULT', 'show_percentage'),
    crop_face_image=                conf.getboolean('DEFAULT', 'crop_face_image'),
    show_name=                      conf.getboolean('DEFAULT', 'show_name'),
    multiple_faces=                 conf.getboolean('DEFAULT', 'multiple_faces'),
    bottom_area=                    conf.getboolean('DEFAULT', 'bottom_area')
)

layout = [
    [sg.Image(filename='', key='display', pad=(0,0))],
    [sg.Button('終了', key='terminate', pad=(0,10))]
]

window = sg.Window(
    'CALL_FACE01GRAPHICS', layout, alpha_channel=1, margins=(0, 0), 
    no_titlebar=True, grab_anywhere=True,
    location=(350,130), modal=True
)

def multi(x):
    name, pict, date, img, location, percentage_and_symbol = x['name'], x['pict'], x['date'], x['img'], x['location'], x['percentage_and_symbol']
    if not name==None:
        print(
            name, "\n",
            "\t", "類似度\t", percentage_and_symbol, "\n", 
            "\t", "座標\t", location, "\n",
            "\t", "時刻\t", date, "\n",
            "\t", "出力\t", pict, "\n",
            "------------\n"
        )
    return img

pool = ThreadPoolExecutor()

for array_x in xs:
    for x in array_x:
        if x=={}:
            continue

        result = pool.submit(multi, x)
        event, _ = window.read(timeout=1)
        
        if  not result.result() is None:
            imgbytes=cv2.imencode(".png", result.result())[1].tobytes()
            window["display"].update(data=imgbytes)
    
        if event=='terminate':
            break
    else:
        continue
    break

window.close()
print('終了します')
