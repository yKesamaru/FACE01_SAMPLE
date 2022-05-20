import os
import sys
from functools import lru_cache

import cv2
import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import POPUP_BUTTONS_OK

@lru_cache(maxsize=None)
def video_capture(kaoninshoDir, movie):
    os.chdir(kaoninshoDir)
    movie=movie
    # 入力映像設定 ===============================================
    if movie=='usb':
        for camera_number in range(-5, 5):  # -1: 自動
            vcap = cv2.VideoCapture(camera_number) # USB カメラ読み込み時使用
            if vcap.isOpened(): 
                print(f'カメラデバイス番号：{camera_number}')
                print('Webカメラ：バッファリング中…')
                break
    else:
        print(movie, '：バッファリング中…')
        vcap = cv2.VideoCapture(movie, cv2.CAP_FFMPEG)
    
    if not vcap.isOpened():
        sg.popup( '不正な映像データのため終了します', 'FACE01 GRAPHICSを再起動して下さい', movie, title='警告', button_type=POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
        sys.exit(0)
    ret, frame = vcap.read()
    if ret == False:
        sg.popup( '映像データが不正に途切れたので終了します', 'FACE01 GRAPHICSを再起動して下さい', movie, title='警告', button_type=POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
        sys.exit(0)
    return vcap

# 検証用
if __name__ == '__main__':
    # kaoninshoDir='/home/terms/ビデオ/one_drive/FACE01GRAPHICS123_UBUNTU_VENV/'
    # movie=kaoninshoDir + 'some_people.mp4'
    movie='usb'
    
    vcap = video_capture(kaoninshoDir, movie)

    layout = [
        [sg.Image('', key='img')]
    ]

    window = sg.Window('', layout)

    while True:
        event, _ = window.read(timeout=1)

        if event == None:
            break
        
        ret, frame = vcap.read()

        frame=cv2.resize(frame, (500,300))

        if ret:
            imbytes=cv2.imencode('.png', frame)[1].tobytes()
            window['img'].update(data=imbytes)

    window.close()

        # opencv-pythonのcv2.imshow()が機能しない_2021年9月11日
        # cv2.imshow('video_capture', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     vcap.release()
        #     cv2.destroyAllWindows()