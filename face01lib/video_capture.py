import os
import sys
from functools import lru_cache

import cv2
import PySimpleGUI as sg

"""マルチスレッド化
イテレーターオブジェクトをマルチスレッドでyieldすることにより
frame送出単位でマルチスレッド化する
see README.md
"""

def return_vcap(movie):
    """vcapをreturnする

    Args:
        movie (str): movie

    Returns:
        object: vcap
    """
    movie=movie
    if movie=='usb':   # USB カメラ読み込み時使用
        live_camera_number:int = 0
        for camera_number in range(-5, 5):
            vcap = cv2.VideoCapture(camera_number)
            ret, frame = vcap.read()
            if ret:
                live_camera_number = camera_number 
        vcap = cv2.VideoCapture(live_camera_number)
        return vcap
    else:
        vcap = cv2.VideoCapture(movie, cv2.CAP_FFMPEG)
        return vcap


@lru_cache(maxsize=None)
def video_capture(kaoninshoDir, movie):
    """video_caputure

    Args:
        kaoninshoDir (str): directory
        movie (str): file name or camera

    Yields:
        Iterator[np.ndarray]: frame
    """
    os.chdir(kaoninshoDir)
    movie=movie
    if movie=='usb':   # USB カメラ読み込み時使用
        camera_number:int = 0
        live_camera_number:int = 0
        for camera_number in range(-5, 5):
            vcap = cv2.VideoCapture(camera_number)
            ret, frame = vcap.read()
            if ret:
                live_camera_number = camera_number 
        vcap = cv2.VideoCapture(live_camera_number)
        print(f'カメラデバイス番号：{camera_number}')
        print('Webカメラ：バッファリング中…')
        while vcap.isOpened(): 
            ret, frame = vcap.read()
            if ret == False:
                sg.popup( '不正な映像データのため終了します', 'システム管理者にお問い合わせください', movie, title='ERROR', button_type=sg.POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
                break
            frame_generator_obj = frame
            yield frame_generator_obj
    else:
        print(movie, '：バッファリング中…')
        vcap = cv2.VideoCapture(movie, cv2.CAP_FFMPEG)
        while vcap.isOpened(): 
            ret, frame = vcap.read()
            """DEBUG
            cv2.imshow("DEBUG", frame)
            cv2.imshow("DEBUG", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            """
            if ret == False:
                sg.popup( '不正な映像データのため終了します', 'システム管理者にお問い合わせください', movie, title='ERROR', button_type=sg.POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
                break
            frame_generator_obj = frame
            yield frame_generator_obj

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
