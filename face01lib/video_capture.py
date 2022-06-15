import logging
import os
import sys
import traceback
from functools import lru_cache

import cv2
import PySimpleGUI as sg

logger = logging.getLogger('face01lib/video_capture')

"""TODO
RTSPを受け付けるようにする
マルチスレッド化
イテレーターオブジェクトをマルチスレッドでyieldすることにより
frame送出単位でマルチスレッド化する
see README.md
"""

""" BUG & TODO frame_skip変数 半自動設定
if len(face_encodings) > 0:
    # frame_skip変数 半自動設定 --------------
    def decide_frame_skip(frame_skip, frame_skip_counter) -> Tuple[int,int]:
        ## マシンスペックが十分な場合、frame_skip = (顔の数)
        if frame_skip == -1:
            # 人数 - 1
            frame_skip = len(face_encodings)
            # 人数が1人の時はframe_skip = 1
            if len(face_encodings)==1:
                frame_skip = 2
        else:
            frame_skip = frame_skip
        if frame_skip_counter < frame_skip:
            frame_skip_counter += 1
        return frame_skip, frame_skip_counter

    frame_skip, frame_skip_counter = decide_frame_skip(frame_skip, frame_skip_counter)
    if frame_skip_counter < frame_skip:
        continue
    # ----------------------------------------
    # fps_ms = fps
    # if frame_skip > 0:
    #     HANDLING_FRAME_TIME / (frame_skip - 1)  < fps_ms
    # elif frame_skip == 0:
    #     HANDLING_FRAME_TIME < fps_ms
    # time.sleep((fps_ms - (HANDLING_FRAME_TIME / (frame_skip - 1))) / 1000)
    """

def resize_frame(set_width, set_height, frame):
    small_frame: cv2.Mat = cv2.resize(frame, (set_width, set_height))
    return small_frame

def return_movie_property(set_width: int, vcap) -> tuple:
    set_width = set_width
    fps: int    = vcap.get(cv2.CAP_PROP_FPS)
    height: int = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width: int  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(fps)
    height= int(height)
    width = int(width)
    # widthが400pxより小さい場合警告を出す
    if width < 400:
        sg.popup('入力指定された映像データ幅が小さすぎます','width: {}px'.format(width), 'このまま処理を続けますが', '性能が発揮できない場合があります','OKを押すと続行します', title='警告', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
    # しかしながらパイプ処理の際フレームサイズがわからなくなるので決め打ちする
    set_height: int = int((set_width * height) / width)
    return set_width,fps,height,width,set_height

# python版
def cal_angle_coordinate(height:int, width:int) -> tuple:
    """画角(TOP_LEFT,TOP_RIGHT)予めを算出

    Args:
        height (int)
        width (int)

    Returns:
        tuple: TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT,CENTER
    """
    TOP_LEFT: tuple =(0,int(height/2),0,int(width/2))
    TOP_RIGHT: tuple =(0,int( height/2),int(width/2),width)
    BOTTOM_LEFT: tuple =(int(height/2),height,0,int(width/2))
    BOTTOM_RIGHT: tuple =(int(height/2),height,int(width/2),width)
    CENTER: tuple =(int(height/4),int(height/4)*3,int(width/4),int(width/4)*3)
    return TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT,CENTER

# frameに対してエリア指定
def angle_of_view_specification(set_area, frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER):
    if set_area=='NONE':
        pass
    elif set_area=='TOP_LEFT':
        frame = frame[TOP_LEFT[0]:TOP_LEFT[1],TOP_LEFT[2]:TOP_LEFT[3]]
    elif set_area=='TOP_RIGHT':
        frame = frame[TOP_RIGHT[0]:TOP_RIGHT[1],TOP_RIGHT[2]:TOP_RIGHT[3]]
    elif set_area=='BOTTOM_LEFT':
        frame = frame[BOTTOM_LEFT[0]:BOTTOM_LEFT[1],BOTTOM_LEFT[2]:BOTTOM_LEFT[3]]
    elif set_area=='BOTTOM_RIGHT':
        frame = frame[BOTTOM_RIGHT[0]:BOTTOM_RIGHT[1],BOTTOM_RIGHT[2]:BOTTOM_RIGHT[3]]
    elif set_area=='CENTER':
        frame = frame[CENTER[0]:CENTER[1],CENTER[2]:CENTER[3]]
    return frame

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

def finalize(vcap):
    # 入力動画処理ハンドルのリリース
    vcap.release()
    # ウィンドウの除去
    cv2.destroyAllWindows()

# @lru_cache(maxsize=None)
def frame_generator(args_dict):
    """初期値"""
    TOP_LEFT = 0
    TOP_RIGHT = 0
    BOTTOM_LEFT = 0
    BOTTOM_RIGHT = 0
    CENTER = 0

    kaoninshoDir = args_dict["kaoninshoDir"] 
    os.chdir(kaoninshoDir)
    movie = args_dict["movie"] 
    set_area = args_dict["set_area"] 
    # 画角値（四隅の座標:Tuple）算出
    if  TOP_LEFT == 0:
        TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER = cal_angle_coordinate(args_dict["height"], args_dict["width"])

    if movie=='usb':   # USB カメラ読み込み時使用
        camera_number:int = 0
        live_camera_number:int = 0
        for camera_number in range(-5, 5):
            vcap = cv2.VideoCapture(camera_number)
            ret, frame = vcap.read()
            if ret:
                live_camera_number = camera_number 
        vcap = cv2.VideoCapture(live_camera_number)
        logger.info(f'カメラデバイス番号：{camera_number}')
    else:
        vcap = cv2.VideoCapture(movie, cv2.CAP_FFMPEG)
    
        resized_frame_list = []
        frame_skip_counter: int = 0
        set_width = args_dict["set_width"]
        set_height = args_dict["set_height"]
        while vcap.isOpened(): 
            ret, frame = vcap.read()
            if ret == False:
                # sg.popup( '不正な映像データのため終了します', 'システム管理者にお問い合わせください', movie, title='ERROR', button_type=sg.POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
                logger.warning("以下のエラーをシステム管理者へお伝えください")
                logger.warning("---------------------------------------------")
                logger.warning(traceback.format_exc(limit=None, chain=True))
                logger.warning("---------------------------------------------")
                finalize(args_dict["vcap"])
                break
            else:
                # frame_skipの数値に満たない場合は処理をスキップ
                if frame_skip_counter < args_dict["frame_skip"]:
                    frame_skip_counter += 1
                    continue

                # 画角値をもとに各frameを縮小
                # python版
                frame = angle_of_view_specification(set_area, frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER)
                # 各frameリサイズ
                resized_frame = resize_frame(set_width, set_height, frame)
                """DEBUG
                cv2.imshow("video_capture_DEBUG", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                """

                yield resized_frame

                # resized_frame_list.append(resized_frame)
                # if len(resized_frame_list) == 5:
                #     resized_frame_list_copy = resized_frame_list
                #     resized_frame_list = []
                #     yield resized_frame_list_copy

