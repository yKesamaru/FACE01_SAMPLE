import logging
from os import chdir, environ
from traceback import format_exc
# from functools import lru_cache

from cv2 import resize, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture, CAP_FFMPEG, destroyAllWindows, imshow, waitKey, destroyAllWindows
from PySimpleGUI import popup, POPUP_BUTTONS_OK
import requests
from requests.auth import HTTPDigestAuth

import io
from PIL import Image
import numpy as np
import cv2
from sys import exit



"""TODO opencvの環境変数変更 要調査"""
# environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



"""Logging"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')

file_handler = logging.FileHandler('face01.log', mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



"""TODO
RTSPを受け付けるようにする
マルチスレッド化
イテレーターオブジェクトをマルチスレッドでyieldすることにより
frame送出単位でマルチスレッド化する
see README.md
"""

environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
"""see bellow

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
    small_frame = resize(frame, (set_width, set_height))
    return small_frame

def return_movie_property(set_width: int, vcap) -> tuple:
    set_width = set_width
    fps: int    = vcap.get(CAP_PROP_FPS)
    height: int = vcap.get(CAP_PROP_FRAME_HEIGHT)
    width: int  = vcap.get(CAP_PROP_FRAME_WIDTH)
    fps = int(fps)
    height= int(height)
    width = int(width)
    if width == 0:
        logger.warning("受信できません")
        logger.warning("-" * 20)
        logger.warning(format_exc(limit=None, chain=True))
        logger.warning("-" * 20)
        logger.warning("終了します")
        exit(0)
    # widthが400pxより小さい場合警告を出す
    if width < 400:
        popup('入力指定された映像データ幅が小さすぎます','width: {}px'.format(width), 'このまま処理を続けますが', '性能が発揮できない場合があります','OKを押すと続行します', title='警告', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
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
    # movie=movie
    if movie=='usb' or movie == 'USB':   # USB カメラ読み込み時使用
        live_camera_number:int = 0
        for camera_number in range(-5, 5):
            vcap = VideoCapture(camera_number)
            ret, frame = vcap.read()
            if ret:
                live_camera_number = camera_number 
        vcap = VideoCapture(live_camera_number)
        return vcap
    else:
        vcap = VideoCapture(movie, CAP_FFMPEG)
        return vcap

def finalize(vcap):
    # 入力動画処理ハンドルのリリース
    vcap.release()
    # ウィンドウの除去
    destroyAllWindows()

# @lru_cache(maxsize=None)
def frame_generator(args_dict):
    """初期値"""
    TOP_LEFT = 0
    TOP_RIGHT = 0
    BOTTOM_LEFT = 0
    BOTTOM_RIGHT = 0
    CENTER = 0
    resized_frame_list = []
    frame_skip_counter: int = 0
    set_width = args_dict["set_width"]
    set_height = args_dict["set_height"]

    kaoninshoDir = args_dict["kaoninshoDir"] 
    chdir(kaoninshoDir)
    movie = args_dict["movie"] 
    set_area = args_dict["set_area"] 
    # 画角値（四隅の座標:Tuple）算出
    if  TOP_LEFT == 0:
        TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER = cal_angle_coordinate(args_dict["height"], args_dict["width"])

    if (movie == 'usb' or movie == 'USB'):   # USB カメラ読み込み時使用
        camera_number:int = 0
        live_camera_number:int = 0
        for camera_number in range(-5, 5):
            vcap = VideoCapture(camera_number)
            ret, frame = vcap.read()
            if ret:
                live_camera_number = camera_number 
                break
        vcap = VideoCapture(live_camera_number)
        logger.info(f'カメラデバイス番号：{camera_number}')
    elif 'http' in movie:
        """DEBUG"""
        # print(movie); exit(0)
        """see bellow
        [Panasonic製ネットワークカメラの画像を取得して顔検出をしてみる](https://qiita.com/mix_dvd/items/a0bdbe0ba628d5282639)
        [Python, Requestsの使い方](https://note.nkmk.me/python-requests-usage/)
        """
        url = movie
        # 画像の取得
        try:
            # responseの内容について分岐
            while True:
                response = requests.get(url, auth=HTTPDigestAuth(args_dict["user"], args_dict["passwd"]))
                # {'Status': '200', 'Connection': 'Close', 'Set-Cookie': 'Session=0', 'Accept-Ranges': 'bytes',
                #  'Cache-Control': 'no-cache', 'Content-length': '40140', 'Content-type': 'image/jpeg'}
                # if response.headers['Status'] == '200' and response.headers['Content-type'] == 'image/jpeg':
                if response.headers['Content-type'] == 'image/jpeg':
                    # 取得した画像データをOpenCVで扱う形式に変換
                    img_bin = io.BytesIO(response.content)
                    img_pil = Image.open(img_bin)
                    img_np  = np.asarray(img_pil)
                    frame  = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    """DEBUG
                    imshow("video_capture_DEBUG", frame)
                    cv2.moveWindow("video_capture_DEBUG", 0,0)
                    cv2.waitKey(5000)
                    cv2.destroyAllWindows()
                    exit(0)
                    """
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
                    imshow("video_capture_DEBUG", frame)
                    cv2.moveWindow("video_capture_DEBUG", 0,0)
                    waitKey(500)
                    destroyAllWindows()
                    """
                    # yield resized_frame
                    # """DEBUG
                    try:
                        yield resized_frame
                    except TypeError as e:
                        logger.warning(e)
                    except Exception as e:
                        logger.warning(e)
                    finally:
                        yield resized_frame
                    # """
                # else:
                #     logger.warning(f"ステータスコード: {response.headers['Status']}")
                #     logger.warning(f"コンテントタイプ: {response.headers['Content-type']}")
                #     logger.warning("以下のエラーをシステム管理者へお伝えください")
                #     logger.warning("-" * 20)
                #     logger.warning(format_exc(limit=None, chain=True))
                #     logger.warning("-" * 20)
                #     logger.warning("終了します")
                #     break
        except:
            logger.warning("urlが開けませんでした")
            logger.warning("以下のエラーをシステム管理者へお伝えください")
            logger.warning("-" * 20)
            logger.warning(format_exc(limit=None, chain=True))
            logger.exception("httプロトコルに以上が発生しました")
            logger.warning("-" * 20)
            logger.warning("終了します")
            exit(0)
        
    # elif 'rtsp' in movie:
    #     """RTSPの場合は通常のテスト動画と同じ"""
    else:
        vcap = VideoCapture(movie, CAP_FFMPEG)
        while vcap.isOpened(): 
            ret, frame = vcap.read()
            if ret == False:
                # popup( '不正な映像データのため終了します', 'システム管理者にお問い合わせください', movie, title='ERROR', button_type=sg.POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
                logger.warning("以下のエラーをシステム管理者へお伝えください")
                logger.warning("-" * 20)
                logger.warning(format_exc(limit=None, chain=True))
                logger.warning("-" * 20)
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
                imshow("video_capture_DEBUG", frame)
                cv2.moveWindow("video_capture_DEBUG", 0,0)
                waitKey(3000)
                destroyAllWindows()
                """
                yield resized_frame

                # resized_frame_list.append(resized_frame)
                # if len(resized_frame_list) == 5:
                #     resized_frame_list_copy = resized_frame_list
                #     resized_frame_list = []
                #     yield resized_frame_list_copy

