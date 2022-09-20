#!python
#cython: language_level=3
"""
cv2.imreadで読み込んだframe変数について。
> 3次元配列が出力されていることがわかる．この配列をA×B×3次元配列とする．
> Aは画素の行数であり，Bは画素の列数である
> (読み込む画像のサイズによって，行列のサイズは変わるため変数A，Bとした)．3は，RGBの輝度である．
> 上の画像において，輝度が縦に大量に並んでいるが，これは
> [[[0行0列目の輝度]~[0行B列目の輝度]]~[[A行0列目の輝度]~[A行B列目の輝度]]]の順に並んでいる．
> (画像において0行0列目は，左上)
> よって，imreadで返される配列とは，画素の輝度を行列の順に格納したものである

> imreadで返された配列の顔画像の部分(顔画像の左上の行列から，右下の行列までの**区分行列**（ブロック行列）の輝度)
> だけを取り出すことで，切り取ることができた．

[imreadで返される配列について](https://qiita.com/Castiel/items/53ecbee3c06b9d92759e)
"""

"""
[NumPyの軸(axis)と次元数(ndim)とは何を意味するのか](https://deepage.net/features/numpy-axis.html)
"""

from os import chdir, environ
from os.path import dirname
from traceback import format_exc
import inspect

import cv2
import requests
from requests.auth import HTTPDigestAuth

from io import BytesIO
from PIL import Image, ImageFile
import numpy as np
from sys import exit
from .logger import Logger
from .Calc import Cal
# from  import return_tuple
# from .sample import size

name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir, None)
Cal().cal_specify_date(logger)
ImageFile.LOAD_TRUNCATED_IMAGES = True
"""TODO #18 opencvの環境変数変更 要調査"""
# environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class VidCap:
    # def __init__(self) -> None:
    #     pass

    # デバッグ用imshow()
    def frame_imshow_for_debug(self, frame):
        self.frame_maybe = frame
        if isinstance(self.frame_maybe, np.ndarray):
            
            cv2.imshow("DEBUG", self.frame_maybe)
            logger.warning(inspect.currentframe().f_back.f_code.co_filename)
            logger.warning(inspect.currentframe().f_back.f_lineno)
            cv2.moveWindow('window DEBUG', 0, 0)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            for fra in self.frame_maybe:
                fr = fra["img"]
                cv2.imshow("DEBUG", fr)
                cv2.moveWindow('window DEBUG', 0, 0)
                logger.warning(inspect.currentframe().f_back.f_code.co_filename)
                logger.warning(inspect.currentframe().f_back.f_lineno)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

    # cpdef
    def resize_frame(self, set_width:int, set_height:int, frame:np.ndarray) -> np.ndarray:
        self.set_width:int = set_width
        self.set_height:int = set_height
        self.frame:np.ndarray = frame
        small_frame:np.ndarray = cv2.resize(self.frame, (self.set_width, self.set_height))
        return small_frame

    # not cdef
    def return_movie_property(self, set_width: int, vcap) -> tuple:
        self.set_width = set_width
        self.vcap = vcap
        
        fps: int    = self.vcap.get(cv2.CAP_PROP_FPS)
        height: int = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width: int  = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # BUGFIX: vcapを閉じておく
        self.finalize(self.vcap)

        fps = int(fps)
        height= int(height)
        width = int(width)
        if width <= 0:
            logger.warning("受信できません")
            logger.warning("-" * 20)
            logger.warning(format_exc(limit=None, chain=True))
            logger.warning("-" * 20)
            logger.warning("終了します")
            exit(0)
        set_height: int = int((self.set_width * height) / width)
        return self.set_width,fps,height,width,set_height

    # cdef, python版
    def cal_angle_coordinate(self, height:int, width:int) -> tuple:
        self.height:int = height
        self.width:int = width
        """画角(TOP_LEFT,TOP_RIGHT)予めを算出

        Args:
            height (int)
            width (int)

        Returns:
            tuple: TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT,CENTER
        """
        # order: (top, bottom, left, right)
        TOP_LEFT: tuple =(0,int(self.height/2),0,int(self.width/2))
        TOP_RIGHT: tuple =(0,int( self.height/2),int(self.width/2),self.width)
        BOTTOM_LEFT: tuple =(int(self.height/2),self.height,0,int(self.width/2))
        BOTTOM_RIGHT: tuple =(int(self.height/2),self.height,int(self.width/2),self.width)
        CENTER: tuple =(int(self.height/4),int(self.height/4)*3,int(self.width/4),int(self.width/4)*3)
        return TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT,CENTER

    # cdef, frameに対してエリア指定
    def angle_of_view_specification(self, set_area: str, frame: np.ndarray, TOP_LEFT, TOP_RIGHT: tuple, BOTTOM_LEFT: tuple, BOTTOM_RIGHT: tuple, CENTER: tuple) ->np.ndarray:
        self.set_area: str = set_area
        self.frame:np.ndarray = frame
        self.TOP_LEFT:tuple = TOP_LEFT
        self.TOP_RIGHT:tuple = TOP_RIGHT
        self.BOTTOM_LEFT:tuple = BOTTOM_LEFT
        self.BOTTOM_RIGHT:tuple = BOTTOM_RIGHT
        self.CENTER:tuple = CENTER
        # VidCap().frame_imshow_for_debug(self.frame)
        # face_location order: top, right, bottom, left
        # TOP_LEFT order: (top, bottom, left, right)
        # how to slice: img[top:bottom, left:right]
        if self.set_area=='NONE':
            pass
        elif self.set_area=='TOP_LEFT':
            self.frame = self.frame[self.TOP_LEFT[0]:self.TOP_LEFT[1],self.TOP_LEFT[2]:self.TOP_LEFT[3]]
        elif self.set_area=='TOP_RIGHT':
            self.frame = self.frame[self.TOP_RIGHT[0]:self.TOP_RIGHT[1],self.TOP_RIGHT[2]:self.TOP_RIGHT[3]]
        elif self.set_area=='BOTTOM_LEFT':
            self.frame = self.frame[self.BOTTOM_LEFT[0]:self.BOTTOM_LEFT[1],self.BOTTOM_LEFT[2]:self.BOTTOM_LEFT[3]]
        elif self.set_area=='BOTTOM_RIGHT':
            self.frame = self.frame[self.BOTTOM_RIGHT[0]:self.BOTTOM_RIGHT[1],self.BOTTOM_RIGHT[2]:self.BOTTOM_RIGHT[3]]
        elif self.set_area=='CENTER':
            self.frame = self.frame[self.CENTER[0]:self.CENTER[1],self.CENTER[2]:self.CENTER[3]]
        # VidCap().frame_imshow_for_debug(self.frame)
        return self.frame

    # not cdef
    def return_vcap(self, movie:str):
        self.movie:str = movie
        """vcapをreturnする

        Args:
            movie (str): movie

        Returns:
            object: vcap
        """
        # movie=movie
        if self.movie=='usb' or self.movie == 'USB':   # USB カメラ読み込み時使用
            live_camera_number:int = 0
            for camera_number in range(0, 5):
                vcap = cv2.VideoCapture(camera_number)
                # BUG: cv2.CAP_FFMPEG
                # vcap = cv2.VideoCapture(camera_number, cv2.CAP_FFMPEG)
                # vcap = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
                ret:bool
                frame:np.ndarray
                ret, frame = vcap.read()
                if ret:
                    """DEBUG
                    self.frame_imshow_for_debug(frame)
                    """
                if ret:
                    live_camera_number = camera_number
                    self.finalize(vcap)
                    break
                if camera_number == 4 and ret is False:
                    logger.warning("USBカメラの受信が出来ません")
                    logger.warning("以下のエラーをシステム管理者へお伝えください")
                    logger.warning("-" * 20)
                    logger.warning(format_exc(limit=None, chain=True))
                    logger.exception("USBカメラとの通信に異常が発生しました")
                    logger.warning("-" * 20)
                    logger.warning("終了します")
                    self.finalize(vcap)
                    exit(0)
            print(f'live_camera_number: {live_camera_number}')
            vcap = cv2.VideoCapture(live_camera_number)
            return vcap
        else:
            vcap = cv2.VideoCapture(self.movie)
            return vcap

    def finalize(self, vcap):
        self.vcap = vcap
        # 入力動画処理ハンドルのリリース
        self.vcap.release()
        # ウィンドウの除去
        cv2.destroyAllWindows()

    # @lru_cache(maxsize=None)
    def frame_generator(self, args_dict):
        self.args_dict = args_dict
        """初期値"""
        TOP_LEFT = 0
        TOP_RIGHT = 0
        BOTTOM_LEFT = 0
        BOTTOM_RIGHT = 0
        CENTER = 0
        resized_frame_list = []
        # frame_skip_counter: int = 0
        set_width = self.args_dict["set_width"]
        set_height = self.args_dict["set_height"]
        movie = self.args_dict["movie"]

        kaoninshoDir = self.args_dict["kaoninshoDir"] 
        chdir(kaoninshoDir)
        movie = self.args_dict["movie"] 
        set_area = self.args_dict["set_area"] 
        # 画角値（四隅の座標:Tuple）算出
        if  TOP_LEFT == 0:
            TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER = self.cal_angle_coordinate(self.args_dict["height"], self.args_dict["width"])
            # TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER = return_tuple(self.args_dict["height"], self.args_dict["width"])

        if (movie == 'usb' or movie == 'USB'):   # USB カメラ読み込み時使用
            camera_number:int = 0
            live_camera_number:int = 0
            for camera_number in range(0, 5):
                vcap = cv2.VideoCapture(camera_number)
                ret, frame = vcap.read()
                if ret:
                    live_camera_number = camera_number 
                    break
            logger.info(f'CAMERA DEVICE NUMBER: {camera_number}')
            while vcap.isOpened(): 
                # frame_skipの数値に満たない場合は処理をスキップ
                for frame_skip_counter in range(1, self.args_dict["frame_skip"]):
                    ret, frame = vcap.read()
                    if frame_skip_counter < self.args_dict["frame_skip"]:
                        continue
                    if ret == False:
                        logger.warning("ERROR OCURRED\nREPORTED BY FACE01")
                        logger.warning("-" * 20)
                        logger.warning(format_exc(limit=None, chain=True))
                        logger.warning("-" * 20)
                        self.finalize(vcap)
                    break
                else:
                    # 画角値をもとに各frameを縮小
                    # python版
                    frame = self.angle_of_view_specification(set_area, frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER)  # type: ignore
                    # 各frameリサイズ
                    resized_frame = self.resize_frame(set_width, set_height, frame)
                    """DEBUG
                    self.frame_imshow_for_debug(resized_frame)
                    """
                    yield resized_frame


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
                    # frame_skipの数値に満たない場合は処理をスキップ
                    for frame_skip_counter in range(1, self.args_dict["frame_skip"]):
                        response = requests.get(url, auth = HTTPDigestAuth(self.args_dict["user"], self.args_dict["passwd"]))
                        # print(f'response: {response}')
                        if frame_skip_counter < self.args_dict["frame_skip"]:
                            continue
                    # {'Status': '200', 'Connection': 'Close', 'Set-Cookie': 'Session=0', 'Accept-Ranges': 'bytes',
                    #  'Cache-Control': 'no-cache', 'Content-length': '40140', 'Content-type': 'image/jpeg'}
                    # if response.headers['Status'] == '200' and response.headers['Content-type'] == 'image/jpeg':
                    if response.headers['Content-type'] == 'image/jpeg':
                        # 取得した画像データをOpenCVで扱う形式に変換
                        img_bin = BytesIO(response.content)
                        img_pil = Image.open(img_bin)
                        img_np  = np.asarray(img_pil)
                        frame  = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                        """DEBUG
                        cv2.imshow("video_capture_DEBUG", frame)
                        cv2.moveWindow("video_capture_DEBUG", 0,0)
                        cv2.waitKey(5000)
                        cv2.destroyAllWindows()
                        exit(0)
                        """
                        # 画角値をもとに各frameを縮小
                        # python版
                        frame = self.angle_of_view_specification(set_area, frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER)
                        # 各frameリサイズ
                        resized_frame = self.resize_frame(set_width, set_height, frame)
                        """DEBUG
                        cv2.imshow("video_capture_DEBUG", frame)
                        cv2.moveWindow("video_capture_DEBUG", 0,0)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                        """
                        try:
                            yield resized_frame
                        except TypeError as e:
                            logger.warning(e)
                        except Exception as e:
                            logger.warning(e)
                        finally:
                            yield resized_frame
                    else:
                        logger.warning("以下のエラーをシステム管理者へお伝えください")
                        logger.warning(f"ステータスコード: {response.headers['Status']}")
                        logger.warning(f"コンテントタイプ: {response.headers['Content-type']}")
                        logger.warning("-" * 20)
                        logger.warning(format_exc(limit=None, chain=True))
                        logger.warning("-" * 20)
                        exit(0)
            except:
                logger.warning("以下のエラーをシステム管理者へお伝えください")
                logger.warning("-" * 20)
                logger.warning(format_exc(limit=None, chain=True))
                logger.exception("通信に異常が発生しました")
                logger.warning("-" * 20)
                logger.warning("終了します")
                exit(0)
            
        # elif 'rtsp' in movie:
        #     """RTSPの場合は通常のテスト動画と同じ"""
        else:
            vcap = cv2.VideoCapture(movie, cv2.CAP_FFMPEG)
            while vcap.isOpened(): 
                # frame_skipの数値に満たない場合は処理をスキップ
                for frame_skip_counter in range(1, self.args_dict["frame_skip"]+1):
                    ret, frame = vcap.read()
                    if frame_skip_counter < self.args_dict["frame_skip"]:
                        continue
                    if ret == False:
                        logger.warning("ERROR OCURRED")
                        logger.warning("DATA RECEPTION HAS ENDED")
                        logger.warning("-" * 20)
                        logger.warning(format_exc(limit=None, chain=True))
                        logger.warning("-" * 20)
                        self.finalize(vcap)
                        break
                # else:
                    if ret == True:
                        """C++実装試験
                        # frame = frame.astype(dtype='float64')
                        size(frame.shape, frame.strides, set_area, frame, TOP_LEFT)
                        """
                        # 画角値をもとに各frameを縮小
                        # python版
                        frame = self.angle_of_view_specification(set_area, frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER)  # type: ignore
                        # 各frameリサイズ
                        resized_frame = self.resize_frame(set_width, set_height, frame)
                        """DEBUG
                        cv2.imshow("video_capture_DEBUG", frame)
                        cv2.moveWindow("video_capture_DEBUG", 0,0)
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()
                        """
                        yield resized_frame
                    elif ret == False:
                        self.finalize(vcap)

                        # resized_frame_list.append(resized_frame)
                        # if len(resized_frame_list) == 5:
                        #     resized_frame_list_copy = resized_frame_list
                        #     resized_frame_list = []
                        #     yield resized_frame_list_copy

