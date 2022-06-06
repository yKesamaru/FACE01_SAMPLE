"""CHECK SYSTEM INFORMATION"""
from GPUtil import getGPUs
import PySimpleGUI as sg
import face01lib.api as faceapi
import platform
import psutil
import sys

sg.theme('LightGray')

def system_check():
    """TODO
    解決できるURLを指定すること
    標準出力にも同様の文を出力すること
    テキストファイルを生成して同じ処理を繰り返させないこと
    """
    sg.popup(
        'FACE01の推奨動作環境を満たしているかシステムチェックを実行します', 
        'Python 3.8以上をお使いください', 
        '現在のバージョン',
        sys.version,
        title='INFORMATION', button_type =sg. POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
    # CPU
    if psutil.cpu_freq().max < 3000 or psutil.cpu_count(logical=False) < 4:
        sg.popup(
            'CPU性能が足りません',
            '3GHz以上のCPUが必要です',
            '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    # MEMORY
    if psutil.virtual_memory().total < 8000000000:
        sg.popup(
        'メモリーが足りません',
        '少なくとも8GB以上が必要です',
        '終了します', 
        title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    # GPU
    if faceapi.dlib.cuda.get_num_devices() == 0:
        sg.popup(
        'CUDAが有効なデバイスが見つかりません',
        '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    if faceapi.dlib.DLIB_USE_CUDA == False:
        sg.popup('dlibビルド時にCUDAが有効化されていません',
        '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    if faceapi.dlib.DLIB_USE_BLAS == False or faceapi.dlib.DLIB_USE_LAPACK == False:
        sg.popup(
            'BLASまたはLAPACKのいずれか、あるいは両方がインストールされていません',
            'パッケージマネージャーでインストールしてください',
            'CUBLAS native runtime libraries(Basic Linear Algebra Subroutines: 基本線形代数サブルーチン)',
            'LAPACK バージョン 3.X(線形代数演算を行う総合的な FORTRAN ライブラリ)',
            'インストール後にdlibを改めて再インストールしてください',
            '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    for gpu in getGPUs():
        gpu_memory = gpu.memoryTotal
        gpu_name = gpu.name
    if gpu_memory < 3000:
        sg.popup(
            'GPUデバイスの性能が足りません',
            '現在のGPUデバイス',
            [gpu_name],
            'NVIDIA GeForce GTX 1660 Ti以上をお使いください',
            '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        # exit()
# system_check()

import configparser
import datetime
import os
from pickletools import uint8
import shutil
import time
from collections import defaultdict
from functools import lru_cache
from pickle import NONE
from typing import Dict, List, Tuple
import traceback
import cv2
import mediapipe as mp
import numpy as np
from numba import f8, i8, njit
from PIL import Image, ImageDraw, ImageFont

from face01lib.load_priset_image import load_priset_image
from face01lib.similar_percentage_to_tolerance import to_tolerance
from face01lib.video_capture import video_capture, return_vcap

"""mediapipe for python, see bellow
https://github.com/google/mediapipe/tree/master/mediapipe/python
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
faceapi: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
"""


# opencvの環境変数変更
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# configファイル読み込み
def configure():
    try:
        conf = configparser.ConfigParser()
        conf.read('config.ini', 'utf-8')
        # dict作成
        conf_dict = {
            'model' : conf.get('DEFAULT','model'),
            'similar_percentage' : float(conf.get('DEFAULT','similar_percentage')),
            'jitters' : int(conf.get('DEFAULT','jitters')),
            'priset_face_images_jitters' : int(conf.get('DEFAULT','priset_face_images_jitters')),
            'upsampling' : int(conf.get('DEFAULT','upsampling')),
            'mode' : conf.get('DEFAULT','mode'),
            'frame_skip' : int(conf.get('DEFAULT','frame_skip')),
            'movie' : conf.get('DEFAULT','movie'),
            'rectangle' : conf.getboolean('DEFAULT','rectangle'),
            'target_rectangle' : conf.getboolean('DEFAULT','target_rectangle'),
            'show_video' : conf.getboolean('DEFAULT','show_video'),
            'frequency_crop_image' : int(conf.get('DEFAULT','frequency_crop_image')),
            'set_area' : conf.get('DEFAULT','set_area'),
            'print_property' : conf.getboolean('DEFAULT','print_property'),
            'calculate_time' : conf.getboolean('DEFAULT','calculate_time'),
            'SET_WIDTH' : int(conf.get('DEFAULT','SET_WIDTH')),
            'default_face_image_draw' : conf.getboolean('DEFAULT', 'default_face_image_draw'),
            'show_overlay' : conf.getboolean('DEFAULT', 'show_overlay'),
            'show_percentage' : conf.getboolean('DEFAULT', 'show_percentage'),
            'crop_face_image' : conf.getboolean('DEFAULT', 'crop_face_image'),
            'show_name' : conf.getboolean('DEFAULT', 'show_name'),
            'should_not_be_multiple_faces' : conf.getboolean('DEFAULT', 'should_not_be_multiple_faces'),
            'bottom_area' : conf.getboolean('DEFAULT', 'bottom_area'),
            'draw_telop_and_logo' : conf.getboolean('DEFAULT', 'draw_telop_and_logo'),
            'use_mediapipe' : conf.getboolean('DEFAULT','use_mediapipe'),
            'model_selection' : int(conf.get('DEFAULT','model_selection')),
            'min_detection_confidence' : float(conf.get('DEFAULT','min_detection_confidence')),
            'person_frame_face_encoding' : conf.getboolean('DEFAULT','person_frame_face_encoding'),
        }
        return conf_dict
    except:
        print("config.ini 読み込み中にエラーが発生しました")
        print("以下のエラーをシステム管理者へお伝えください")
        print("---------------------------------------------")
        print(traceback.format_exc(limit=None, chain=True))
        print("---------------------------------------------")
        quit()

def cal_specify_date() -> None:
    """指定日付計算
    """
    limit_date = datetime.datetime(2022, 12, 1, 0, 0, 0)   # 指定日付
    today = datetime.datetime.now()

    def limit_date_alart() -> None:
        if today >= limit_date:
            print('指定日付を過ぎました')
            sg.popup('サンプルアプリケーションをお使いいただきありがとうございます','使用可能期限を過ぎました', '引き続きご利用になる場合は下記までご連絡下さい', '東海顔認証　担当：袈裟丸','y.kesamaru@tokai-kaoninsho.com', '', 'アプリケーションを終了します', title='', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
            exit()
        elif today < limit_date:
            remaining_days = limit_date - today
            if remaining_days.days < 30:
                dialog_text = 'お使い頂ける残日数は' + str(remaining_days.days) + '日です'
                sg.popup('サンプルアプリケーションをお使いいただきありがとうございます', dialog_text, title='', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
    limit_date_alart()
cal_specify_date()

# ホームディレクトリ固定
def home() -> tuple:
    kaoninshoDir: str = os.path.dirname(__file__)
    priset_face_imagesDir: str = f'{os.path.dirname(__file__)}/priset_face_images/'
    return kaoninshoDir, priset_face_imagesDir

# Initialize variables, load images
def initialize(conf_dict, headless):
    kaoninshoDir, priset_face_imagesDir = home()
    os.chdir(kaoninshoDir)

    known_face_encodings, known_face_names = load_priset_image(kaoninshoDir,priset_face_imagesDir)

    # Get frame info (fps, height etc)
    vcap = return_vcap(conf_dict["movie"])
    if not 'height' in locals():
        SET_WIDTH, fps, height, width, SET_HEIGHT = return_movie_property(conf_dict["SET_WIDTH"], vcap)

    # 画像読み込み系
    if headless == False:
        load_telop_image: bool = False
        load_logo_image: bool = False
        load_unregistered_face_image: bool = False

        rect01_png:cv2.Mat = cv2.imread("images/rect01.png", cv2.IMREAD_UNCHANGED)

        # Load Telop image
        telop_image: cv2.Mat
        if not load_telop_image:
            telop_image = cv2.imread("images/telop.png", cv2.IMREAD_UNCHANGED)
            load_telop_image = True
            _, orgWidth = telop_image.shape[:2]
            ratio: float = SET_WIDTH / orgWidth / 1.5  ## テロップ幅は横幅の半分に設定
            resized_telop_image = cv2.resize(telop_image, None, fx = ratio, fy = ratio)  # type: ignore
            cal_resized_telop_nums = cal_resized_telop_image(resized_telop_image)

        # Load Logo image
        logo_image: cv2.Mat
        if not load_logo_image:
            logo_image: cv2.Mat = cv2.imread("images/Logo.png", cv2.IMREAD_UNCHANGED)
            load_logo_image = True
            _, logoWidth = logo_image.shape[:2]
            logoRatio = SET_WIDTH / logoWidth / 10
            resized_logo_image = cv2.resize(logo_image, None, fx = logoRatio, fy = logoRatio)
            cal_resized_logo_nums = cal_resized_logo_image(resized_logo_image,  SET_HEIGHT,SET_WIDTH)

        # Load unregistered_face_image
        unregistered_face_image: cv2.Mat
        if not load_unregistered_face_image:
            unregistered_face_image = np.array(Image.open('./images/顔画像未登録.png'))
            unregistered_face_image = cv2.cvtColor(unregistered_face_image, cv2.COLOR_BGR2RGBA)
            load_unregistered_face_image = True

    # 日付時刻算出
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒

    # 辞書作成
    if headless == False:
        init_dict = {
            'known_face_encodings': known_face_encodings,
            'known_face_names': known_face_names,
            'date': date,
            'rect01_png': rect01_png,
            'telop_image': telop_image,
            'resized_telop_image': resized_telop_image,
            'cal_resized_telop_nums': cal_resized_telop_nums,
            'logo_image': logo_image,
            'cal_resized_logo_nums': cal_resized_logo_nums,
            'unregistered_face_image': unregistered_face_image,
            'SET_WIDTH': SET_WIDTH,
            'fps': fps,
            'height': height,
            'width': width,
            'SET_HEIGHT': SET_HEIGHT,
            'kaoninshoDir': kaoninshoDir,
            'priset_face_imagesDir': priset_face_imagesDir,
            'headless': False
        }
    elif headless == True:
        init_dict = {
            'known_face_encodings': known_face_encodings,
            'known_face_names': known_face_names,
            'date': date,
            'SET_WIDTH': SET_WIDTH,
            'fps': fps,
            'height': height,
            'width': width,
            'SET_HEIGHT': SET_HEIGHT,
            'kaoninshoDir': kaoninshoDir,
            'priset_face_imagesDir': priset_face_imagesDir,
            'headless': True
        }

    # 辞書結合
    args_dict = {**init_dict, **conf_dict}

    # ヘッドレス実装
    if headless == True:
        args_dict['rectangle'] = False
        args_dict['target_rectangle'] = False
        args_dict['show_video'] = False
        args_dict['default_face_image_draw'] = False
        args_dict['show_overlay'] = False
        args_dict['show_percentage'] = False
        args_dict['show_name'] = False
        args_dict['should_not_be_multiple_faces'] = False
        args_dict['bottom_area'] = False
        args_dict['draw_telop_and_logo'] = False
        args_dict['person_frame_face_encoding'] = False
        args_dict['headless'] = True

    return args_dict

def cal_resized_telop_image(resized_telop_image):
    x1, y1, x2, y2 = 0, 0, resized_telop_image.shape[1], resized_telop_image.shape[0]
    a = (1 - resized_telop_image[:,:,3:] / 255)
    b = resized_telop_image[:,:,:3] * (resized_telop_image[:,:,3:] / 255)
    cal_resized_telop_nums = (x1, y1, x2, y2, a, b)
    return cal_resized_telop_nums

def cal_resized_logo_image(resized_logo_image,  SET_HEIGHT,SET_WIDTH):
    x1, y1, x2, y2 = SET_WIDTH - resized_logo_image.shape[1], SET_HEIGHT - resized_logo_image.shape[0], SET_WIDTH, SET_HEIGHT
    a = (1 - resized_logo_image[:,:,3:] / 255)
    b = resized_logo_image[:,:,:3] * (resized_logo_image[:,:,3:] / 255)
    cal_resized_logo_nums = (x1, y1, x2, y2, a, b)
    return cal_resized_logo_nums

@lru_cache()
def return_fontpath():
    # フォントの設定(フォントファイルのパスと文字の大きさ)
    operating_system: str  = platform.system()
    fontpath: str = ''
    if (operating_system == 'Linux'):
        fontpath = "/usr/share/fonts/truetype/mplus/mplus-1mn-bold.ttf"
    elif (operating_system == 'Windows'):
                    # fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICR.TTC"
        fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICB.TTC"  ## bold体
    else:
        print('オペレーティングシステムの確認が出来ません。システム管理者にご連絡ください')
    return fontpath

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

def resize_frame(SET_WIDTH, SET_HEIGHT, frame):
    small_frame: cv2.Mat = cv2.resize(frame, (SET_WIDTH, SET_HEIGHT))
    return small_frame

def draw_telop(cal_resized_telop_nums, SET_WIDTH: int, resized_telop_image: np.ndarray, frame: np.ndarray):
    x1, y1, x2, y2, a, b = cal_resized_telop_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        """TODO
        logging warning level"""
        print("telopが描画できません")

    """DEBUG
    src1 = frame[y1:y2, x1:x2,]
    # imshowで可視化実験
    # cv2.imshow("DEBUG", frame[y1:y2, x1:x2] * (1 - resized_telop_image[:,:,3:] / 255))  # operands could not be broadcast together with shapes (55,433) (55,433,1) 
    # cv2.imshow("DEBUG", frame[y1:y2, x1:x2] * (1 - resized_telop_image[:,:,3:] / 255))  # operands could not be broadcast together with shapes (55,433) (55,433,1)
    cv2.imshow("DEBUG", src1)  # 
    # cv2.imshow("DEBUG", (1 - resized_telop_image[:,:,3:] / 255))  # ネガ化
    # cv2.imshow("DEBUG", (1 - resized_telop_image[:,:] / 255))  # 色味が変わる
    # cv2.imshow("DEBUG", resized_telop_image[:,:,:3] * (resized_telop_image[:,:,3:] / 255))  # 普通*ポジ化 -> ほとんど真っ白
    # cv2.imshow("DEBUG", resized_telop_image[:,:,3:] / 255)  # ポジ化
    # cv2.imshow("DEBUG", resized_telop_image[:,:,:3])  # 普通
    # cv2.imshow("DEBUG", resized_telop_image[:,:,3:])  # ポジ化
    cv2.imshow("DEBUG", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    """
    return  frame

def draw_logo(cal_resized_logo_nums, frame,logo_image,  SET_HEIGHT,SET_WIDTH):
    ## ロゴマークを合成　画面右下
    x1, y1, x2, y2, a, b = cal_resized_logo_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        """TODO
        logging warning level"""
        print("logoが描画できません")
    return frame

# python版
@lru_cache()
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

def return_movie_property(SET_WIDTH: int, vcap) -> tuple:
    SET_WIDTH = SET_WIDTH
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
    SET_HEIGHT: int = int((SET_WIDTH * height) / width)
    return SET_WIDTH,fps,height,width,SET_HEIGHT

def finalize(vcap):
    # 入力動画処理ハンドルのリリース
    vcap.release()
    # ウィンドウの除去
    cv2.destroyAllWindows()

def mp_face_detection_func(small_frame, model_selection=0, min_detection_confidence=0.4):
    face = mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    )
    """refer to
    https://solutions.mediapipe.dev/face_detection#python-solution-api
    """    
    # 推論処理
    inference = face.process(small_frame)
    """
    Processes an RGB image and returns a list of the detected face location data.
    Args:
          image: An RGB image represented as a numpy ndarray.
    Raises:
          RuntimeError: If the underlying graph throws any error.
    ValueError: 
          If the input image is not three channel RGB.
     Returns:
           A NamedTuple object with a "detections" field that contains a list of the
           detected face location data.'
    """
    return inference

def return_face_coordinates(small_frame, SET_WIDTH, SET_HEIGHT, model_selection, min_detection_confidence) -> Tuple:
    """TODO
    計算効率のためのリファクタリング"""
    """
    return:
                face_location_list
                concatenate_face_location_list
                person_frame_list
    """
    small_frame.flags.writeable = False
    face_location_list: List = list()
    concatenate_face_location_list = list()
    person_frame = np.empty([0,0])
    person_frame_list: List = list()
    result = mp_face_detection_func(small_frame, model_selection, min_detection_confidence)
    if not result.detections:
        return face_location_list, concatenate_face_location_list,person_frame_list
    else:
        # print('\n------------')
        # print(f'人数: {len(result.detections)}人')
        # print(f'exec_times: {exec_times}')

        detection_counter:int = 0
        for detection in result.detections:
            xleft:int = int(detection.location_data.relative_bounding_box.xmin * SET_WIDTH)
            xtop :int= int(detection.location_data.relative_bounding_box.ymin * SET_HEIGHT)
            xright:int = int(detection.location_data.relative_bounding_box.width * SET_WIDTH + xleft)
            xbottom:int = int(detection.location_data.relative_bounding_box.height * SET_HEIGHT + xtop)
            """see bellow
            https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
            """
            # print(f'信頼度: {round(detection.score[0]*100, 2)}%')
            # print(f'座標: {(xtop,xright,xbottom,xleft)}')

            # xleft or xtop がマイナスになる場合があるとバグる
            if xleft <= 0 or xtop <= 0:
                continue

            face_location_list.append((xtop,xright,xbottom,xleft))  # faceapi order
            """face_location_listはsmall_frame上の顔座標"""

            # person_frame用コード
            person_frame = small_frame[xtop:xbottom, xleft:xright]
            finally_height_size:int = 200

            """# person_frameの顔周りを拡張する
            expand_size:int = 25  # px
            if xtop - expand_size <= 0:
                xtop = 0
            else:
                xtop = xtop - expand_size
            if xbottom + expand_size >= SET_HEIGHT:
                xbottom = SET_HEIGHT
            else:
                xbottom = xbottom + expand_size
            if xleft - expand_size <= 0:
                xleft = 0
            else:
                xleft = xleft - expand_size
            if xright + expand_size >= SET_WIDTH:
                xright = SET_WIDTH
            else:
                xright = xright + expand_size
            """

            # person_frameをリサイズする
            height:int = xbottom - xtop
            width:int = xright - xleft
            # 拡大・縮小率を算出
            fy:float = finally_height_size / height
            finally_width_size:int = int(width * fy)
            # fx:float = finally_width_size / width
            person_frame = cv2.resize(person_frame, dsize=(finally_width_size, finally_height_size))
            """DEBUG
            cv2.imshow("DEBUG", person_frame)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            """
            person_frame_list.append(person_frame)

            # 拡大率に合わせて各座標を再計算する
            # person_frame上の各座標
            person_frame_xtop:int = 0
            person_frame_xright:int = finally_width_size
            person_frame_xbottom:int = finally_height_size
            person_frame_xleft:int = 0
            # 連結されたperson_frame上の各座標
            concatenated_xtop:int = person_frame_xtop
            concatenated_xright:int = person_frame_xright + (finally_width_size * detection_counter)
            concatenated_xbottom:int = person_frame_xbottom 
            concatenated_xleft:int = person_frame_xleft + (finally_width_size * detection_counter)

            concatenate_face_location_list.append((concatenated_xtop,concatenated_xright,concatenated_xbottom,concatenated_xleft))  # faceapi order
            detection_counter += 1
            """about coordinate order
            dlib: (Left, Top, Right, Bottom,)
            faceapi: (top, right, bottom, left)
            see bellow
            https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
            """
        small_frame.flags.writeable = True
        return face_location_list, concatenate_face_location_list,person_frame_list
"""
    x1 = inner_bottom_area_left
    y1 = inner_bottom_area_top
    x2 = inner_bottom_area_left + unregistered_face_image.shape[1]
    y2 = inner_bottom_area_top + unregistered_face_image.shape[0]
    try:
        small_frame[y1:y2, x1:x2] = small_frame[y1:y2, x1:x2]
"""

def check_compare_faces(known_face_encodings, face_encoding, tolerance):
    try:
        matches = faceapi.compare_faces(known_face_encodings, face_encoding, tolerance)
        return matches
    except:
        print('DEBUG: npKnown.npzが壊れているか予期しないエラーが発生しました。')
        print('npKnown.npzの自動削除は行われません。原因を特定の上、必要な場合npKnown.npzを削除して下さい。')
        print('処理を終了します。FACE01を再起動して下さい。')
        print("以下のエラーをシステム管理者へお伝えください")
        print("---------------------------------------------")
        print(traceback.format_exc(limit=None, chain=True))
        print("---------------------------------------------")
        exit()

# Get face_names
def return_face_names(args_dict, face_names, known_face_encodings, face_encoding, known_face_names, matches, name):
    # 各プリセット顔画像のエンコーディングと動画中の顔画像エンコーディングとの各顔距離を要素としたアレイを算出
    face_distances = faceapi.face_distance(known_face_encodings, face_encoding)  ## face_distances -> shape:(677,), face_encoding -> shape:(128,)
    # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素のインデックスを求める
    best_match_index: int = np.argmin(face_distances)
    # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素の値を求める
    min_face_distance: str = str(min(face_distances))  # あとでファイル名として文字列として加工するので予めstr型にしておく
    # アレイ中のインデックスからknown_face_names中の同インデックスの要素を算出
    face_names = face_names_append(args_dict, matches, best_match_index, min_face_distance, face_names, name)
    return face_names

def draw_pink_rectangle(small_frame, top,bottom,left,right):
    cv2.rectangle(small_frame, (left, top), (right, bottom), (255, 87, 243), 2) # pink
    return small_frame

def draw_white_rectangle(rectangle, small_frame, top, left, right, bottom):
    cv2.rectangle(small_frame, (left-18, top-18), (right+18, bottom+18), (175, 175, 175), 2) # 灰色内枠
    cv2.rectangle(small_frame, (left-20, top-20), (right+20, bottom+20), (255,255,255), 2) # 白色外枠
    return small_frame

# パーセンテージ表示
def display_percentage(percentage_and_symbol,small_frame, p, left, right, bottom, tolerance):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # パーセンテージ表示用の灰色に塗りつぶされた四角形の描画
    cv2.rectangle(small_frame, (left-25, bottom + 75), (right+25, bottom+50), (30,30,30), cv2.FILLED) # 灰色
    # テキスト表示位置
    fontsize = 14
    putText_center = int((left-25 + right+25)/2)
    putText_chaCenter = int(5/2)
    putText_pos = putText_center - (putText_chaCenter*fontsize) - int(fontsize/2)
    putText_position = (putText_pos, bottom + 75 - int(fontsize / 2))
    # toleranceの値によってフォント色を変える
    if p < tolerance:
        # パーセンテージを白文字表示
        small_frame = cv2.putText(small_frame, percentage_and_symbol, putText_position, font, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        # パーセンテージをピンク表示
        small_frame = cv2.putText(small_frame, percentage_and_symbol, putText_position, font, 1, (255, 87, 243), 1, cv2.LINE_AA)
    return small_frame

# デフォルト顔画像の表示面積調整
def adjust_display_area(default_face_image,top,left,right):
    """TODO
    繰り返し計算させないようリファクタリング"""
    _, default_face_image_width = default_face_image.shape[:2]
    default_face_image_ratio = ((right - left) / default_face_image_width / 1.5)
    default_face_small_image = cv2.resize(default_face_image, None, fx = default_face_image_ratio, fy = default_face_image_ratio)  # type: ignore
    x1, y1, x2, y2 = right+5, top, right+5+default_face_small_image.shape[1], top + default_face_small_image.shape[0]
    return x1, y1, x2, y2, default_face_small_image

# 顔部分の領域をクロップ画像ファイルとして出力
def make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, number_of_crops, frequency_crop_image):
    """TODO
    マルチスレッド化"""
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    imgCroped = pil_img_obj_rgb.crop((left -20,top -20,right +20,bottom +20)).resize((200, 200))
    filename = "output/%s_%s_%s.png" % (name, date, dis)
    imgCroped.save(filename)
    return filename,number_of_crops, frequency_crop_image

# デフォルト顔画像の描画処理
def draw_default_face_image(small_frame, default_face_small_image, x1, y1, x2, y2):
    try:
        small_frame[y1:y2, x1:x2] = small_frame[y1:y2, x1:x2] * (1 - default_face_small_image[:,:,3:] / 255) + default_face_small_image[:,:,:3] * (default_face_small_image[:,:,3:] / 255)
    except:
        """TODO
        loggingによる警告レベル"""
        print('デフォルト顔画像の描画が出来ません')
        print('描画面積が足りないか他に問題があります')
    return small_frame

def calculate_text_position(left,right,name,fontsize,bottom):
    center = int((left + right)/2)
    chaCenter = int(len(name)/2)
    pos = center - (chaCenter*fontsize) - int(fontsize/2)
    position = (pos, bottom + (fontsize * 2))
    Unknown_position = (pos + fontsize, bottom + (fontsize * 2))
    return position, Unknown_position

# 帯状四角形（ピンク）の描画
def draw_error_messg_rectangle(small_frame, SET_HEIGHT, SET_WIDTH):
    error_messg_rectangle_top: int  = int((SET_HEIGHT + 20) / 2)
    error_messg_rectangle_bottom : int = int((SET_HEIGHT + 120) / 2)
    error_messg_rectangle_left: int  = 0
    error_messg_rectangle_right : int = SET_WIDTH
    cv2.rectangle(small_frame, (error_messg_rectangle_left, error_messg_rectangle_top), (error_messg_rectangle_right, error_messg_rectangle_bottom), (255, 87, 243), cv2.FILLED)  # pink
    return error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom

# bottom_area_rectangle描画
def draw_bottom_area_rectangle(name,bottom_area, SET_HEIGHT, SET_WIDTH, small_frame):
        bottom_area_rectangle_left: int  = 0
        bottom_area_rectangle_top: int  = SET_HEIGHT
        bottom_area_rectangle_right : int = SET_WIDTH
        bottom_area_rectangle_bottom = bottom_area_rectangle_top + 190
        BLUE: tuple = (255,0,0)
        RED: tuple = (0,0,255)
        GREEN: tuple = (0,255,0)
        if name=='Unknown':
            cv2.rectangle(small_frame, (bottom_area_rectangle_left, bottom_area_rectangle_top), (bottom_area_rectangle_right, bottom_area_rectangle_bottom), RED, cv2.FILLED)
        else:
            small_frame = cv2.rectangle(small_frame, (bottom_area_rectangle_left, bottom_area_rectangle_top), (bottom_area_rectangle_right, bottom_area_rectangle_bottom), BLUE, cv2.FILLED)
        return small_frame

def draw_bottom_area(name,small_frame):
    # default_image描画
    inner_bottom_area_left = 20
    inner_bottom_area_top = SET_HEIGHT + 20
    unregistered_face_image = np.array(Image.open('images/顔画像未登録.png'))
    h: int
    w: int
    h, w = unregistered_face_image.shape[:2]
    WIDTH = 120
    h = int(h * (WIDTH / w))
    try:
        # unregistered_face_image = cv2.resize(unregistered_face_image, None, fx = width_ratio, fy = height_ratio)
        unregistered_face_image = cv2.resize(unregistered_face_image, dsize=(WIDTH, h))
    except:
        pass
    x1 = inner_bottom_area_left
    y1 = inner_bottom_area_top
    x2 = inner_bottom_area_left + unregistered_face_image.shape[1]
    y2 = inner_bottom_area_top + unregistered_face_image.shape[0]
    try:
        small_frame[y1:y2, x1:x2] = small_frame[y1:y2, x1:x2] * (1 - unregistered_face_image[:,:,3:] / 255) + \
                    unregistered_face_image[:,:,:3] * (unregistered_face_image[:,:,3:] / 255)
    except:
        print('下部エリアのデフォルト顔画像が表示できません')
    return unregistered_face_image, small_frame

# ボトムエリア内テキスト描画
def draw_text_in_bottom_area(draw, inner_bottom_area_char_left, inner_bottom_area_char_top,name,percentage_and_symbol,date):
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    """TODO 動作未確認"""
    fontpath = return_fontpath()
    fontsize = 30
    font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
    # テキスト表示位置
    # pos = center - (chaCenter*fontsize) - int(fontsize/2)
    # position = (pos, bottom + (fontsize * 2))
    position = (inner_bottom_area_char_left, inner_bottom_area_char_top)
    # nameとpercentage_and_symbolの描画
    draw.text(position, name, fill=(255, 255, 255, 255), font = font)
    fontsize = 25
    position = (inner_bottom_area_char_left, inner_bottom_area_char_top + fontsize + 5)
    draw.text(position, percentage_and_symbol, fill=(255, 255, 255, 255), font = font)
    # dateの描画
    position = (inner_bottom_area_char_left, inner_bottom_area_char_top + fontsize * 2 + 20)
    fontsize = 12
    font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
    draw.text(position, date, fill=(255, 255, 255, 255), font = font)

# pil_imgオブジェクトを生成
def pil_img_instance(frame):
    pil_img_obj= Image.fromarray(frame)
    return pil_img_obj

# pil_img_rgbオブジェクトを生成
def pil_img_rgb_instance(frame):
    pil_img_obj_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    return  pil_img_obj_rgb

# drawオブジェクトを生成
def  make_draw_object(frame):
    draw = ImageDraw.Draw(Image.fromarray(frame))
    return draw

# draw_rgbオブジェクトを生成
def  make_draw_rgb_object(pil_img_obj_rgb):
    draw_rgb = ImageDraw.Draw(pil_img_obj_rgb)
    return draw_rgb

# pil_img_objをnumpy配列に変換
def convert_pil_img_to_ndarray(pil_img_obj):
    frame = np.array(pil_img_obj)
    return frame

def face_names_append(args_dict, matches, best_match_index, min_face_distance, face_names, name):
    if matches[best_match_index]:  # tolerance以下の人物しかここは通らない。
        """TODO
        arg: known_face_names
        """
        file_name = args_dict["known_face_names"][best_match_index]
        name = file_name + ':' + min_face_distance
    """TODO
    else:
        print("debug")
    """
    face_names.append(name)
    return face_names

def make_error_messg_rectangle_font(fontpath, error_messg_rectangle_fontsize, encoding = 'utf-8'):
    error_messg_rectangle_font = ImageFont.truetype(fontpath, error_messg_rectangle_fontsize, encoding = 'utf-8')
    return error_messg_rectangle_font

def decide_text_position(error_messg_rectangle_bottom,error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_fontsize,error_messg_rectangle_messg):
    error_messg_rectangle_center = int((error_messg_rectangle_left + error_messg_rectangle_right)/2)
    error_messg_rectangle_chaCenter = int(len(error_messg_rectangle_messg)/2)
    error_messg_rectangle_pos = error_messg_rectangle_center - (error_messg_rectangle_chaCenter * error_messg_rectangle_fontsize) - int(error_messg_rectangle_fontsize / 2)
    error_messg_rectangle_position = (error_messg_rectangle_pos + error_messg_rectangle_fontsize, error_messg_rectangle_bottom - (error_messg_rectangle_fontsize * 2))
    return error_messg_rectangle_position

def draw_error_messg_rectangle_messg(draw, error_messg_rectangle_position, error_messg_rectangle_messg, error_messg_rectangle_font):
    draw.text(error_messg_rectangle_position, error_messg_rectangle_messg, fill=(255, 255, 255, 255), font = error_messg_rectangle_font)

def make_frame_datas_array(args_dict, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,small_frame):
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
    person_data_list.append(person_data)
    frame_datas = {'img':small_frame, 'person_data_list': person_data_list}
    frame_datas_array.append(frame_datas)
    return frame_datas_array

def draw_default_face(name, top, left, right, small_frame):
    default_name_png = name + '_default.png'
    default_face_image_name_png = './priset_face_images/' + default_name_png
    """TODO デフォルト顔画像ファイル読み込みを選択性にする"""
    ## for処理毎にディスクからデフォルト顔画像を読み出している点（2021年9月7日）
    ## {name:default_image_ndarray}という辞書的変数を作りメモリに格納するのはどうか→ver.1.2.9で。
    if not name in default_face_image_dict:  # default_face_image_dictにnameが存在しなかった場合
        # 各人物のデフォルト顔画像ファイルの読み込み
        if os.path.exists(default_face_image_name_png):
            """TODO"""
            # WINDOWSのopencv-python4.2.0.32ではcv2.imread()でpng画像を読み込めないバグが
            # 存在する可能性があると思う。そこでPNG画像の読み込みにはpillowを用いることにする
            default_face_image = np.array(Image.open(default_face_image_name_png))
            # BGAをRGBへ変換
            default_face_image = cv2.cvtColor(default_face_image, cv2.COLOR_BGR2RGBA)
        else:
            print(f'{name}さんのデフォルト顔画像ファイルがpriset_face_imagesフォルダに存在しません')
            print(f'{name}さんのデフォルト顔画像ファイルをpriset_face_imagesフォルダに用意してください')
            print('処理を終了します')
            exit()
        # if default_face_image.ndim == 3:  # RGBならアルファチャンネル追加 small_frameがアルファチャンネルを持っているから。
        # default_face_imageをメモリに保持
        default_face_image_dict[name] = default_face_image  # キーnameと値default_face_imageの組み合わせを挿入する
    else:  # default_face_image_dictにnameが存在した場合
        default_face_image = default_face_image_dict[name]  # キーnameに対応する値をdefault_face_imageへ格納する
        x1, y1, x2, y2 , default_face_small_image = adjust_display_area(default_face_image,top,left,right)
        small_frame = draw_default_face_image(small_frame, default_face_small_image, x1, y1, x2, y2)
    return small_frame

def draw_rectangle_for_name(name,small_frame, left, right,bottom):
    if name == 'Unknown':   # nameがUnknownだった場合
        small_frame = cv2.rectangle(small_frame, (left-25, bottom + 25), (right+25, bottom+50), (255, 87, 243), cv2.FILLED) # pink
    else:                   # nameが既知だった場合
        # cv2.rectangle(small_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), thickness = 1) # 濃い水色の線
        small_frame = cv2.rectangle(small_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), cv2.FILLED) # 濃い水色
    return small_frame

def draw_text_for_name(left,right,bottom,name, p,tolerance,pil_img_obj):
    fontpath = return_fontpath()
    """TODO FONTSIZEハードコーティング訂正"""
    fontsize = 14
    font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
    # テキスト表示位置決定
    position, Unknown_position = calculate_text_position(left,right,name,fontsize,bottom)
    # nameの描画
    pil_img_obj = draw_name(name,pil_img_obj, Unknown_position, font, p, tolerance, position)
    # pil_img_objをnumpy配列に変換
    small_frame = convert_pil_img_to_ndarray(pil_img_obj)
    return small_frame

def draw_name(name,pil_img_obj, Unknown_position, font, p, tolerance, position):
    local_draw_obj = ImageDraw.Draw(pil_img_obj)
    if name == 'Unknown':  ## nameがUnknownだった場合
        # draw.text(Unknown_position, '照合不一致', fill=(255, 255, 255, 255), font = font)
        local_draw_obj.text(Unknown_position, '　未登録', fill=(255, 255, 255, 255), font = font)
    else:  ## nameが既知の場合
        # if percentage > 99.0:
        if p < tolerance:
            # nameの描画
            local_draw_obj.text(position, name, fill=(255, 255, 255, 255), font = font)
        else:
            local_draw_obj.text(position, name, fill=(255, 87, 243, 255), font = font)
    return pil_img_obj

# target_rectangleの描画
def draw_target_rectangle(rect01_png,small_frame,top,bottom,left,right,name):
    width_ratio: float
    height_ratio: float
    face_width: int
    face_height: int
    if not name == 'Unknown':  ## nameが既知の場合
        face_width: int = right - left
        face_height: int = bottom - top
        orgHeight, orgWidth = rect01_png.shape[:2]
        width_ratio = 1.0 * (face_width / orgWidth)
        height_ratio = 1.0 * (face_height / orgHeight)
        rect01_png = cv2.resize(rect01_png, None, fx = width_ratio, fy = height_ratio)  # type: ignore
        x1, y1, x2, y2 = left, top, left + rect01_png.shape[1], top + rect01_png.shape[0]
        # TODO ---------------------
        try:
            small_frame[y1:y2, x1:x2] = small_frame[y1:y2, x1:x2] * (1 - rect01_png[:,:,3:] / 255) + \
                        rect01_png[:,:,:3] * (rect01_png[:,:,3:] / 255)
        except:
            """TODO
            logging -> warn level"""
            pass
        # ---- ---------------------
    else:  ## nameがUnknownだった場合
        face_width = right - left
        face_height = bottom - top
        # rect01_NG_png←ピンクのtarget_rectangle
        rect01_NG_png: cv2.Mat = cv2.imread("images/rect01_NG.png", cv2.IMREAD_UNCHANGED)
        orgHeight, orgWidth = rect01_NG_png.shape[:2]
        width_ratio = 1.0 * (face_width / orgWidth)
        height_ratio = 1.0 * (face_height / orgHeight)
        rect01_NG_png = cv2.resize(rect01_NG_png, None, fx = width_ratio, fy = height_ratio)
        x1, y1, x2, y2 = left, top, left + rect01_NG_png.shape[1], top + rect01_NG_png.shape[0]
        try:
            small_frame[y1:y2, x1:x2] = small_frame[y1:y2, x1:x2] * (1 - rect01_NG_png[:,:,3:] / 255) + \
                        rect01_NG_png[:,:,:3] * (rect01_NG_png[:,:,3:] / 255)
        except:
            pass
    return small_frame

def return_percentage(p):  # python版
    percentage = -4.76190475 *(p**2)-(0.380952375*p)+100
    return percentage

# 処理時間の測定
def Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR):
        HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)  ## 小数点以下がミリ秒
        print(f'(1frameあたりの)処理時間: {round(HANDLING_FRAME_TIME * 1000, 2)}[ミリ秒]')
        # fps_ms = fps
        # if frame_skip > 0:
        #     HANDLING_FRAME_TIME / (frame_skip - 1)  < fps_ms
        # elif frame_skip == 0:
        #     HANDLING_FRAME_TIME < fps_ms
        # time.sleep((fps_ms - (HANDLING_FRAME_TIME / (frame_skip - 1))) / 1000)

"""初期設定"""
# 使用期限算出
cal_specify_date()

# Initialize variables (Outer frame)
frame_datas: Dict = {}
default_face_image_dict: Dict = {}
matches: List = [bool]

# 動画の保存 & 画像認識部 ================================================
def main(args_dict):
    
    # 画角値（四隅の座標:Tuple）算出
    if not 'TOP_LEFT' in locals():
        TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER = cal_angle_coordinate(args_dict["height"], args_dict["width"])

    # ###################################################
    # BUG FIX
    # Counter reset (Inner frame)
    unloaded_frame_count: int = 0
    frame_skip_counter: int = 0
    number_of_crops: int = 0
    # Initialize variables (Inner frame)
    percentage:float = 0.0
    person_data_list: List = []
    # 半透明値
    alpha: float = 0.3
    # ###################################################

    # toleranceの算出
    if not 'tolerance' in locals():
        tolerance = to_tolerance(args_dict["similar_percentage"])

    # ⭐️各frameを処理⭐️ -------------------------------------------
    
    # while True:
    while True:
        try:
            frame = video_capture(args_dict["kaoninshoDir"], args_dict["movie"]).__next__()
        except:
            break
        # 処理時間の測定（前半）
        if args_dict["calculate_time"] == True:
            HANDLING_FRAME_TIME_FRONT = time.perf_counter()

        """DEBUG
        type(frame)
        cv2.imshow("DEBUG", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        """

        """TODO"""
        # frame内変数初期化 ========
        # face_location_list: List[Tuple[int,int,int,int]]
        face_encodings:List = []
        # face_encoding: np.ndarray
        face_names: List[str] =[]
        name: str = 'Unknown'
        filename: str = ''
        # percentage_and_symbol: str = ''
        # dis: str = ''
        # p: float = 1.0
        top: int =0
        bottom: int =0
        left: int =0
        right: int =0
        frame_datas_array: List = []
        percentage_and_symbol:str = ''
        # ==========================
        """保留
        # 映像データ入力有無確認
        if ret == False:
            if unloaded_frame_count < 1000:
                unloaded_frame_count += 1
                # print(f'入力信号が確認されません: INPUT ERROR({unloaded_frame_count})')
                continue
            else:
                print('入力信号<ret>がないためプログラムを終了します')
                break
        if len(frame) == 0:
            if unloaded_frame_count < 1000:
                unloaded_frame_count += 1
                # print(f'入力信号が確認されません: INPUT ERROR({unloaded_frame_count})')
                continue
            else:
                print('入力信号<frame>がないためプログラムを終了します')
                break
        """

        # frame_skipの数値に満たない場合は処理をスキップ
        if frame_skip_counter < args_dict["frame_skip"]:
            frame_skip_counter += 1
            continue

        # 画角値をもとに各frameを縮小
        # python版
        frame = angle_of_view_specification(args_dict["set_area"], frame, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER)
        # 各frameリサイズ
        small_frame = resize_frame(args_dict["SET_WIDTH"], args_dict["SET_HEIGHT"], frame)

        if  args_dict["headless"] == False:
            # bottom area描画
            if args_dict["bottom_area"]==True:
                # small_frameの下方向に余白をつける
                small_frame = cv2.copyMakeBorder(small_frame, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        
            # 半透明処理（前半）
            if args_dict["show_overlay"]==True:
                overlay: cv2.Mat = small_frame.copy()

            """
            テロップとロゴマークの合成
            貼り付け先座標を左上にセット
            """
            if args_dict["draw_telop_and_logo"] == True:
                small_frame =  draw_telop(args_dict["cal_resized_telop_nums"], args_dict["SET_WIDTH"], args_dict["resized_telop_image"], small_frame)
                small_frame = draw_logo(args_dict["cal_resized_logo_nums"], small_frame, args_dict["logo_image"],  args_dict["SET_HEIGHT"],args_dict["SET_WIDTH"])

        # 顔認証処理 ここから ====================================================
        # 顔ロケーションを求める
        if args_dict["use_mediapipe"] == True:
            face_location_list, concatenate_face_location_list, person_frame_list = \
                return_face_coordinates(small_frame, args_dict["SET_WIDTH"], args_dict["SET_HEIGHT"], args_dict["model_selection"], args_dict["min_detection_confidence"])
        else:
            face_location_list = faceapi.face_locations(small_frame, args_dict["upsampling"], args_dict["mode"])
        """face_location_list
        [(144, 197, 242, 99), (97, 489, 215, 371)]
        """

        # 顔がなかったら以降のエンコード処理を行わない
        if len(face_location_list) == 0:
            frame_datas_array = make_frame_datas_array(args_dict, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,small_frame)
            yield frame_datas_array
            continue

        """TODO 顔が一定数以上なら以降のエンコード処理を行わない（試験的）
        """
        number_of_people: int = 5
        if len(face_location_list) >= number_of_people:
            print(f'{number_of_people}人以上を検出しました')
            frame_datas_array = make_frame_datas_array(args_dict, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,small_frame)
            yield frame_datas_array
            continue

        """TODO 動作チェック"""
        # ボトムエリア内複数人エラーチェック処理 ---------------------
        if args_dict["should_not_be_multiple_faces"]==True:
            if len(face_location_list) > 1:
                small_frame = draw_pink_rectangle(small_frame, top,bottom,left,right)
                error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom = draw_error_messg_rectangle(small_frame,SET_HEIGHT, args_dict["SET_WIDTH"])
                fontpath = return_fontpath()
                error_messg_rectangle_messg = '複数人が検出されています'
                error_messg_rectangle_fontsize = 24
                error_messg_rectangle_font = make_error_messg_rectangle_font(fontpath, error_messg_rectangle_fontsize, encoding = 'utf-8')
                # テキスト表示位置
                error_messg_rectangle_position = decide_text_position(error_messg_rectangle_bottom,error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_fontsize,error_messg_rectangle_messg)
                # error_messg_rectangle_messgの描画
                draw = make_draw_object(small_frame)
                draw_error_messg_rectangle_messg(draw, error_messg_rectangle_position, error_messg_rectangle_messg, error_messg_rectangle_font)
                # PILをnumpy配列に変換
                small_frame = convert_pil_img_to_ndarray(pil_img_obj)
                frame_datas_array = make_frame_datas_array(args_dict, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,small_frame)
                # frame_datas_array.append(frame_datas)
                yield frame_datas_array
                continue

        """ ⭐️顔がある場合の処理ここから⭐️ """
        # 顔ロケーションからエンコーディングを求める
        if args_dict["use_mediapipe"] == True and  args_dict["person_frame_face_encoding"] == True:
            """FIX
            人数分を繰り返し処理しているので時間がかかる。
            dlibは一つの画像に複数の座標を与えて一度に処理をする。
            なので各person_frameをくっつけて一つの画像にすれば処理時間は短くなる。
                numpy.hstack(tup)[source]
                Stack arrays in sequence horizontally (column wise).
                https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
            """
            concatenate_person_frame = np.hstack(person_frame_list)
            """DEBUG
            cv2.imshow("face_encodings", concatenate_person_frame)
            cv2.moveWindow("face_encodings", 800,600)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            quit()
            print("---------------------------------")
            print(f'concatenate_face_location_list: {concatenate_face_location_list}')
            print("---------------------------------")
            """
            face_encodings = faceapi.face_encodings(concatenate_person_frame, concatenate_face_location_list, args_dict["jitters"], args_dict["model"])
        elif args_dict["use_mediapipe"] == True and  args_dict["person_frame_face_encoding"] == False:
            face_encodings = faceapi.face_encodings(small_frame, face_location_list, args_dict["jitters"], args_dict["model"])
        elif args_dict["use_mediapipe"] == False and  args_dict["person_frame_face_encoding"] == True:
            print("\n---------------------------------")
            print("config.ini:")
            print("mediapipe = False  の場合 person_frame_face_encoding = True  には出来ません")
            print("システム管理者へ連絡の後、設定を変更してください")
            print("処理を終了します")
            print("---------------------------------")
            quit()
        elif args_dict["use_mediapipe"] == False and args_dict["person_frame_face_encoding"] == False:
            face_encodings = faceapi.face_encodings(small_frame, face_location_list, args_dict["jitters"], args_dict["model"])


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
            """
            
        """ ⭐️各frame、face_encodingsに基づいてひとりづつ処理: ここから⭐️ """
        # 名前リスト作成
        for face_encoding in face_encodings:
            # Initialize name, matches (Inner frame)
            name = "Unknown"
            matches = check_compare_faces(args_dict["known_face_encodings"], face_encoding, tolerance)
            # 名前リスト(face_names)生成
            face_names = return_face_names(args_dict, face_names, args_dict["known_face_encodings"], face_encoding, args_dict["known_face_names"], matches, name)

        # face_location_listについて繰り返し処理→frame_datas_array作成
        for (top, right, bottom, left), name in zip(face_location_list, face_names):
            person_data = defaultdict(int)
            default_name_png = ''
            default_face_image_name_png = ''
            if name == 'Unknown':
                # percentage_and_symbol: str = ''
                dis: str = ''
                p: float = 1.0
            else:  # nameが誰かの名前の場合
                distance: str
                name, distance = name.split(':')
                # パーセンテージの算出
                dis = str(round(float(distance), 2))
                p = float(distance)
                # return_percentage(p)
                percentage = return_percentage(p)
                percentage = round(percentage, 1)
                percentage_and_symbol = str(percentage) + '%'
                # ファイル名を最初のアンダーバーで区切る（アンダーバーは複数なのでmaxsplit = 1）
                try:
                    name, _ = name.split('_', maxsplit = 1)
                except:
                    """TODO
                    logging warn level"""
                    sg.popup_error('ファイル名に異常が見つかりました',name,'NAME_default.png あるいはNAME_001.png (001部分は001からはじまる連番)にしてください','noFaceフォルダに移動します')
                    shutil.move(name, './noFace/')
                    continue

            # tolerance未満の場合、'顔画像未登録'に。
            if p > tolerance:
                name = '(不鮮明)' + name

            # クロップ画像保存
            if args_dict["crop_face_image"]==True:
                if args_dict["frequency_crop_image"] < number_of_crops:
                    pil_img_obj_rgb = pil_img_rgb_instance(small_frame)
                    filename,number_of_crops, frequency_crop_image = \
                        make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, number_of_crops, args_dict["frequency_crop_image"])
                    number_of_crops = 0
                else:
                    number_of_crops += 1

            # 描画系
            if args_dict["headless"] == False:
                # デフォルト顔画像の描画
                if p <= tolerance:  # ディスタンスpがtolerance以下の場合
                    if args_dict["default_face_image_draw"] == True:
                        small_frame = draw_default_face(name, top, left, right, small_frame)

                # ピンクまたは白の四角形描画
                if args_dict["rectangle"] == True:
                    if name == 'Unknown':  # プリセット顔画像に対応する顔画像がなかった場合
                        small_frame = draw_pink_rectangle(small_frame, top,bottom,left,right)
                    else:  # プリセット顔画像に対応する顔画像があった場合
                        small_frame = draw_white_rectangle(args_dict["rectangle"], small_frame, top, left, right, bottom)
                    
                # パーセンテージ描画
                if args_dict["show_percentage"]==True:
                    small_frame = display_percentage(percentage_and_symbol,small_frame, p, left, right, bottom, tolerance)

                # 名前表示と名前用四角形の描画
                if args_dict["show_name"]==True:
                    small_frame = draw_rectangle_for_name(name,small_frame, left, right,bottom)
                    pil_img_obj= Image.fromarray(small_frame)
                    small_frame = draw_text_for_name(left,right,bottom,name, p,tolerance,pil_img_obj)

                # target_rectangleの描画
                if args_dict["target_rectangle"] == True:
                    small_frame = draw_target_rectangle(args_dict["rect01_png"], small_frame,top,bottom,left,right,name)

                if args_dict["bottom_area"] == True:
                    small_frame = draw_bottom_area_rectangle(name,args_dict["bottom_area"], SET_HEIGHT, args_dict["SET_WIDTH"], small_frame)

                # bottom_area中の描画
                if args_dict["bottom_area"]==True:
                    unregistered_face_image, small_frame = draw_bottom_area(name,small_frame)
                    # name等描画
                    inner_bottom_area_char_left = 200
                    inner_bottom_area_char_top = SET_HEIGHT + 30
                    # draw  =  make_draw_object(small_frame)
                    draw_text_in_bottom_area(draw, inner_bottom_area_char_left, inner_bottom_area_char_top,name,percentage_and_symbol,date)
                    small_frame = convert_pil_img_to_ndarray(pil_img_obj)

            date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
            person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
            person_data_list.append(person_data)
        # End for (top, right, bottom, left), name in zip(face_location_list, face_names)

        # Make generator_1frameに対して1回
        if args_dict["headless"] == False:
            frame_datas = {'img':small_frame, 'person_data_list': person_data_list}
            # frame_datas_array.append(frame_datas)
        elif args_dict["headless"] == True:
            frame_datas = {'img':'', 'person_data_list': person_data_list}
            # frame_datas_array.append(frame_datas)
        
        if args_dict["headless"] == False:
            # 半透明処理（後半）_1frameに対して1回
            if args_dict["show_overlay"]==True:
                cv2.addWeighted(overlay, alpha, small_frame, 1-alpha, 0, small_frame)
        
        yield frame_datas
        # yield frame_datas_array

        """機能停止
        # yield出力ブロック ===================================
        ## パイプ出力機構も含む
        ## TODO: frame_datas_arrayから値を取り出す処理に変えること
        if not frame_datas == None:
            if output_frame_data == True:  ## pipe出力時
                # frame_datas['stream'] = small_frame
                # yield frame_datas
                pass
            elif output_frame_data == False:  ## 通常使用時
                frame_datas = {'name': name, 'pict':filename,  'date':date, 'img':small_frame, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
                frame_datas_array.append(frame_datas)
                yield frame_datas_array
                # sys.stdout.buffer.write(frame_datas['stream'])  ## 'stream'を出力する
                # print(type(small_frame))  ## <class 'numpy.ndarray'>
                # print(type(frame_datas['stream']))  ## <class 'numpy.ndarray'>

                # cv2.imshow('FACE01', frame_datas['stream'])
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        # =====================================================

        # パイプ使用時の必要情報を表示 ========
        if print_property==True:
            print('fps: ', fps)
            print('frame shape: ', small_frame.shape)  ## (450, 800, 3)
            print('dtype: ', small_frame.dtype)  ## uint8
            print('frame size: ', small_frame.size) ## 1080000←450*800*3
            exit()
        # =====================================
        """
        
        # Reset frame_skip_counter to 0
        frame_skip_counter = 0

        # 処理時間の測定（後半）
        if args_dict["calculate_time"] == True:
            HANDLING_FRAME_TIME_REAR = time.perf_counter()
            Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR)

    # End of while ---------------------------------------------

    finalize(args_dict["vcap"])

"""TODO
マルチスレッド化"""
# from concurrent.futures import ThreadPoolExecutor
# with ThreadPoolExecutor() as th:
#     th.submit(main())


# main =================================================================
if __name__ == '__main__':
    """ 並行処理用コード_1
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    # pool = ProcessPoolExecutor()
    pool = ThreadPoolExecutor()
    # window実装
    layout = [
        # [sg.Text('FACE01 GRAPHICS ver.1.2.8', font=('BIZ-UDGOTHICB.TTC', 15)), sg.Text('GUI実装例', font=('BIZ-UDGOTHICB.TTC', 10))],
        [sg.Image(filename='', key='cam1', pad=(0,0))],
        [sg.Button('終了', key='terminate', pad=(0,10))]
    ]
    window = sg.Window(
        'window1', layout, alpha_channel = 1, margins=(0, 0),
        no_titlebar = True, grab_anywhere = True,
        location=(350,130), modal = True
    )
    # 並行処理
    def multi(x):
        img, person_data_list = x['img'], x['person_data_list']
        for person_data in person_data_list:
            name, pict, date,  location, percentage_and_symbol = person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
            if not name == 'Unknown':
                print(
                    "並行処理用コード_1が動作しています", "\n",
                    name, "\n",
                    "\t", "類似度\t", percentage_and_symbol, "\n",
                    "\t", "座標\t", location, "\n",
                    "\t", "時刻\t", date, "\n",
                    "\t", "出力\t", pict, "\n",
                    "-------\n"
                )
            person_data_list.pop(0)
            return img
    for array_x in xs:
        for x in array_x:
            event, _ = window.read(timeout = 1)
            # befor_time = time.perf_counter()
            result = pool.submit(multi, x)
            if  not result.result() is None:
                imgbytes = cv2.imencode(".png", result.result())[1].tobytes()
                window["cam1"].update(data = imgbytes)
                # after_time = time.perf_counter()
                # print(f'xs処理時間: {round((after_time - befor_time) * 1000, 2)}[ミリ秒]')
        if event=='terminate':
            break
    window.close()
    print('終了します')
    """


    """ 並行処理用コード_2
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    # pool = ProcessPoolExecutor()
    pool = ThreadPoolExecutor()
    # window実装
    layout = [
        # [sg.Text('FACE01 GRAPHICS ver.1.2.8', font=('BIZ-UDGOTHICB.TTC', 15)), sg.Text('GUI実装例', font=('BIZ-UDGOTHICB.TTC', 10))],
        [sg.Image(filename='', key='cam1', pad=(0,0))],
        [sg.Button('終了', key='terminate', pad=(0,10))]
    ]
    window = sg.Window(
        'window1', layout, alpha_channel = 1, margins=(0, 0),
        no_titlebar = True, grab_anywhere = True,
        location=(350,130), modal = True
    )
    # 並行処理
    def multi(array_x):
        for x in array_x:
            img, person_data_list = x['img'], x['person_data_list']
            for person_data in person_data_list:
                name, pict, date,  location, percentage_and_symbol = person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                if not name == 'Unknown':
                    print(
                        "並行処理用コード_2が動作しています", "\n",
                        name, "\n",
                        "\t", "類似度\t", percentage_and_symbol, "\n",
                        "\t", "座標\t", location, "\n",
                        "\t", "時刻\t", date, "\n",
                        "\t", "出力\t", pict, "\n",
                        "-------\n"
                    )
                person_data_list.pop(0)
                return img
    # main処理
    for array_x in xs:
        event, _ = window.read(timeout = 1)
        result = pool.submit(multi, array_x)
        if  not result.result() is None:
            imgbytes = cv2.imencode(".png", result.result())[1].tobytes()
            window["cam1"].update(data = imgbytes)
            # after_time = time.perf_counter()
            # print(f'xs処理時間: {round((after_time - befor_time) * 1000, 2)}[ミリ秒]')
        if event=='terminate':
            break
    window.close()
    print('終了します')
    """


    """BUG 並行処理用コード_3
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    pool = ProcessPoolExecutor()
    # pool = ThreadPoolExecutor()
    # 並行処理
    def multi(xs):
        for array_x in xs:
            for x in array_x:
                img, person_data_list = x['img'], x['person_data_list']
                for person_data in person_data_list:
                    name, pict, date,  location, percentage_and_symbol = person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                    if not name == 'Unknown':
                        print(
                            "並行処理用コード_3が動作しています", "\n",
                            name, "\n",
                            "\t", "類似度\t", percentage_and_symbol, "\n",
                            "\t", "座標\t", location, "\n",
                            "\t", "時刻\t", date, "\n",
                            "\t", "出力\t", pict, "\n",
                            "-------\n"
                        )
                    person_data_list.pop(0)
                    return img
        print('終了します')
    while True:
        result = pool.submit(multi, xs)
    """


# """プロファイリング用コード
    import cProfile as pr
    # headless = False
    headless = True
    if headless == False:
        layout = [
            [sg.Image(filename='', key='display', pad=(0,0))],
            [sg.Button('終了', key='terminate', pad=(0,10))]
        ]
        window = sg.Window(
        'CALL_FACE01GRAPHICS', layout, alpha_channel = 1, margins=(10, 10),
        location=(350,130), modal = True
    )
    
    exec_times: int = 50
    profile_HANDLING_FRAME_TIME: float = 0.0
    profile_HANDLING_FRAME_TIME_FRONT: float = 0.0
    profile_HANDLING_FRAME_TIME_REAR: float = 0.0

    def profile(exec_times):
        profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
        main_func_signature = initialize(configure(), headless)
        for frame_datas in main(main_func_signature):
            exec_times = exec_times - 1
            if  exec_times <= 0:
                break
            else:
                print(f'exec_times: {exec_times}')
                if headless == False:
                    event, _ = window.read(timeout = 1)
                img, person_data_list = frame_datas['img'], frame_datas['person_data_list']
                for person_data in person_data_list:
                    name, pict, date,  location, percentage_and_symbol = person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                    if not name == 'Unknown':
                        print(
                            "プロファイリング用コードが動作しています", "\n",
                            "statsファイルが出力されます", "\n",
                            name, "\n",
                            "\t", "類似度\t", percentage_and_symbol, "\n",
                            "\t", "座標\t", location, "\n",
                            "\t", "時刻\t", date, "\n",
                            "\t", "出力\t", pict, "\n",
                            "-------\n"
                        )
                del person_data_list

                if headless == False:
                    imgbytes = cv2.imencode(".png", img)[1].tobytes()
                    window["display"].update(data = imgbytes)
            if headless == False:
                if event =='terminate':
                    break
        if headless == False:
            window.close()
        print('終了します')
        profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
        profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
        print(f'profile()関数の処理時間合計: {round(profile_HANDLING_FRAME_TIME , 3)}[秒]')
    pr.run('profile(exec_times)', 'restats')
# """
