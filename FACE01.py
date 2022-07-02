from datetime import datetime
import logging
from sys import version_info, version, exit
from time import perf_counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from os import chdir, environ
from os.path import dirname, exists
from platform import system
from shutil import move
from traceback import format_exc
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg
from GPUtil import getGPUs
from PIL import Image, ImageDraw, ImageFont
from psutil import cpu_count, cpu_freq, virtual_memory

import face01lib.api as faceapi
import face01lib.video_capture as video_capture
from face01lib.load_priset_image import load_priset_image
from face01lib.similar_percentage_to_tolerance import to_tolerance

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

sg.theme('LightGray')

"""mediapipe for python, see bellow
https://github.com/google/mediapipe/tree/master/mediapipe/python
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
faceapi: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
"""

global_memory = {
# 半透明値,
'alpha' : 0.3,
'number_of_crops' : 0
}

# ホームディレクトリ固定✅
def home() -> tuple:
    kaoninshoDir: str = dirname(__file__)
    chdir(kaoninshoDir)
    priset_face_imagesDir: str = f'{dirname(__file__)}/priset_face_images/'
    return kaoninshoDir, priset_face_imagesDir

# configファイル読み込み✅
def configure():
    kaoninshoDir, priset_face_imagesDir = home()
    try:
        conf = ConfigParser()
        conf.read('config.ini', 'utf-8')
        # dict作成
        conf_dict = {
            'kaoninshoDir' :kaoninshoDir,
            'priset_face_imagesDir' :priset_face_imagesDir,
            'headless' : conf.getboolean('DEFAULT','headless'),
            'model' : conf.get('DEFAULT','model'),
            'similar_percentage' : float(conf.get('DEFAULT','similar_percentage')),
            'jitters' : int(conf.get('DEFAULT','jitters')),
            'number_of_people' : int(conf.get('DEFAULT','number_of_people')),
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
            'set_width' : int(conf.get('DEFAULT','set_width')),
            'default_face_image_draw' : conf.getboolean('DEFAULT', 'default_face_image_draw'),
            'show_overlay' : conf.getboolean('DEFAULT', 'show_overlay'),
            'show_percentage' : conf.getboolean('DEFAULT', 'show_percentage'),
            'crop_face_image' : conf.getboolean('DEFAULT', 'crop_face_image'),
            'show_name' : conf.getboolean('DEFAULT', 'show_name'),
            'draw_telop_and_logo' : conf.getboolean('DEFAULT', 'draw_telop_and_logo'),
            'use_mediapipe' : conf.getboolean('DEFAULT','use_mediapipe'),
            'model_selection' : int(conf.get('DEFAULT','model_selection')),
            'min_detection_confidence' : float(conf.get('DEFAULT','min_detection_confidence')),
            'person_frame_face_encoding' : conf.getboolean('DEFAULT','person_frame_face_encoding'),
            'crop_with_multithreading' : conf.getboolean('DEFAULT','crop_with_multithreading'),
            'Python_version': conf.get('DEFAULT','Python_version'),
            'cpu_freq': conf.get('DEFAULT','cpu_freq'),
            'cpu_count': conf.get('DEFAULT','cpu_count'),
            'memory': conf.get('DEFAULT','memory'),
            'gpu_check' : conf.getboolean('DEFAULT','gpu_check'),
            'user': conf.get('DEFAULT','user'),
            'passwd': conf.get('DEFAULT','passwd'),
        }
        return conf_dict
    except:
        logger.warning("config.ini 読み込み中にエラーが発生しました")
        logger.exception("conf_dictが正常に作成できませんでした")
        logger.warning("以下のエラーをシステム管理者へお伝えください")
        logger.warning("-" * 20)
        logger.warning(format_exc(limit=None, chain=True))
        logger.warning("-" * 20)
        logger.warning("終了します")
        exit(0)

# configure関数実行
conf_dict = configure()

# 評価版のみ実行
def cal_specify_date() -> None:
    """指定日付計算
    """
    limit_date = datetime(2022, 12, 1, 0, 0, 0)   # 指定日付
    today = datetime.now()

    def limit_date_alart() -> None:
        if today >= limit_date:
            print('指定日付を過ぎました')
            sg.popup('サンプルアプリケーションをお使いいただきありがとうございます','使用可能期限を過ぎました', '引き続きご利用になる場合は下記までご連絡下さい', '東海顔認証　担当：袈裟丸','y.kesamaru@tokai-kaoninsho.com', '', 'アプリケーションを終了します', title='', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
            exit(0)
        elif today < limit_date:
            remaining_days = limit_date - today
            if remaining_days.days < 30:
                dialog_text = 'お使い頂ける残日数は' + str(remaining_days.days) + '日です'
                sg.popup('サンプルアプリケーションをお使いいただきありがとうございます', dialog_text, title='', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
    limit_date_alart()
cal_specify_date()

def cal_resized_telop_image(resized_telop_image):
    x1, y1, x2, y2 = 0, 0, resized_telop_image.shape[1], resized_telop_image.shape[0]
    a = (1 - resized_telop_image[:,:,3:] / 255)
    b = resized_telop_image[:,:,:3] * (resized_telop_image[:,:,3:] / 255)
    cal_resized_telop_nums = (x1, y1, x2, y2, a, b)
    return cal_resized_telop_nums

def cal_resized_logo_image(resized_logo_image,  set_height,set_width):
    x1, y1, x2, y2 = set_width - resized_logo_image.shape[1], set_height - resized_logo_image.shape[0], set_width, set_height
    a = (1 - resized_logo_image[:,:,3:] / 255)
    b = resized_logo_image[:,:,:3] * (resized_logo_image[:,:,3:] / 255)
    cal_resized_logo_nums = (x1, y1, x2, y2, a, b)
    return cal_resized_logo_nums

"""CHECK SYSTEM INFORMATION"""
# Initialize variables, load images
def initialize(conf_dict):
    headless = conf_dict["headless"]
    known_face_encodings, known_face_names = load_priset_image(conf_dict["kaoninshoDir"],conf_dict["priset_face_imagesDir"])

    # set_width,fps,height,width,set_height
    set_width,fps,height,width,set_height = \
        video_capture.return_movie_property(conf_dict["set_width"], video_capture.return_vcap(conf_dict["movie"]))
    
    # toleranceの算出
    tolerance = to_tolerance(conf_dict["similar_percentage"])

    # 画像読み込み系
    if headless == False:
        # それぞれの画像が1度だけしか読み込まれない仕組み
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
            ratio: float = conf_dict["set_width"] / orgWidth / 3  ## テロップ幅は横幅を分母として設定
            resized_telop_image = cv2.resize(telop_image, None, fx = ratio, fy = ratio)
            cal_resized_telop_nums = cal_resized_telop_image(resized_telop_image)

        # Load Logo image
        logo_image: cv2.Mat
        if not load_logo_image:
            logo_image: cv2.Mat = cv2.imread("images/Logo.png", cv2.IMREAD_UNCHANGED)
            load_logo_image = True
            _, logoWidth = logo_image.shape[:2]
            logoRatio = conf_dict["set_width"] / logoWidth / 15
            resized_logo_image = cv2.resize(logo_image, None, fx = logoRatio, fy = logoRatio)
            cal_resized_logo_nums = cal_resized_logo_image(resized_logo_image,  set_height,set_width)

        # Load unregistered_face_image
        unregistered_face_image: cv2.Mat
        if not load_unregistered_face_image:
            unregistered_face_image = np.array(Image.open('./images/顔画像未登録.png'))
            unregistered_face_image = cv2.cvtColor(unregistered_face_image, cv2.COLOR_BGR2RGBA)
            load_unregistered_face_image = True

    # 日付時刻算出
    date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒

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
            'height': height,
            'width': width,
            'set_height': set_height,
            'tolerance': tolerance,
            'default_face_image_dict': {}
        }
    elif headless == True:
        init_dict = {
            'known_face_encodings': known_face_encodings,
            'known_face_names': known_face_names,
            'date': date,
            'height': height,
            'width': width,
            'set_height': set_height,
            'tolerance': tolerance,
            'default_face_image_dict': {}
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
        args_dict['draw_telop_and_logo'] = False
        args_dict['person_frame_face_encoding'] = False
        args_dict['headless'] = True

    return args_dict

# initialize関数実行
args_dict = initialize(conf_dict)

def system_check(args_dict):
    logger.info("FACE01の推奨動作環境を満たしているかシステムチェックを実行します")
    logger.info("- Python version check")
    major_ver, minor_ver_1, minor_ver_2 = args_dict["Python_version"].split('.', maxsplit = 3)
    if (version_info < (int(major_ver), int(minor_ver_1), int(minor_ver_2))):
        logger.warning("警告: Python 3.8.10以降を使用してください")
        exit(0)
    else:
        logger.info(f"  [OK] {str(version)}")
    # CPU
    logger.info("- CPU check")
    if cpu_freq().max < float(args_dict["cpu_freq"]) * 1_000 or cpu_count(logical=False) < int(args_dict["cpu_count"]):
        logger.warning("CPU性能が足りません")
        logger.warning("処理速度が実用に達しない恐れがあります")
        logger.warning("終了します")
        exit(0)
    else:
        logger.info(f"  [OK] {str(cpu_freq().max)[0] + '.' +  str(cpu_freq().max)[1:3]}GHz")
        logger.info(f"  [OK] {cpu_count(logical=False)}core")
    # MEMORY
    logger.info("- Memory check")
    if virtual_memory().total < int(args_dict["memory"]) * 1_000_000_000:
        logger.warning("メモリーが足りません")
        logger.warning("少なくとも4GByte以上が必要です")
        logger.warning("終了します")
        exit(0)
    else:
        if int(virtual_memory().total) < 10:
            logger.info(f"  [OK] {str(virtual_memory().total)[0]}GByte")
        else:
            logger.info(f"  [OK] {str(virtual_memory().total)[0:2]}GByte")
    # GPU
    logger.info("- CUDA devices check")
    if args_dict["gpu_check"] == True:
        if faceapi.dlib.cuda.get_num_devices() == 0:
            logger.warning("CUDAが有効なデバイスが見つかりません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] cuda devices: {faceapi.dlib.cuda.get_num_devices()}")

        # Dlib build check: CUDA
        logger.info("- Dlib build check: CUDA")
        if faceapi.dlib.DLIB_USE_CUDA == False:
            logger.warning("dlibビルド時にCUDAが有効化されていません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] DLIB_USE_CUDA: True")

        # Dlib build check: BLAS
        logger.info("- Dlib build check: BLAS, LAPACK")
        if faceapi.dlib.DLIB_USE_BLAS == False or faceapi.dlib.DLIB_USE_LAPACK == False:
            logger.warning("BLASまたはLAPACKのいずれか、あるいは両方がインストールされていません")
            logger.warning("パッケージマネージャーでインストールしてください")
            logger.warning("\tCUBLAS native runtime libraries(Basic Linear Algebra Subroutines: 基本線形代数サブルーチン)")
            logger.warning("\tLAPACK バージョン 3.X(線形代数演算を行う総合的な FORTRAN ライブラリ)")
            logger.warning("インストール後にdlibを改めて再インストールしてください")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info("  [OK] DLIB_USE_BLAS, LAPACK: True")

        # VRAM check
        logger.info("- VRAM check")
        for gpu in getGPUs():
            gpu_memory = gpu.memoryTotal
            gpu_name = gpu.name
        if gpu_memory < 3000:
            logger.warning("GPUデバイスの性能が足りません")
            logger.warning(f"現在のGPUデバイス: {gpu_name}")
            logger.warning("NVIDIA GeForce GTX 1660 Ti以上をお使いください")
            logger.warning("終了します")
            exit(0)
        else:
            if int(gpu_memory) < 9999:
                logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0]}GByte")
            else:
                logger.info(f"  [OK] VRAM: {str(int(gpu_memory))[0:2]}GByte")
            logger.info(f"  [OK] GPU device: {gpu_name}")

    logger.info("  ** System check: Done **\n")

# system_check関数実行
if not exists("face01.log"):
    system_check(args_dict)

def return_fontpath():
    # フォントの設定(フォントファイルのパスと文字の大きさ)
    operating_system: str  = system()
    fontpath: str = ''
    if (operating_system == 'Linux'):
        fontpath = "/usr/share/fonts/truetype/mplus/mplus-1mn-bold.ttf"
    elif (operating_system == 'Windows'):
                    # fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICR.TTC"
        fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICB.TTC"  ## bold体
    else:
        logger.info('オペレーティングシステムの確認が出来ません。システム管理者にご連絡ください')
    return fontpath

def draw_telop(cal_resized_telop_nums, set_width: int, resized_telop_image: np.ndarray, frame: np.ndarray):
    x1, y1, x2, y2, a, b = cal_resized_telop_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        logger.info("telopが描画できません")
    return  frame

def draw_logo(cal_resized_logo_nums, frame,logo_image,  set_height,set_width):
    ## ロゴマークを合成
    x1, y1, x2, y2, a, b = cal_resized_logo_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        logger.info("logoが描画できません")
    return frame

def mp_face_detection_func(resized_frame, model_selection=0, min_detection_confidence=0.4):
    face = mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    )
    """refer to
    https://solutions.mediapipe.dev/face_detection#python-solution-api
    """    
    # 推論処理
    inference = face.process(resized_frame)
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

def return_face_location_list(resized_frame, set_width, set_height, model_selection, min_detection_confidence) -> Tuple:
    """
    return: face_location_list
    """
    resized_frame.flags.writeable = False
    face_location_list: List = []
    person_frame = np.empty((2,0), dtype=np.float64)
    result = mp_face_detection_func(resized_frame, model_selection, min_detection_confidence)
    if not result.detections:
        return face_location_list
    else:
        for detection in result.detections:
            xleft:int = int(detection.location_data.relative_bounding_box.xmin * set_width)
            xtop :int= int(detection.location_data.relative_bounding_box.ymin * set_height)
            xright:int = int(detection.location_data.relative_bounding_box.width * set_width + xleft)
            xbottom:int = int(detection.location_data.relative_bounding_box.height * set_height + xtop)
            # see bellow
            # https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
            
            if xleft <= 0 or xtop <= 0:  # xleft or xtop がマイナスになる場合があるとバグる
                continue
            face_location_list.append((xtop,xright,xbottom,xleft))  # faceapi order

    resized_frame.flags.writeable = True
    return face_location_list

def return_concatenate_location_and_frame(resized_frame, face_location_list):
    """face_location_listはresized_frame上の顔座標"""
    finally_height_size:int = 150
    concatenate_face_location_list = []
    detection_counter:int = 0
    person_frame_list: List = list()
    for xtop,xright,xbottom,xleft in face_location_list:
        person_frame = resized_frame[xtop:xbottom, xleft:xright]
        # person_frameをリサイズする
        height:int = xbottom - xtop
        width:int = xright - xleft
        # 拡大・縮小率を算出
        fy:float = finally_height_size / height
        finally_width_size:int = int(width * fy)
        # fx:float = finally_width_size / width
        person_frame = cv2.resize(person_frame, dsize=(finally_width_size, finally_height_size))
        person_frame_list.append(person_frame)
        """DEBUG
        cv2.imshow("DEBUG", person_frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        """
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

        concatenate_person_frame = np.hstack(person_frame_list)
        """DEBUG
        cv2.imshow("face_encodings", concatenate_person_frame)
        cv2.moveWindow("face_encodings", 800,600)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        exit(0)
        print("---------------------------------")
        print(f'concatenate_face_location_list: {concatenate_face_location_list}')
        print("---------------------------------")
        """
    return concatenate_face_location_list, concatenate_person_frame
    

def check_compare_faces(known_face_encodings, face_encoding, tolerance):
    try:
        matches = faceapi.compare_faces(known_face_encodings, face_encoding, tolerance)
        return matches
    except:
        logger.warning("DEBUG: npKnown.npzが壊れているか予期しないエラーが発生しました。")
        logger.warning("npKnown.npzの自動削除は行われません。原因を特定の上、必要な場合npKnown.npzを削除して下さい。")
        logger.warning("処理を終了します。FACE01を再起動して下さい。")
        logger.warning("以下のエラーをシステム管理者へお伝えください")
        logger.warning("-" * 20)
        logger.warning(format_exc(limit=None, chain=True))
        logger.warning("-" * 20)
        exit(0)

# Get face_names
def return_face_names(args_dict, face_names, face_encoding, matches, name):
    # 各プリセット顔画像のエンコーディングと動画中の顔画像エンコーディングとの各顔距離を要素としたアレイを算出
    face_distances = faceapi.face_distance(args_dict["known_face_encodings"], face_encoding)  ## face_distances -> shape:(677,), face_encoding -> shape:(128,)
    # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素のインデックスを求める
    best_match_index: int = np.argmin(face_distances)
    # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素の値を求める
    min_face_distance: str = str(min(face_distances))  # あとでファイル名として文字列として加工するので予めstr型にしておく
    # アレイ中のインデックスからknown_face_names中の同インデックスの要素を算出
    if matches[best_match_index]:  # tolerance以下の人物しかここは通らない。
        file_name = args_dict["known_face_names"][best_match_index]
        name = file_name + ':' + min_face_distance
    face_names.append(name)
    return face_names

def draw_pink_rectangle(resized_frame, top,bottom,left,right):
    cv2.rectangle(resized_frame, (left, top), (right, bottom), (255, 87, 243), 2) # pink
    return resized_frame

def draw_white_rectangle(rectangle, resized_frame, top, left, right, bottom):
    cv2.rectangle(resized_frame, (left-18, top-18), (right+18, bottom+18), (175, 175, 175), 2) # 灰色内枠
    cv2.rectangle(resized_frame, (left-20, top-20), (right+20, bottom+20), (255,255,255), 2) # 白色外枠
    return resized_frame

# パーセンテージ表示
def display_percentage(percentage_and_symbol,resized_frame, p, left, right, bottom, tolerance):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # パーセンテージ表示用の灰色に塗りつぶされた四角形の描画
    cv2.rectangle(resized_frame, (left-25, bottom + 75), (right+25, bottom+50), (30,30,30), cv2.FILLED) # 灰色
    # テキスト表示位置
    fontsize = 14
    putText_center = int((left-25 + right+25)/2)
    putText_chaCenter = int(5/2)
    putText_pos = putText_center - (putText_chaCenter*fontsize) - int(fontsize/2)
    putText_position = (putText_pos, bottom + 75 - int(fontsize / 2))
    # toleranceの値によってフォント色を変える
    if p < tolerance:
        # パーセンテージを白文字表示
        resized_frame = cv2.putText(resized_frame, percentage_and_symbol, putText_position, font, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        # パーセンテージをピンク表示
        resized_frame = cv2.putText(resized_frame, percentage_and_symbol, putText_position, font, 1, (255, 87, 243), 1, cv2.LINE_AA)
    return resized_frame

# 顔部分の領域をクロップ画像ファイルとして出力
def make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, number_of_crops, frequency_crop_image):
    date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    imgCroped = pil_img_obj_rgb.crop((left -20,top -20,right +20,bottom +20)).resize((200, 200))
    filename = "output/%s_%s_%s.png" % (name, date, dis)
    imgCroped.save(filename)
    return filename,number_of_crops, frequency_crop_image

# デフォルト顔画像の表示面積調整
def adjust_display_area(args_dict, default_face_image):
    """TODO
    繰り返し計算させないようリファクタリング"""
    face_image_width = int(args_dict["set_width"] / 15)
    default_face_small_image = cv2.resize(default_face_image, dsize=(face_image_width, face_image_width))  # 幅・高さともに同じとする
    # 高さ = face_image_width
    x1, y1, x2, y2 = 0, args_dict["set_height"] - face_image_width - 10, face_image_width, args_dict["set_height"] - 10
    return x1, y1, x2, y2, default_face_small_image, face_image_width

# デフォルト顔画像の描画処理
def draw_default_face_image(resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width):
    try:
        x1 = x1 + (number_of_people * face_image_width)
        x2 = x2 + (number_of_people * face_image_width)
        resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - default_face_small_image[:,:,3:] / 255) + default_face_small_image[:,:,:3] * (default_face_small_image[:,:,3:] / 255)
        # resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * a + b  # ValueError: assignment destination is read-only
        """DEBUG"""
        # frame_imshow_for_debug(resized_frame)
    except:
        logger.info('デフォルト顔画像の描画が出来ません')
        logger.info('描画面積が足りないか他に問題があります')
    return resized_frame

def draw_default_face(args_dict, name, resized_frame, number_of_people):
    default_face_image_dict = args_dict["default_face_image_dict"]

    default_name_png = name + '_default.png'
    default_face_image_name_png = './priset_face_images/' + default_name_png
    if not name in default_face_image_dict:  # default_face_image_dictにnameが存在しなかった場合
        # 各人物のデフォルト顔画像ファイルの読み込み
        if exists(default_face_image_name_png):
            # WINDOWSのopencv-python4.2.0.32ではcv2.imread()でpng画像を読み込めないバグが
            # 存在する可能性があると思う。そこでPNG画像の読み込みにはpillowを用いることにする
            default_face_image = np.array(Image.open(default_face_image_name_png))
            """DEBUG
            frame_imshow_for_debug(default_face_image)
            """
            
            # BGAをRGBへ変換
            default_face_image = cv2.cvtColor(default_face_image, cv2.COLOR_BGR2RGBA)
            """DEBUG
            frame_imshow_for_debug(default_face_image)
            """

        else:
            logger.info(f'{name}さんのデフォルト顔画像ファイルがpriset_face_imagesフォルダに存在しません')
            logger.info(f'{name}さんのデフォルト顔画像ファイルをpriset_face_imagesフォルダに用意してください')
        # if default_face_image.ndim == 3:  # RGBならアルファチャンネル追加 resized_frameがアルファチャンネルを持っているから。
        # default_face_imageをメモリに保持
        default_face_image_dict[name] = default_face_image  # キーnameと値default_face_imageの組み合わせを挿入する
    else:  # default_face_image_dictにnameが存在した場合
        default_face_image = default_face_image_dict[name]  # キーnameに対応する値をdefault_face_imageへ格納する
        """DEBUG
        frame_imshow_for_debug(default_face_image)  # OK
        """
        x1, y1, x2, y2 , default_face_small_image, face_image_width = adjust_display_area(args_dict, default_face_image)
        resized_frame = draw_default_face_image(resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width)
    return resized_frame

def draw_rectangle_for_name(name,resized_frame, left, right,bottom):
    if name == 'Unknown':   # nameがUnknownだった場合
        resized_frame = cv2.rectangle(resized_frame, (left-25, bottom + 25), (right+25, bottom+50), (255, 87, 243), cv2.FILLED) # pink
    else:                   # nameが既知だった場合
        # cv2.rectangle(resized_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), thickness = 1) # 濃い水色の線
        resized_frame = cv2.rectangle(resized_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), cv2.FILLED) # 濃い水色
    return resized_frame

def calculate_text_position(left,right,name,fontsize,bottom):
    center = int((left + right)/2)
    chaCenter = int(len(name)/2)
    pos = center - (chaCenter*fontsize) - int(fontsize/2)
    position = (pos, bottom + (fontsize * 2))
    Unknown_position = (pos + fontsize, bottom + (fontsize * 2))
    return position, Unknown_position

# 帯状四角形（ピンク）の描画
def draw_error_messg_rectangle(resized_frame, set_height, set_width):
    error_messg_rectangle_top: int  = int((set_height + 20) / 2)
    error_messg_rectangle_bottom : int = int((set_height + 120) / 2)
    error_messg_rectangle_left: int  = 0
    error_messg_rectangle_right : int = set_width
    cv2.rectangle(resized_frame, (error_messg_rectangle_left, error_messg_rectangle_top), (error_messg_rectangle_right, error_messg_rectangle_bottom), (255, 87, 243), cv2.FILLED)  # pink
    return error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom

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

def make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame):
    """データ構造(frame_datas_list)を返す

    person_data:
        {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
        
        person_data内のlocationは個々人の顔座標です。個々人を特定しない場合の顔座標はframe_detas['face_location_list']を使ってください。
    
    person_data_list: 
        person_data_list.append(person_data)
    
    frame_datas:
        {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}

    frame_datas_list: 
        frame_datas_array.append(frame_datas)

    return: frame_datas_list
    """    
    date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
    person_data_list.append(person_data)
    frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
    frame_datas_array.append(frame_datas)
    return frame_datas_array

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
    resized_frame = convert_pil_img_to_ndarray(pil_img_obj)
    return resized_frame

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
def draw_target_rectangle(rect01_png,resized_frame,top,bottom,left,right,name):
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
        rect01_png = cv2.resize(rect01_png, None, fx = width_ratio, fy = height_ratio)
        x1, y1, x2, y2 = left, top, left + rect01_png.shape[1], top + rect01_png.shape[0]
        try:
            resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - rect01_png[:,:,3:] / 255) + \
                        rect01_png[:,:,:3] * (rect01_png[:,:,3:] / 255)
        except:
            pass
    else:  ## nameがUnknownだった場合
        fx: float = 0.0
        face_width = right - left
        face_height = bottom - top
        # rect01_NG_png←ピンクのtarget_rectangle
        rect01_NG_png: cv2.Mat = cv2.imread("images/rect01_NG.png", cv2.IMREAD_UNCHANGED)
        orgHeight, orgWidth = rect01_NG_png.shape[:2]
        width_ratio = float(1.0 * (face_width / orgWidth))
        height_ratio = 1.0 * (face_height / orgHeight)
        rect01_NG_png = cv2.resize(rect01_NG_png, None, fx = width_ratio, fy = height_ratio)
        x1, y1, x2, y2 = left, top, left + rect01_NG_png.shape[1], top + rect01_NG_png.shape[0]
        try:
            resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - rect01_NG_png[:,:,3:] / 255) + \
                        rect01_NG_png[:,:,:3] * (rect01_NG_png[:,:,3:] / int(255))
        except:
            pass
    return resized_frame

def return_percentage(p):  # python版
    percentage = -4.76190475 *(p**2)-(0.380952375*p)+100
    return percentage

# 処理時間の測定（算出）
def Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR):
        HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)  ## 小数点以下がミリ秒
        logger.info(f'処理時間: {round(HANDLING_FRAME_TIME * 1000, 2)}[ミリ秒]')

# 処理時間の測定（前半）
def Measure_processing_time_forward():
    if args_dict["calculate_time"] == True:
        HANDLING_FRAME_TIME_FRONT = perf_counter()
        return HANDLING_FRAME_TIME_FRONT

# 処理時間の測定（後半）
def Measure_processing_time_backward():
    if args_dict["calculate_time"] == True:
        HANDLING_FRAME_TIME_FRONT = Measure_processing_time_forward()
        HANDLING_FRAME_TIME_REAR = perf_counter()
        Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR)

# デバッグ用imshow()
def frame_imshow_for_debug(frame):
    # print(type(frame))
    if isinstance(frame, np.ndarray):
        cv2.imshow("DEBUG", frame)
        cv2.moveWindow('window DEBUG', 0, 0)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        for fra in frame:
            fr = fra["img"]
            cv2.imshow("DEBUG", fr)
            cv2.moveWindow('window DEBUG', 0, 0)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

# フレーム前処理
def frame_pre_processing(args_dict, resized_frame):
    person_data_list = []
    name = 'Unknown'
    filename = ''
    top = ()
    bottom = ()
    left = ()
    right = ()
    frame_datas_array = []
    face_location_list = []
    percentage_and_symbol = ''
    overlay = np.empty(0)

    # 描画系（bottom area, 半透明, telop, logo）
    if  args_dict["headless"] == False:
        """1.3.06でボトムエリア描画は廃止予定
        # bottom area描画
        if args_dict["bottom_area"]==True:
            # resized_frameの下方向に余白をつける
            resized_frame = cv2.copyMakeBorder(resized_frame, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        """

        # 半透明処理（前半）
        if args_dict["show_overlay"]==True:
            overlay: cv2.Mat = resized_frame.copy()

        # テロップとロゴマークの合成
        if args_dict["draw_telop_and_logo"] == True:
            resized_frame =  draw_telop(args_dict["cal_resized_telop_nums"], args_dict["set_width"], args_dict["resized_telop_image"], resized_frame)
            resized_frame = draw_logo(args_dict["cal_resized_logo_nums"], resized_frame, args_dict["logo_image"],  args_dict["set_height"],args_dict["set_width"])

    # 顔座標算出
    if args_dict["use_mediapipe"] == True:
        face_location_list = return_face_location_list(resized_frame, args_dict["set_width"], args_dict["set_height"], args_dict["model_selection"], args_dict["min_detection_confidence"])
    else:
        face_location_list = faceapi.face_locations(resized_frame, args_dict["upsampling"], args_dict["mode"])
    """face_location_list
    [(144, 197, 242, 99), (97, 489, 215, 371)]
    """

    # 顔がなかったら以降のエンコード処理を行わない
    if len(face_location_list) == 0:
        frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
        return frame_datas_array

    # 顔が一定数以上なら以降のエンコード処理を行わない
    if len(face_location_list) >= args_dict["number_of_people"]:
        logger.info(f'{args_dict["number_of_people"]}人以上を検出しました')
        frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
        return frame_datas_array

    frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
    return frame_datas_array

# 顔のエンコーディング
# @profile()
def face_encoding_process(args_dict, frame_datas_array):
    """frame_datas_arrayの定義
    person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
    person_data_list.append(person_data)
    frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
    frame_datas_array.append(frame_datas)
    """
    face_encodings = []
    for frame_data in frame_datas_array:
        resized_frame = frame_data["img"]
        face_location_list = frame_data["face_location_list"]  # [(139, 190, 257, 72)]
        if len(face_location_list) == 0:
            return face_encodings, frame_datas_array
        elif len(face_location_list) > 0:
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
                concatenate_face_location_list, concatenate_person_frame = \
                    return_concatenate_location_and_frame(resized_frame, face_location_list)
                face_encodings = faceapi.face_encodings(concatenate_person_frame, concatenate_face_location_list, args_dict["jitters"], args_dict["model"])
            elif args_dict["use_mediapipe"] == True and  args_dict["person_frame_face_encoding"] == False:
                face_encodings = faceapi.face_encodings(resized_frame, face_location_list, args_dict["jitters"], args_dict["model"])
            elif args_dict["use_mediapipe"] == False and  args_dict["person_frame_face_encoding"] == True:
                logger.warning("config.ini:")
                logger.warning("mediapipe = False  の場合 person_frame_face_encoding = True  には出来ません")
                logger.warning("システム管理者へ連絡の後、設定を変更してください")
                logger.warning("-" * 20)
                logger.warning(format_exc(limit=None, chain=True))
                logger.warning("-" * 20)
                logger.warning("処理を終了します")
                exit(0)
            elif args_dict["use_mediapipe"] == False and args_dict["person_frame_face_encoding"] == False:
                face_encodings = faceapi.face_encodings(resized_frame, face_location_list, args_dict["jitters"], args_dict["model"])
        return face_encodings, frame_datas_array

# フレーム後処理
# @profile()
def frame_post_processing(args_dict, face_encodings, frame_datas_array, global_memory):
    """frame_datas_arrayの定義
    person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
    person_data_list.append(person_data)
    frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
    frame_datas_array.append(frame_datas)
    """
    face_names = []
    face_location_list = []
    filename = ''
    debug_frame_turn_count = 0
    modified_frame_list = []

    for frame_data in frame_datas_array:
        if "face_location_list" not in frame_data:
            if args_dict["headless"] == False:
                # 半透明処理（後半）_1frameに対して1回
                if args_dict["show_overlay"]==True:
                    cv2.addWeighted(frame_data["overlay"], global_memory["alpha"], frame_data["img"], 1-global_memory["alpha"], 0, frame_data["img"])
            continue

        resized_frame = frame_data["img"]
        face_location_list = frame_data["face_location_list"]
        overlay = frame_data["overlay"]
        person_data_list = frame_data["person_data_list"]
        date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")

        # 名前リスト作成
        for face_encoding in face_encodings:
            # Initialize name, matches (Inner frame)
            name = "Unknown"
            matches = check_compare_faces(args_dict["known_face_encodings"], face_encoding, args_dict["tolerance"])
            # 名前リスト(face_names)生成
            face_names = return_face_names(args_dict, face_names, face_encoding,  matches, name)

        # face_location_listについて繰り返し処理→frame_datas_array作成
        number_of_people = 0  # 人数。計算上0人から始める。draw_default_face()で使用する
        for (top, right, bottom, left), name in zip(face_location_list, face_names):
            person_data = defaultdict(int)
            if name == 'Unknown':
                percentage_and_symbol: str = ''
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
                    logger warn level"""
                    sg.popup_error('ファイル名に異常が見つかりました',name,'NAME_default.png あるいはNAME_001.png (001部分は001からはじまる連番)にしてください','noFaceフォルダに移動します')
                    move(name, './noFace/')
                    return

            # クロップ画像保存
            if args_dict["crop_face_image"]==True:
                if args_dict["frequency_crop_image"] < global_memory['number_of_crops']:
                    pil_img_obj_rgb = pil_img_rgb_instance(resized_frame)
                    if args_dict["crop_with_multithreading"] == True:
                        # """1.3.08 multithreading 9.05s
                        with ThreadPoolExecutor() as executor:
                            future = executor.submit(make_crop_face_image, name, dis, pil_img_obj_rgb, top, left, right, bottom, global_memory['number_of_crops'], args_dict["frequency_crop_image"])
                            filename,number_of_crops, frequency_crop_image = future.result()
                        # """
                    else:
                        # """ORIGINAL: 1.3.08で変更 8.69s
                        filename,number_of_crops, frequency_crop_image = \
                            make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, global_memory['number_of_crops'], args_dict["frequency_crop_image"])
                        # """
                    global_memory['number_of_crops'] = 0
                else:
                    global_memory['number_of_crops'] += 1

            # 描画系
            if args_dict["headless"] == False:
                # デフォルト顔画像の描画
                if p <= args_dict["tolerance"]:  # ディスタンスpがtolerance以下の場合
                    if args_dict["default_face_image_draw"] == True:
                        resized_frame = draw_default_face(args_dict, name, resized_frame, number_of_people)
                        number_of_people += 1  # 何人目か
                        """DEBUG"""
                        # frame_imshow_for_debug(resized_frame)

                # ピンクまたは白の四角形描画
                if args_dict["rectangle"] == True:
                    if name == 'Unknown':  # プリセット顔画像に対応する顔画像がなかった場合
                        resized_frame = draw_pink_rectangle(resized_frame, top,bottom,left,right)
                    else:  # プリセット顔画像に対応する顔画像があった場合
                        resized_frame = draw_white_rectangle(args_dict["rectangle"], resized_frame, top, left, right, bottom)
                    
                # パーセンテージ描画
                if args_dict["show_percentage"]==True:
                    resized_frame = display_percentage(percentage_and_symbol,resized_frame, p, left, right, bottom, args_dict["tolerance"])
                    """DEBUG"""
                    # frame_imshow_for_debug(resized_frame)

                # 名前表示と名前用四角形の描画
                if args_dict["show_name"]==True:
                    resized_frame = draw_rectangle_for_name(name,resized_frame, left, right,bottom)
                    pil_img_obj= Image.fromarray(resized_frame)
                    resized_frame = draw_text_for_name(left,right,bottom,name, p,args_dict["tolerance"],pil_img_obj)
                    """DEBUG"""
                    # frame_imshow_for_debug(resized_frame)

                # target_rectangleの描画
                if args_dict["target_rectangle"] == True:
                    resized_frame = draw_target_rectangle(args_dict["rect01_png"], resized_frame,top,bottom,left,right,name)
                    """DEBUG
                    frame_imshow_for_debug(resized_frame)
                    """

            person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
            person_data_list.append(person_data)
        # End for (top, right, bottom, left), name in zip(face_location_list, face_names)

        # _1frameに対して1回
        if args_dict["headless"] == False:
            frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
            """DEBUG
            frame_imshow_for_debug(resized_frame)
            frame_datas_array.append(frame_datas)
            """
            modified_frame_list.append(frame_datas)

        elif args_dict["headless"] == True:
            frame_datas = {'img':'no-data_img', 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}  # TypeError: list indices must be integers or slices, not str -> img
            # frame_datas_array.append(frame_datas)
            modified_frame_list.append(frame_datas)
        else:
            frame_datas = {'img':'no-data_img', 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': 'no-data_person_data_list'} 
            # frame_datas_array.append(frame_datas)
            modified_frame_list.append(frame_datas)

        if args_dict["headless"] == False:
            # 半透明処理（後半）_1frameに対して1回
            if args_dict["show_overlay"]==True:
                # cv2.addWeighted(overlay, global_memory["alpha"], resized_frame, 1-global_memory["alpha"], 0, resized_frame)
                for modified_frame in modified_frame_list:
                    cv2.addWeighted(modified_frame["overlay"], global_memory["alpha"], modified_frame["img"], 1-global_memory["alpha"], 0, modified_frame["img"])
                # """DEBUG"""
                # frame_imshow_for_debug(resized_frame)
        
    # return frame_datas
    """DEBUG
    print(f"modified_frame_list.__sizeof__(): {modified_frame_list.__sizeof__()}MB")
    """
    return modified_frame_list

frame_generator_obj = video_capture.frame_generator(args_dict)
# @profile()
def main_process():
    frame_datas_array = frame_pre_processing(args_dict, frame_generator_obj.__next__())
    face_encodings, frame_datas_array = face_encoding_process(args_dict, frame_datas_array)
    frame_datas_array = frame_post_processing(args_dict, face_encodings, frame_datas_array, global_memory)
    yield frame_datas_array

# main =================================================================
if __name__ == '__main__':
    import cProfile as pr

    exec_times: int = 100
    
    profile_HANDLING_FRAME_TIME: float = 0.0
    profile_HANDLING_FRAME_TIME_FRONT: float = 0.0
    profile_HANDLING_FRAME_TIME_REAR: float = 0.0

    # PySimpleGUIレイアウト
    if args_dict["headless"] == False:
        layout = [
            [sg.Image(filename='', key='display', pad=(0,0))],
            [sg.Button('終了', key='terminate', pad=(0,10), expand_x=True)]
        ]
        window = sg.Window(
            'FACE01 プロファイリング利用例', layout, alpha_channel = 1, margins=(10, 10),
            location=(150,130), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
        )

    profile_HANDLING_FRAME_TIME_FRONT = perf_counter()

    # from memory_profiler import profile
    # @profile()
    def profile(exec_times):
        while True:
            frame_datas_array = main_process().__next__()
            if StopIteration == frame_datas_array:
                logger.info("StopIterationです")
                break
            exec_times = exec_times - 1
            if  exec_times <= 0:
                break
            else:
                print(f'exec_times: {exec_times}')
                if args_dict["headless"] == False:
                    event, _ = window.read(timeout = 1)
                    if event == sg.WIN_CLOSED:
                        logger.info("ウィンドウが手動で閉じられました")
                        break
                for frame_datas in frame_datas_array:
                    if "face_location_list" in frame_datas:
                        img, face_location_list, overlay, person_data_list = \
                            frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
                        for person_data in person_data_list:
                            name, pict, date,  location, percentage_and_symbol = \
                                person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
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
                                """DEBUG"""
                                print(f"args_dict.__sizeof__(): {args_dict.__sizeof__()}MB")
                        if args_dict["headless"] == False:
                            imgbytes = cv2.imencode(".png", img)[1].tobytes()
                            window["display"].update(data = imgbytes)
            if args_dict["headless"] == False:
                if event =='terminate':
                    break
        if args_dict["headless"] == False:
            window.close()
        print('プロファイリングを終了します')
        
        profile_HANDLING_FRAME_TIME_REAR = perf_counter()
        profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
        print(f'profile()関数の処理時間合計: {round(profile_HANDLING_FRAME_TIME , 3)}[秒]')

    pr.run('profile(exec_times)', 'restats')