"""CHECK SYSTEM INFORMATION"""
from GPUtil import getGPUs
import PySimpleGUI as sg
import face01lib.api as faceapi
import platform
import psutil
import sys
import click
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

sg.theme('LightGray')

def system_check():
    """TODO
    解決できるURLを指定すること
    テキストファイルを生成して同じ処理を繰り返させないこと
    """
    logging.info("FACE01の推奨動作環境を満たしているかシステムチェックを実行します")
    # Python version
    logging.info("- Python version check")
    if (sys.version_info < (3, 8)):
        logging.warning("警告: Python 3.8以降を使用してください")
        sg.popup(
            "警告: Python 3.8以降を使用してください",
            title='INFORMATION', button_type =sg. POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        logging.info(f"\t[OK] {str(sys.version)}")
        # logging.info(f"\t[OK] {str(sys.version).replace('\n', '')}")
    # CPU
    logging.info("- CPU check")
    if psutil.cpu_freq().max < 3000 or psutil.cpu_count(logical=False) < 4:
        sg.popup(
            'CPU性能が足りません',
            '3GHz以上のCPUが必要です',
            '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        logging.info(f"\t[OK] {str(psutil.cpu_freq().max)[0] + '.' +  str(psutil.cpu_freq().max)[1:3]}GHz")
        logging.info(f"\t[OK] {psutil.cpu_count(logical=False)}core")
    # MEMORY
    logging.info("- Memory check")
    if psutil.virtual_memory().total < 4000000000:
        sg.popup(
        'メモリーが足りません',
        '少なくとも4GByte以上が必要です',
        '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        if psutil.virtual_memory().total < 10000000000:
            logging.info(f"\t[OK] {str(psutil.virtual_memory().total)[0]}GByte"); exit()
        else:
            logging.info(f"\t[OK] {str(psutil.virtual_memory().total)[0:2]}GByte")

    # GPU
    logging.info("- CUDA devices check")
    if faceapi.dlib.cuda.get_num_devices() == 0:
        sg.popup(
        'CUDAが有効なデバイスが見つかりません',
        '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        logging.info(f"\t[OK] cuda devices: {faceapi.dlib.cuda.get_num_devices()}")

    # Dlib build check: CUDA
    logging.info("- Dlib build check: CUDA")
    if faceapi.dlib.DLIB_USE_CUDA == False:
        sg.popup('dlibビルド時にCUDAが有効化されていません',
        '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        logging.info(f"\t[OK] DLIB_USE_CUDA: True")

    # Dlib build check: BLAS
    logging.info("- Dlib build check: BLAS, LAPACK")
    if faceapi.dlib.DLIB_USE_BLAS == False or faceapi.dlib.DLIB_USE_LAPACK == False:
        sg.popup(
            'BLASまたはLAPACKのいずれか、あるいは両方がインストールされていません',
            'パッケージマネージャーでインストールしてください',
            'CUBLAS native runtime libraries(Basic Linear Algebra Subroutines: 基本線形代数サブルーチン)',
            'LAPACK バージョン 3.X(線形代数演算を行う総合的な FORTRAN ライブラリ)',
            'インストール後にdlibを改めて再インストールしてください',
            '終了します', title='INFORMATION', button_type = sg.POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
        exit()
    else:
        logging.info("\t[OK] DLIB_USE_BLAS, LAPACK: True")

    # VRAM check
    logging.info("- VRAM check")
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
        exit()
    else:
        if int(gpu_memory) < 9999:
            logging.info(f"\t[OK] VRAM: {str(int(gpu_memory))[0]}GByte")
        else:
            logging.info(f"\t[OK] VRAM: {str(int(gpu_memory))[0:2]}GByte")
        logging.info(f"\t[OK] GPU device: {gpu_name}")

    logging.info("  ** System check: Done **\n")
system_check()

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import configparser
import datetime
import os
from pickletools import uint8
import shutil
import time
from collections import defaultdict
from functools import lru_cache, partial
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
import face01lib.video_capture as video_capture

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

global_memory = {
# 半透明値,
'alpha' : 0.3,
}

# ホームディレクトリ固定
def home() -> tuple:
    kaoninshoDir: str = os.path.dirname(__file__)
    os.chdir(kaoninshoDir)
    priset_face_imagesDir: str = f'{os.path.dirname(__file__)}/priset_face_images/'
    return kaoninshoDir, priset_face_imagesDir

# configファイル読み込み
def configure():
    kaoninshoDir, priset_face_imagesDir = home()
    try:
        conf = configparser.ConfigParser()
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
        logging.warning("config.ini 読み込み中にエラーが発生しました")
        logging.warning("以下のエラーをシステム管理者へお伝えください")
        logging.warning("---------------------------------------------")
        logging.warning(traceback.format_exc(limit=None, chain=True))
        logging.warning("---------------------------------------------")
        quit()

# configure関数実行
conf_dict = configure()

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
# 評価版のみ実行する
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
            ratio: float = conf_dict["set_width"] / orgWidth / 1.5  ## テロップ幅は横幅の半分に設定
            resized_telop_image = cv2.resize(telop_image, None, fx = ratio, fy = ratio)  # type: ignore
            cal_resized_telop_nums = cal_resized_telop_image(resized_telop_image)

        # Load Logo image
        logo_image: cv2.Mat
        if not load_logo_image:
            logo_image: cv2.Mat = cv2.imread("images/Logo.png", cv2.IMREAD_UNCHANGED)
            load_logo_image = True
            _, logoWidth = logo_image.shape[:2]
            logoRatio = conf_dict["set_width"] / logoWidth / 10
            resized_logo_image = cv2.resize(logo_image, None, fx = logoRatio, fy = logoRatio)
            cal_resized_logo_nums = cal_resized_logo_image(resized_logo_image,  set_height,set_width)

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
        args_dict['should_not_be_multiple_faces'] = False
        args_dict['bottom_area'] = False
        args_dict['draw_telop_and_logo'] = False
        args_dict['person_frame_face_encoding'] = False
        args_dict['headless'] = True

    return args_dict

# initialize関数実行
args_dict = initialize(conf_dict)

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
        logging.info('オペレーティングシステムの確認が出来ません。システム管理者にご連絡ください')
    return fontpath

def draw_telop(cal_resized_telop_nums, set_width: int, resized_telop_image: np.ndarray, frame: np.ndarray):
    x1, y1, x2, y2, a, b = cal_resized_telop_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        logging.info("telopが描画できません")

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

def draw_logo(cal_resized_logo_nums, frame,logo_image,  set_height,set_width):
    ## ロゴマークを合成　画面右下
    x1, y1, x2, y2, a, b = cal_resized_logo_nums
    try:
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    except:
        logging.info("logoが描画できません")
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
    """TODO
    計算効率のためのリファクタリング"""
    """
    return: face_location_list
    """
    resized_frame.flags.writeable = False
    face_location_list: List = list()
    person_frame = np.empty([0,0])
    result = mp_face_detection_func(resized_frame, model_selection, min_detection_confidence)
    if not result.detections:
        return face_location_list
    else:
        # print('\n------------')
        # print(f'人数: {len(result.detections)}人')
        # print(f'exec_times: {exec_times}')
        for detection in result.detections:
            xleft:int = int(detection.location_data.relative_bounding_box.xmin * set_width)
            xtop :int= int(detection.location_data.relative_bounding_box.ymin * set_height)
            xright:int = int(detection.location_data.relative_bounding_box.width * set_width + xleft)
            xbottom:int = int(detection.location_data.relative_bounding_box.height * set_height + xtop)
            """see bellow
            https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
            """
            # print(f'信頼度: {round(detection.score[0]*100, 2)}%')
            # print(f'座標: {(xtop,xright,xbottom,xleft)}')

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
    """# person_frameの顔周りを拡張する
    expand_size:int = 25  # px
    if xtop - expand_size <= 0:
        xtop = 0
    else:
        xtop = xtop - expand_size
    if xbottom + expand_size >= set_height:
        xbottom = set_height
    else:
        xbottom = xbottom + expand_size
    if xleft - expand_size <= 0:
        xleft = 0
    else:
        xleft = xleft - expand_size
    if xright + expand_size >= set_width:
        xright = set_width
    else:
        xright = xright + expand_size
    """
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
        quit()
        print("---------------------------------")
        print(f'concatenate_face_location_list: {concatenate_face_location_list}')
        print("---------------------------------")
        """
    return concatenate_face_location_list, concatenate_person_frame
    """
        x1 = inner_bottom_area_left
        y1 = inner_bottom_area_top
        x2 = inner_bottom_area_left + unregistered_face_image.shape[1]
        y2 = inner_bottom_area_top + unregistered_face_image.shape[0]
        try:
            resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2]
    """

def check_compare_faces(known_face_encodings, face_encoding, tolerance):
    try:
        matches = faceapi.compare_faces(known_face_encodings, face_encoding, tolerance)
        return matches
    except:
        logging.warning("DEBUG: npKnown.npzが壊れているか予期しないエラーが発生しました。")
        logging.warning("npKnown.npzの自動削除は行われません。原因を特定の上、必要な場合npKnown.npzを削除して下さい。")
        logging.warning("処理を終了します。FACE01を再起動して下さい。")
        logging.warning("以下のエラーをシステム管理者へお伝えください")
        logging.warning("---------------------------------------------")
        logging.warning(traceback.format_exc(limit=None, chain=True))
        logging.warning("---------------------------------------------")
        exit()

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
    """TODO
    マルチスレッド化"""
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
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
    """DEBUG"""
    # frame_imshow_for_debug(resized_frame)  # OK
    # x1, y1, x2, y2, a, b = cal_resized_logo_nums
    # x1, y1, x2, y2, a, b = cal_default_face_small_image(default_face_small_image)
    # try:
    #     frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * a + b
    try:
        x1 = x1 + (number_of_people * face_image_width)
        x2 = x2 + (number_of_people * face_image_width)
        resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - default_face_small_image[:,:,3:] / 255) + default_face_small_image[:,:,:3] * (default_face_small_image[:,:,3:] / 255)
        # resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * a + b  # ValueError: assignment destination is read-only
        """DEBUG"""
        # frame_imshow_for_debug(resized_frame)
    except:
        logging.info('デフォルト顔画像の描画が出来ません')
        logging.info('描画面積が足りないか他に問題があります')
    return resized_frame

def draw_default_face(args_dict, name, resized_frame, number_of_people):
    default_face_image_dict = args_dict["default_face_image_dict"]

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
            """DEBUG"""
            # frame_imshow_for_debug(default_face_image)
            
            # BGAをRGBへ変換
            default_face_image = cv2.cvtColor(default_face_image, cv2.COLOR_BGR2RGBA)
            """DEBUG"""
            # frame_imshow_for_debug(default_face_image)

        else:
            logging.warning(f'{name}さんのデフォルト顔画像ファイルがpriset_face_imagesフォルダに存在しません')
            logging.warning(f'{name}さんのデフォルト顔画像ファイルをpriset_face_imagesフォルダに用意してください')
            logging.warning('処理を終了します')
            exit()
        # if default_face_image.ndim == 3:  # RGBならアルファチャンネル追加 resized_frameがアルファチャンネルを持っているから。
        # default_face_imageをメモリに保持
        default_face_image_dict[name] = default_face_image  # キーnameと値default_face_imageの組み合わせを挿入する
    else:  # default_face_image_dictにnameが存在した場合
        default_face_image = default_face_image_dict[name]  # キーnameに対応する値をdefault_face_imageへ格納する
        """DEBUG"""
        # frame_imshow_for_debug(default_face_image)  # OK
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

# bottom_area_rectangle描画
def draw_bottom_area_rectangle(name,bottom_area, set_height, set_width, resized_frame):
        bottom_area_rectangle_left: int  = 0
        bottom_area_rectangle_top: int  = set_height
        bottom_area_rectangle_right : int = set_width
        bottom_area_rectangle_bottom = bottom_area_rectangle_top + 190
        BLUE: tuple = (255,0,0)
        RED: tuple = (0,0,255)
        GREEN: tuple = (0,255,0)
        if name=='Unknown':
            cv2.rectangle(resized_frame, (bottom_area_rectangle_left, bottom_area_rectangle_top), (bottom_area_rectangle_right, bottom_area_rectangle_bottom), RED, cv2.FILLED)
        else:
            resized_frame = cv2.rectangle(resized_frame, (bottom_area_rectangle_left, bottom_area_rectangle_top), (bottom_area_rectangle_right, bottom_area_rectangle_bottom), BLUE, cv2.FILLED)
        return resized_frame

def draw_bottom_area(name,resized_frame):
    # default_image描画
    inner_bottom_area_left = 20
    inner_bottom_area_top = set_height + 20
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
        resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - unregistered_face_image[:,:,3:] / 255) + \
                    unregistered_face_image[:,:,:3] * (unregistered_face_image[:,:,3:] / 255)
    except:
        logging.info('下部エリアのデフォルト顔画像が表示できません')
    return unregistered_face_image, resized_frame

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
    """_summary_ = 
    'person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
     person_data_list.append(person_data)
     frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
     frame_datas_array.append(frame_datas)
    """    
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
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
        rect01_png = cv2.resize(rect01_png, None, fx = width_ratio, fy = height_ratio)  # type: ignore
        x1, y1, x2, y2 = left, top, left + rect01_png.shape[1], top + rect01_png.shape[0]
        # TODO ---------------------
        try:
            resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - rect01_png[:,:,3:] / 255) + \
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
            resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * (1 - rect01_NG_png[:,:,3:] / 255) + \
                        rect01_NG_png[:,:,:3] * (rect01_NG_png[:,:,3:] / 255)
        except:
            pass
    return resized_frame

def return_percentage(p):  # python版
    percentage = -4.76190475 *(p**2)-(0.380952375*p)+100
    return percentage

# 処理時間の測定（算出）
def Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR):
        HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)  ## 小数点以下がミリ秒
        logging.info(f'処理時間: {round(HANDLING_FRAME_TIME * 1000, 2)}[ミリ秒]')

# 処理時間の測定（前半）
def Measure_processing_time_forward():
    if args_dict["calculate_time"] == True:
        HANDLING_FRAME_TIME_FRONT = time.perf_counter()
        return HANDLING_FRAME_TIME_FRONT

# 処理時間の測定（後半）
def Mesure_processing_time_backward():
    if args_dict["calculate_time"] == True:
        HANDLING_FRAME_TIME_FRONT = Measure_processing_time_forward()
        HANDLING_FRAME_TIME_REAR = time.perf_counter()
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
        # bottom area描画
        if args_dict["bottom_area"]==True:
            # resized_frameの下方向に余白をつける
            resized_frame = cv2.copyMakeBorder(resized_frame, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    
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
        logging.info(f'{args_dict["number_of_people"]}人以上を検出しました')
        frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
        return frame_datas_array

    """TODO 動作チェック"""
    # ボトムエリア内複数人エラーチェック処理 ---------------------
    if args_dict["should_not_be_multiple_faces"]==True:
        if len(face_location_list) > 1:
            resized_frame = draw_pink_rectangle(resized_frame, top,bottom,left,right)
            error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom = draw_error_messg_rectangle(resized_frame,args_dict["set_height"], args_dict["set_width"])
            fontpath = return_fontpath()
            error_messg_rectangle_messg = '複数人が検出されています'
            error_messg_rectangle_fontsize = 24
            error_messg_rectangle_font = make_error_messg_rectangle_font(fontpath, error_messg_rectangle_fontsize, encoding = 'utf-8')
            # テキスト表示位置
            error_messg_rectangle_position = decide_text_position(error_messg_rectangle_bottom,error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_fontsize,error_messg_rectangle_messg)
            # error_messg_rectangle_messgの描画
            draw = make_draw_object(resized_frame)
            draw_error_messg_rectangle_messg(draw, error_messg_rectangle_position, error_messg_rectangle_messg, error_messg_rectangle_font)
            # PILをnumpy配列に変換
            resized_frame = convert_pil_img_to_ndarray(pil_img_obj)
            frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
            # frame_datas_array.append(frame_datas)
            return frame_datas_array
        
    frame_datas_array = make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame)
    return frame_datas_array

# 顔のエンコーディング
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
                logging.warning("\n---------------------------------")
                logging.warning("config.ini:")
                logging.warning("mediapipe = False  の場合 person_frame_face_encoding = True  には出来ません")
                logging.warning("システム管理者へ連絡の後、設定を変更してください")
                logging.warning("処理を終了します")
                logging.warning("---------------------------------")
                quit()
            elif args_dict["use_mediapipe"] == False and args_dict["person_frame_face_encoding"] == False:
                face_encodings = faceapi.face_encodings(resized_frame, face_location_list, args_dict["jitters"], args_dict["model"])
        return face_encodings, frame_datas_array

# フレーム後処理
def frame_post_processing(args_dict, face_encodings, frame_datas_array):
    """frame_datas_arrayの定義
    person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
    person_data_list.append(person_data)
    frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
    frame_datas_array.append(frame_datas)
    """
    face_names = []
    face_location_list = []
    number_of_crops = 0
    filename = ''
    debug_frame_turn_count = 0
    modified_frame_list = []

    for frame_data in frame_datas_array:
        """DEBUG"""
        # debug_frame_turn_count += 1; print('*******',debug_frame_turn_count, '周目*******')
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
        date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")

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
                    logging warn level"""
                    sg.popup_error('ファイル名に異常が見つかりました',name,'NAME_default.png あるいはNAME_001.png (001部分は001からはじまる連番)にしてください','noFaceフォルダに移動します')
                    shutil.move(name, './noFace/')
                    return


            # クロップ画像保存
            if args_dict["crop_face_image"]==True:
                if args_dict["frequency_crop_image"] < number_of_crops:
                    pil_img_obj_rgb = pil_img_rgb_instance(resized_frame)
                    filename,number_of_crops, frequency_crop_image = \
                        make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, number_of_crops, args_dict["frequency_crop_image"])
                    number_of_crops = 0
                else:
                    number_of_crops += 1

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
                    """DEBUG"""
                    # frame_imshow_for_debug(resized_frame)

                if args_dict["bottom_area"] == True:
                    resized_frame = draw_bottom_area_rectangle(name,args_dict["bottom_area"], args_dict["set_height"], args_dict["set_width"], resized_frame)

                # bottom_area中の描画
                if args_dict["bottom_area"]==True:
                    unregistered_face_image, resized_frame = draw_bottom_area(name,resized_frame)
                    # name等描画
                    inner_bottom_area_char_left = 200
                    inner_bottom_area_char_top = args_dict["set_height"] + 30
                    # draw  =  make_draw_object(resized_frame)
                    draw_text_in_bottom_area(draw, inner_bottom_area_char_left, inner_bottom_area_char_top,name,percentage_and_symbol,date)
                    resized_frame = convert_pil_img_to_ndarray(pil_img_obj)

            person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
            person_data_list.append(person_data)
        # End for (top, right, bottom, left), name in zip(face_location_list, face_names)

        # _1frameに対して1回
        if args_dict["headless"] == False:
            frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
            """DEBUG"""
            # frame_imshow_for_debug(resized_frame)
            # frame_datas_array.append(frame_datas)
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
    return modified_frame_list

    """機能停止
    # yield出力ブロック ===================================
    ## パイプ出力機構も含む
    ## TODO: frame_datas_arrayから値を取り出す処理に変えること
    if not frame_datas == None:
        if output_frame_data == True:  ## pipe出力時
            # frame_datas['stream'] = resized_frame
            # yield frame_datas
            pass
        elif output_frame_data == False:  ## 通常使用時
            frame_datas = {'name': name, 'pict':filename,  'date':date, 'img':resized_frame, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
            frame_datas_array.append(frame_datas)
            yield frame_datas_array
            # sys.stdout.buffer.write(frame_datas['stream'])  ## 'stream'を出力する
            # print(type(resized_frame))  ## <class 'numpy.ndarray'>
            # print(type(frame_datas['stream']))  ## <class 'numpy.ndarray'>

            # cv2.imshow('FACE01', frame_datas['stream'])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    # =====================================================

    # パイプ使用時の必要情報を表示 ========
    if print_property==True:
        print('fps: ', fps)
        print('frame shape: ', resized_frame.shape)  ## (450, 800, 3)
        print('dtype: ', resized_frame.dtype)  ## uint8
        print('frame size: ', resized_frame.size) ## 1080000←450*800*3
        exit()
    # =====================================
    """


frame_generator_obj = video_capture.frame_generator(args_dict)
def main_process():
    frame_datas_array = frame_pre_processing(args_dict, frame_generator_obj.__next__())
    face_encodings, frame_datas_array = face_encoding_process(args_dict, frame_datas_array)
    frame_datas_array = frame_post_processing(args_dict, face_encodings, frame_datas_array)
    yield frame_datas_array

# main =================================================================
if __name__ == '__main__':
    import cProfile as pr
    exec_times: int = 10000
    
    profile_HANDLING_FRAME_TIME: float = 0.0
    profile_HANDLING_FRAME_TIME_FRONT: float = 0.0
    profile_HANDLING_FRAME_TIME_REAR: float = 0.0

    # PySimpleGUIレイアウト
    if args_dict["headless"] == False:
        layout = [
            [sg.Image(filename='', key='display', pad=(0,0))],
            [sg.Button('終了', key='terminate', pad=(0,10))]
        ]
        window = sg.Window(
            'FACE01 プロファイリング利用例', layout, alpha_channel = 1, margins=(10, 10),
            location=(150,130), modal = True
        )

    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()

    def profile(exec_times):
        while True:
            frame_datas_array = main_process().__next__()
            if StopIteration == frame_datas_array:
                logging.info("StopIterationです")
                break
            exec_times = exec_times - 1
            if  exec_times <= 0:
                break
            else:
                print(f'exec_times: {exec_times}')
                if args_dict["headless"] == False:
                    event, _ = window.read(timeout = 1)
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
                        if args_dict["headless"] == False:
                            imgbytes = cv2.imencode(".png", img)[1].tobytes()
                            window["display"].update(data = imgbytes)
                        # del person_data_list
                # del frame_datas_array
            if args_dict["headless"] == False:
                if event =='terminate':
                    break
            # frame_datas_array_copy = frame_datas_array
            # del frame_datas_array
            # yield frame_datas_array_copy
        if args_dict["headless"] == False:
            window.close()
        print('プロファイリングを終了します')
        
        profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
        profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
        print(f'profile()関数の処理時間合計: {round(profile_HANDLING_FRAME_TIME , 3)}[秒]')

    pr.run('profile(exec_times)', 'restats')
    """# マルチプロセス化で使いやすいように別名をつける
    task = partial(frame_processing, args_dict)
    # with ProcessPoolExecutor() as executor:
    with ThreadPoolExecutor() as executor:
        while True:
            resized_frame = video_capture.video_capture(args_dict).__next__()
            if len(resized_frame) > 0:
                future = executor.submit(task, resized_frame)
                yield future.result()
                # RuntimeError
                # Error while calling cudaGetDevice(&the_device_id) in file /tmp/pip-install-983gqknr/dlib_66282e4ffadf4aa6965801c6f7ff7671/dlib/cuda/gpu_data.cpp:204.
                # code: 3,
                # reason: initialization error
            else:
                break
    """
    # while True:
    # for resized_frame in video_capture.frame_generator(args_dict, vcap, set_area, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER):
    # if resized_frame_obj.__name__  == 'frame_generator':
    # if len(resized_frame) > 0:
        # for resized_frame in resized_frame_obj:
    # for resized_frame in video_capture.frame_generator(args_dict):
    """DEBUG"""
    # frame_imshow_for_debug(frame_datas_array)
    # face_encodings, frame_datas_array = face_encoding_process(args_dict, frame_datas_array)
    """DEBUG"""
    # frame_imshow_for_debug(frame_datas_array)
    # frame_datas_array = frame_post_processing(args_dict, face_encodings, frame_datas_array)
    """DEBUG"""
    # frame_imshow_for_debug(frame_datas_array)

    # frame_datas_array_copy = frame_datas_array
    # del frame_datas_array
    # yield frame_datas_array_copy
    # else:
    #     break
    # 入力を終了する


