from configparser import ConfigParser
from os import chdir
from os.path import dirname, exists
from sys import exit, version, version_info
from time import perf_counter
from traceback import format_exc

import cv2
from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory

from face01lib.api import Dlib_api
from face01lib.video_capture import VidCap
from face01lib.Core import Core
from face01lib.Initialize import Initialize
from face01lib.logger import Logger

GLOBAL_MEMORY = {
# 半透明値,
'alpha' : 0.3,
'number_of_crops' : 0
}
name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir)
def configure():
    kaoninshoDir: str = dir
    chdir(kaoninshoDir)
    priset_face_imagesDir: str = f'{dirname(__file__)}/priset_face_images/'

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
            'calculate_time' : conf.getboolean('DEFAULT','calculate_time'),
            'set_width' : int(conf.get('DEFAULT','set_width')),
            'default_face_image_draw' : conf.getboolean('DEFAULT', 'default_face_image_draw'),
            'show_overlay' : conf.getboolean('DEFAULT', 'show_overlay'),
            'show_percentage' : conf.getboolean('DEFAULT', 'show_percentage'),
            'crop_face_image' : conf.getboolean('DEFAULT', 'crop_face_image'),
            'show_name' : conf.getboolean('DEFAULT', 'show_name'),
            'draw_telop_and_logo' : conf.getboolean('DEFAULT', 'draw_telop_and_logo'),
            'use_pipe' : conf.getboolean('DEFAULT','use_pipe'),
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
        logger.warning("以下のエラーをシステム管理者様へお伝えください")
        logger.warning("-" * 20)
        logger.warning(format_exc(limit=None, chain=True))
        logger.warning("-" * 20)
        logger.warning("終了します")
        exit(0)

conf_dict = configure()

# initialize
args_dict =  Initialize().initialize(conf_dict)

"""CHECK SYSTEM INFORMATION"""
def system_check(args_dict):
    # lock
    with open("SystemCheckLock", "w") as f:
        f.write('')
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
        if Dlib_api().dlib.cuda.get_num_devices() == 0:
            logger.warning("CUDAが有効なデバイスが見つかりません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] cuda devices: {Dlib_api().dlib.cuda.get_num_devices()}")

        # Dlib build check: CUDA
        logger.info("- Dlib build check: CUDA")
        if Dlib_api().dlib.DLIB_USE_CUDA == False:
            logger.warning("dlibビルド時にCUDAが有効化されていません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] DLIB_USE_CUDA: True")

        # Dlib build check: BLAS
        logger.info("- Dlib build check: BLAS, LAPACK")
        if Dlib_api().dlib.DLIB_USE_BLAS == False or Dlib_api().dlib.DLIB_USE_LAPACK == False:
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
if not exists("SystemCheckLock"):
    system_check(args_dict)

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

frame_generator_obj = VidCap().frame_generator(args_dict)
# @profile()
def main_process():
    frame_datas_array = Core().frame_pre_processing(logger, args_dict, frame_generator_obj.__next__())
    face_encodings, frame_datas_array = Core().face_encoding_process(logger, args_dict, frame_datas_array)
    frame_datas_array = Core().frame_post_processing(logger, args_dict, face_encodings, frame_datas_array, GLOBAL_MEMORY)
    yield frame_datas_array

# main =================================================================
if __name__ == '__main__':
    import cProfile as pr

    import PySimpleGUI as sg

    exec_times: int = 100
    
    profile_HANDLING_FRAME_TIME: float = 0.0
    profile_HANDLING_FRAME_TIME_FRONT: float = 0.0
    profile_HANDLING_FRAME_TIME_REAR: float = 0.0

    sg.theme('LightGray')
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
        event = ''
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
