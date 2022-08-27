import inspect
from configparser import ConfigParser
from os import chdir
from os.path import dirname, exists
from sys import exit, version, version_info, getsizeof
from time import perf_counter
from traceback import format_exc

"""DEBUG: MEMORY LEAK"""
from face01lib.memory_leak import Memory_leak
m = Memory_leak(limit=7, key_type='traceback', nframe=20)
# m = Memory_leak(limit=7, key_type='lineno')
m.memory_leak_analyze_start()


import gc
import cv2
from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory

from memory_profiler import profile  # @profile()
from face01lib.api import Dlib_api

Dlib_api_obj = Dlib_api()
from face01lib.Core import Core

Core_obj = Core()
from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.video_capture import VidCap

GLOBAL_MEMORY = {
# 半透明値,
'alpha' : 0.3,
'number_of_crops' : 0
}

name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir, 'debug')

# @profile()
def configure():
    kaoninshoDir: str = dir
    chdir(kaoninshoDir)
    priset_face_imagesDir: str = f'{dirname(__file__)}/priset_face_images/'

    try:
        conf = ConfigParser()
        conf.read('config.ini', 'utf-8')
        # dict作成
        conf_dict = {
            'model' : conf.get('DEFAULT','model'),
            'headless' : conf.getboolean('MAIN','headless'),
            'anti_spoof' : conf.getboolean('MAIN','anti_spoof'),
            'output_debug_log' : conf.getboolean('MAIN','output_debug_log'),
            'set_width' : int(conf.get('SPEED_OR_PRECISE','set_width')),
            'similar_percentage' : float(conf.get('SPEED_OR_PRECISE','similar_percentage')),
            'jitters' : int(conf.get('SPEED_OR_PRECISE','jitters')),
            'priset_face_images_jitters' : int(conf.get('SPEED_OR_PRECISE','priset_face_images_jitters')),
            'priset_face_imagesDir' :priset_face_imagesDir,
            'upsampling' : int(conf.get('SPEED_OR_PRECISE','upsampling')),
            'mode' : conf.get('SPEED_OR_PRECISE','mode'),
            'frame_skip' : int(conf.get('SPEED_OR_PRECISE','frame_skip')),
            'number_of_people' : int(conf.get('SPEED_OR_PRECISE','number_of_people')),
            'use_pipe' : conf.getboolean('dlib','use_pipe'),
            'model_selection' : int(conf.get('dlib','model_selection')),
            'min_detection_confidence' : float(conf.get('dlib','min_detection_confidence')),
            'person_frame_face_encoding' : conf.getboolean('dlib','person_frame_face_encoding'),
            'set_area' : conf.get('INPUT','set_area'),
            'movie' : conf.get('INPUT','movie'),
            'user': conf.get('Authentication','user'),
            'passwd': conf.get('Authentication','passwd'),
            'rectangle' : conf.getboolean('DRAW_INFOMATION','rectangle'),
            'target_rectangle' : conf.getboolean('DRAW_INFOMATION','target_rectangle'),
            'draw_telop_and_logo' : conf.getboolean('DRAW_INFOMATION', 'draw_telop_and_logo'),
            'default_face_image_draw' : conf.getboolean('DRAW_INFOMATION', 'default_face_image_draw'),
            'show_overlay' : conf.getboolean('DRAW_INFOMATION', 'show_overlay'),
            'show_percentage' : conf.getboolean('DRAW_INFOMATION', 'show_percentage'),
            'show_name' : conf.getboolean('DRAW_INFOMATION', 'show_name'),
            'crop_face_image' : conf.getboolean('SAVE_FACE_IMAGE', 'crop_face_image'),
            'frequency_crop_image' : int(conf.get('SAVE_FACE_IMAGE','frequency_crop_image')),
            'crop_with_multithreading' : conf.getboolean('SAVE_FACE_IMAGE','crop_with_multithreading'),
            'Python_version': conf.get('system_check','Python_version'),
            'cpu_freq': conf.get('system_check','cpu_freq'),
            'cpu_count': conf.get('system_check','cpu_count'),
            'memory': conf.get('system_check','memory'),
            'gpu_check' : conf.getboolean('system_check','gpu_check'),
            'calculate_time' : conf.getboolean('DEBUG','calculate_time'),
            'show_video' : conf.getboolean('Scheduled_to_be_abolished','show_video'),
            'kaoninshoDir' :kaoninshoDir,
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
# @profile()
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
        if Dlib_api_obj.dlib.cuda.get_num_devices() == 0:
            logger.warning("CUDAが有効なデバイスが見つかりません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] cuda devices: {Dlib_api_obj.dlib.cuda.get_num_devices()}")

        # Dlib build check: CUDA
        logger.info("- Dlib build check: CUDA")
        if Dlib_api_obj.dlib.DLIB_USE_CUDA == False:
            logger.warning("dlibビルド時にCUDAが有効化されていません")
            logger.warning("終了します")
            exit(0)
        else:
            logger.info(f"  [OK] DLIB_USE_CUDA: True")

        # Dlib build check: BLAS
        logger.info("- Dlib build check: BLAS, LAPACK")
        if Dlib_api_obj.dlib.DLIB_USE_BLAS == False or Dlib_api_obj.dlib.DLIB_USE_LAPACK == False:
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
    try:

        frame_datas_array = Core_obj.frame_pre_processing(logger, args_dict, frame_generator_obj.__next__())
        
        """DEBUG"""
        logger.debug(inspect.currentframe().f_back.f_code.co_filename)
        logger.debug(inspect.currentframe().f_back.f_lineno)
        logger.debug(f'frame_datas_array size: {len(frame_datas_array), getsizeof(frame_datas_array)}')
        logger.debug(inspect.currentframe().f_back.f_code.co_filename)
        logger.debug(inspect.currentframe().f_back.f_lineno)
        logger.debug(f'args_dict size: {len(args_dict), getsizeof(args_dict)}')
        
        face_encodings, frame_datas_array = Core_obj.face_encoding_process(logger, args_dict, frame_datas_array)
        frame_datas_array = Core_obj.frame_post_processing(logger, args_dict, face_encodings, frame_datas_array, GLOBAL_MEMORY)
        
        yield frame_datas_array

        # メモリ解放
        del frame_datas_array
        gc.collect()

    except StopIteration:
        logger.warning("DATA RECEPTION HAS ENDED")
        exit(0)
    except Exception as e:
        logger.warning("ERROR OCURRED")
        logger.warning("-" * 20)
        logger.warning(f"ERROR TYPE: {e}")
        logger.warning(format_exc(limit=None, chain=True))
        logger.warning("-" * 20)
        exit(0)

# main =================================================================
if __name__ == '__main__':
    # import cProfile as pr
    import time
    import traceback

    import PySimpleGUI as sg

    # from face01lib.Core import Core
    # Core_obj = Core()
    
    profile_HANDLING_FRAME_TIME: float = 0.0
    profile_HANDLING_FRAME_TIME_FRONT: float = 0.0
    profile_HANDLING_FRAME_TIME_REAR: float = 0.0

    """DEBUG
    Set the number of playback frames"""
    exec_times: int = 50
    ALL_FRAME = exec_times

    # PySimpleGUI layout
    sg.theme('LightGray')
    if args_dict["headless"] == False:
        layout = [
            [sg.Image(filename='', key='display', pad=(0,0))],
            [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
        ]
        window = sg.Window(
            'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
            location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
        )

    # @profile()
    def common_main(exec_times):
        profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
        event = ''
        while True:
            try:
                frame_datas_array = main_process().__next__()
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                exit(0)
            exec_times = exec_times - 1
            if  exec_times <= 0:
                break
            else:
                print(f'exec_times: {exec_times}')
                if args_dict["headless"] == False:
                    event, _ = window.read(timeout = 1)
                    if event == sg.WIN_CLOSED:
                        print("The window was closed manually")
                        break
                for frame_datas in frame_datas_array:
                    if "face_location_list" in frame_datas:
                        img = frame_datas['img']
                        person_data_list = frame_datas['person_data_list']
                        
                        for person_data in person_data_list:
                            if person_data == {}:
                                continue

                            name = person_data['name']
                            pict = person_data['pict']
                            date = person_data['date']
                            location = person_data['location']
                            percentage_and_symbol = person_data['percentage_and_symbol']

                            spoof_or_real, score, ELE = \
                                Core_obj.return_anti_spoof(img, location)
                            # ELE: Equally Likely Events
                            if not name == 'Unknown':
                                # Bug fix
                                if args_dict["anti_spoof"] == True:
                                    if ELE == False and spoof_or_real == 'real':
                                        print(
                                            name, "\n",
                                            "\t", "Anti spoof\t\t", spoof_or_real, "\n",
                                            "\t", "Anti spoof score\t", round(score * 100, 2), "%\n",
                                            "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                            "\t", "coordinate\t\t", location, "\n",
                                            "\t", "time\t\t\t", date, "\n",
                                            "\t", "output\t\t\t", pict, "\n",
                                            "-------\n"
                                        )
                                else:
                                    if ELE == False:
                                        print(
                                            name, "\n",
                                            "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                            "\t", "coordinate\t\t", location, "\n",
                                            "\t", "time\t\t\t", date, "\n",
                                            "\t", "output\t\t\t", pict, "\n",
                                            "-------\n"
                                        )
                        if args_dict["headless"] == False:
                            imgbytes = cv2.imencode(".png", img)[1].tobytes()
                            window["display"].update(data = imgbytes)
            if args_dict["headless"] == False:
                if event =='terminate':
                    break
        
        # メモリ解放
        del frame_datas_array
        gc.collect()
        
        if args_dict["headless"] == False:
            window.close()
        
        profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
        profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
        print(f'Predetermined number of frames: {ALL_FRAME}')
        print(f'Number of frames processed: {ALL_FRAME - exec_times}')
        print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
        print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / (ALL_FRAME - exec_times), 3)}[seconds]')
    # pr.run('common_main(exec_times)', 'restats')


    common_main(exec_times)

"""DEBUG: MEMORY LEAK"""
m.memory_leak_analyze_stop()


# from pympler import summary, muppy
# all_objects = muppy.get_objects()
# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)