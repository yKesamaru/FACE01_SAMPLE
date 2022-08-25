
"""DEBUG: MEMORY LEAK
from face01lib.memory_leak import Memory_leak
Memory_leak_obj = Memory_leak()
line_or_traceback = 'line'  # 'line' or 'traceback'
Memory_leak_obj.memory_leak_analyze_start(line_or_traceback)
"""
import gc
import cProfile as pr
import time

import cv2
import PySimpleGUI as sg

from face01lib.video_capture import VidCap

VidCap_obj = VidCap()
from face01lib.Core import Core

Core_obj = Core()
import inspect
from os.path import dirname
from sys import exit
from traceback import format_exc
import sys
from memory_profiler import profile  # @profile()

import FACE01 as fg
fg_main_process_obj = fg.main_process()

from face01lib.Calc import Cal
from face01lib.logger import Logger

name = __name__
dir = dirname(__file__)
if fg.args_dict[output_debug_log] == True:
    logger = Logger().logger(name, dir, 'debug')
else:
    logger = Logger().logger(name, dir, None)


Cal().cal_specify_date(logger)

"""DEBUG
Set the number of playback frames"""
exec_times: int = 30
ALL_FRAME = exec_times

# PySimpleGUI layout
sg.theme('LightGray')
if fg.args_dict["headless"] == False:
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
        location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
    )

@profile()
def common_main(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    event = ''
    while True:
        try:
            # frame_datas_array = fg_main_process_obj.__next__()
            frame_datas_array = fg.main_process().__next__()
            """DEBUG"""
            logger.debug(f"frame_datas_array: {sys.getsizeof(frame_datas_array) / 1024 / 1048}MiB")
            logger.debug(inspect.currentframe().f_back.f_code.co_filename)
            logger.debug(inspect.currentframe().f_back.f_lineno)

        except Exception as e:
            print(format_exc(limit=None, chain=True))
            print(e)
            exit(0)
        exec_times = exec_times - 1
        if  exec_times <= 0:
            break
        else:
            print(f'exec_times: {exec_times}')

            """DEBUG"""
            logger.debug(f"args_dict: {sys.getsizeof(fg.args_dict) / 1024 / 1048}MiB")
            logger.debug(inspect.currentframe().f_back.f_code.co_filename)
            logger.debug(inspect.currentframe().f_back.f_lineno)

            if fg.args_dict["headless"] == False:
                event, _ = window.read(timeout = 1)
                if event == sg.WIN_CLOSED:
                    print("The window was closed manually")
                    break
            for frame_datas in frame_datas_array:
                """DEBUG
                logger.warning(f'frame_datas_array size: {len(frame_datas_array)}')
                logger.warning(inspect.currentframe().f_back.f_code.co_filename)
                logger.warning(inspect.currentframe().f_back.f_lineno)
                """

                if "face_location_list" in frame_datas:
                    img, face_location_list, overlay, person_data_list = \
                        frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
                    for person_data in person_data_list:
                        if len(person_data) == 0:
                            continue
                        name, pict, date,  location, percentage_and_symbol = \
                            person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                        # ELE: Equally Likely Events
                        if name != 'Unknown':
                            spoof_or_real, score, ELE = Core_obj.return_anti_spoof(frame_datas['img'], person_data["location"])
                            
                            if fg.args_dict["anti_spoof"] is True:
                                if ELE is False:
                                    pass
                                    # print(
                                    #     name, "\n",
                                    #     "\t", "Anti spoof\t\t", spoof_or_real, "\n",
                                    #     "\t", "Anti spoof score\t", round(score * 100, 2), "%\n",
                                    #     "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                    #     "\t", "coordinate\t\t", location, "\n",
                                    #     "\t", "time\t\t\t", date, "\n",
                                    #     "\t", "output\t\t\t", pict, "\n",
                                    #     "-------\n"
                                    # )
                            # else:
                            #     if ELE is False:
                            #         print(
                            #             name, "\n",
                            #             "\t", "similarity\t\t", percentage_and_symbol, "\n",
                            #             "\t", "coordinate\t\t", location, "\n",
                            #             "\t", "time\t\t\t", date, "\n",
                            #             "\t", "output\t\t\t", pict, "\n",
                            #             "-------\n"
                            #         )
                    if fg.args_dict["headless"] == False:
                        imgbytes = cv2.imencode(".png", img)[1].tobytes()
                        window["display"].update(data = imgbytes)
            # メモリ解放
            del frame_datas_array
            gc.collect()

        if fg.args_dict["headless"] == False:
            if event =='terminate':
                break
    if fg.args_dict["headless"] == False:
        window.close()
    
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'Predetermined number of frames: {ALL_FRAME}')
    print(f'Number of frames processed: {ALL_FRAME - exec_times}')
    print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
    print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / (ALL_FRAME - exec_times), 3)}[seconds]')
pr.run('common_main(exec_times)', 'restats')


"""If you want to extract only face coordinates"""
next_frame_gen_obj = VidCap_obj.frame_generator(fg.args_dict)
# @profile()
def extract_face_locations(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    i: int = 0
    for i in range(exec_times):
        i += 1
        if i >= exec_times:
            break
        next_frame = next_frame_gen_obj.__next__()
        """DEBUG
        # fg.frame_imshow_for_debug(next_frame)
        print(f"fg.args_dict.__sizeof__(): {fg.args_dict.__sizeof__()}MB")
        """
        frame_datas_array = Core_obj.frame_pre_processing(fg.logger, fg.args_dict,next_frame)
        for frame_datas in frame_datas_array:
            for face_location in frame_datas["face_location_list"]:
                print(face_location)
    
    print('Finish profiling')
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'Predetermined number of frames: {ALL_FRAME}')
    print(f'Number of frames processed: {i}')
    print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
    print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / i, 3)}[seconds]')
# pr.run('extract_face_locations(exec_times)', 'restats')

"""DEBUG: MEMORY LEAK
Memory_leak_obj.memory_leak_analyze_stop(line_or_traceback)
"""
# from pympler import summary, muppy
# all_objects = muppy.get_objects()
# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)