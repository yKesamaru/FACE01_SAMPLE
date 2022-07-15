import cProfile as pr
import FACE01 as fg
import PySimpleGUI as sg
import cv2
import time
import face01lib.video_capture as vc
from memory_profiler import profile  # @profile()
from sys import exit

"""DEBUG
再生フレーム数をセット"""
exec_times: int = 50
ALL_FRAME = exec_times

# PySimpleGUIレイアウト
sg.theme('LightGray')
if fg.args_dict["headless"] == False:
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('終了', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
        location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
    )

# @profile()
def common_main(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    while True:
        try:
            frame_datas_array = fg.main_process().__next__()
        except Exception as e:
            print(e)
            exit(0)
        exec_times = exec_times - 1
        if  exec_times <= 0:
            break
        else:
            print(f'exec_times: {exec_times}')
            if fg.args_dict["headless"] == False:
                event, _ = window.read(timeout = 1)
                if event == sg.WIN_CLOSED:
                    print("ウィンドウが手動で閉じられました")
                    break
            for frame_datas in frame_datas_array:
                if "face_location_list" in frame_datas:
                    img, face_location_list, overlay, person_data_list = \
                        frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
                    for person_data in person_data_list:
                        name, pict, date,  location, percentage_and_symbol = \
                            person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                        if not name == 'Unknown':
                            # print("メモリプロファイリング中")
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
                            """DEBUG
                            print(f"fg.args_dict.__sizeof__(): {fg.args_dict.__sizeof__()}MB")
                            """
                    if fg.args_dict["headless"] == False:
                        imgbytes = cv2.imencode(".png", img)[1].tobytes()
                        window["display"].update(data = imgbytes)
        if fg.args_dict["headless"] == False:
            if event =='terminate':
                break
    if fg.args_dict["headless"] == False:
        window.close()
    print('プロファイリングを終了します')
    
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'予定フレーム数: {ALL_FRAME}')
    print(f'処理フレーム数: {ALL_FRAME - exec_times}')
    print(f'profile()関数の処理時間合計: {round(profile_HANDLING_FRAME_TIME , 3)}[秒]')
    print(f'1フレーム: {round(profile_HANDLING_FRAME_TIME / (ALL_FRAME - exec_times), 3)}[秒]')
# pr.run('common_main(exec_times)', 'restats')


"""顔座標のみ抽出したい場合"""
next_frame_gen_obj = vc.frame_generator(fg.args_dict)
# @profile()
def extract_face_locations(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    for i in range(exec_times):
        i += 1
        if i >= exec_times:
            break
        next_frame = next_frame_gen_obj.__next__()
        """DEBUG
        # fg.frame_imshow_for_debug(next_frame)
        print(f"fg.args_dict.__sizeof__(): {fg.args_dict.__sizeof__()}MB")
        """
        frame_datas_array = fg.frame_pre_processing(fg.args_dict,next_frame)
        for frame_datas in frame_datas_array:
            for face_location in frame_datas["face_location_list"]:
                print(face_location)
    
    print('プロファイリングを終了します')
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'予定フレーム数: {ALL_FRAME}')
    print(f'処理フレーム数: {i}')
    print(f'profile()関数の処理時間合計: {round(profile_HANDLING_FRAME_TIME , 3)}[秒]')
    print(f'1フレーム: {round(profile_HANDLING_FRAME_TIME / i, 3)}[秒]')
# extract_face_locations(exec_times)
pr.run('extract_face_locations(exec_times)', 'restats')