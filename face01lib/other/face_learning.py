import heapq
import os

import PySimpleGUI as sg

# import load_priset_image
# from load_priset_image import load_priset_image
from . import load_priset_image


def face_learning(
    kaoninshoDir,
    priset_face_imagesDir,
    face_learning_frame_counter,
    tolerance,
    face_distances,
    p,
    name,
    how_many_face_learning_images,
    face_learning_filename_counter,
    imgCroped,
    percentage_and_symbol,

):
    if face_learning_frame_counter % 60 ==0:
    # if face_learning_frame_counter % 10 ==0:
        if p >= 0.28 and p <= tolerance:  ## 0.28→99.5%, 0.42→99.0000000046%, 0.37→99.207%
            # face_learningファイルをいくつまで作成するか定義する(_face_learning_2)
            # if not os.path.exists(priset_face_imagesDir + name + '_face_learning_' + str(face_learning_filename_counter) + '.png'):
            face_learning_file = priset_face_imagesDir + name + '_face_learning_' + str(how_many_face_learning_images) + '.png'
            if not os.path.exists(face_learning_file):  ## もし無かったら。
                face_learning_outputDir = priset_face_imagesDir + name + '_face_learning_' + str(face_learning_filename_counter) + '.png'
                # もし同じ名前でface_learning_file_counterが\dだったらカウンターを戻して以降の処理を飛ばす
                if face_learning_filename_counter > int(how_many_face_learning_images):
                    face_learning_filename_counter = 1
                    return
                else:
                    tempo_file = 'output/temp_' + name + '.png'  ## PySimpleGUIではpng形式が表示できる jpgは不可
                    imgCroped.save(tempo_file)
                    # priset_face_imagesDirにname_default.pngがあるか確認
                    if os.path.exists(priset_face_imagesDir + name + '_default.png'):
                        default_face_file_fullpath = priset_face_imagesDir + name + '_default.png'
                        face_learning_window_layout = [
                                    [sg.Text('【光学歪み較正】')],
                                    [sg.Text('この人は'), sg.Text(name), sg.Text('さんですか？')],
                                    # [sg.Image(tempo_file, size=(100,100))],
                                    [sg.Image(tempo_file, pad=(25,25)), sg.Image(default_face_file_fullpath, pad=(25,25))],
                                    [sg.Text('参考顔距離: '), sg.Text(round(p, 3)), sg.Text(percentage_and_symbol)],
                                    [sg.Button(button_text='はい'), sg.Button(button_text='いいえ', button_color='red')]
                                ]
                        face_learning_window_first = sg.Window('光学補正 第1候補', face_learning_window_layout)
                        while True:             
                            event, values = face_learning_window_first.read()
                            if event in (sg.WIN_CLOSED, 'はい', 'いいえ'):
                                return
                        face_learning_window_first.close()
                        os.remove(tempo_file)
                        if event=='はい':
                            imgCroped.save(face_learning_outputDir)
                            face_learning_filename_counter += 1  ## str型にする直前にカウンターを加算する
                            # 配列を読み込み直す必要がある
                            known_face_encodings, known_face_names = load_priset_image.load_priset_image(kaoninshoDir, priset_face_imagesDir)
                            flag = True
                        elif event=='いいえ':
                            # もし同じ名前でface_learning_file_counterが\dだったらカウンターを戻して以降の処理を飛ばす
                            if face_learning_filename_counter > how_many_face_learning_images:
                                face_learning_filename_counter = 1
                            else:
                                second_match = heapq.nsmallest(3, face_distances)[1]
                                second_index = face_distances.tolist().index(second_match)
                                second_match_name = known_face_names[second_index]
                                second_match_name_only, _ = second_match_name.split('_',1)
                                if os.path.exists(priset_face_imagesDir + second_match_name_only + '_default.png'):
                                    default_face_file_fullpath = priset_face_imagesDir + second_match_name_only + '_default.png'
                                
                                    tempo_file_2 = 'output/temp_' + second_match_name_only + '.png'  ## PySimpleGUIではpng形式が表示できる jpgは不可
                                    imgCroped.save(tempo_file_2)
                                    face_learning_window_layout_2 = [
                                            [sg.Text('第2候補です')],
                                            [sg.Text('この人は'), sg.Text(second_match_name_only), sg.Text('さんですか？')],
                                            [sg.Image(tempo_file_2, pad=(25,25)), sg.Image(default_face_file_fullpath, pad=(25,25))],
                                            [sg.Text('参考顔距離: '), sg.Text(round(p, 3)), sg.Text(percentage_and_symbol)],
                                            [sg.Button(button_text='はい'), sg.Button(button_text='いいえ', button_color='red')]
                                        ]
                                    face_learning_window = sg.Window('光学補正 第2候補', face_learning_window_layout_2)
                                    while True:             
                                        event, values = face_learning_window.read()
                                        if event in (sg.WIN_CLOSED, 'はい', 'いいえ'):
                                            return
                                    face_learning_window.close()
                                    os.remove(tempo_file_2)
                                    if event=='はい':
                                        face_learning_filename_counter += 1  ## str型にする直前にカウンターを加算する
                                        face_learning_outputDir = priset_face_imagesDir + second_match_name_only + '_face_learning_' + str(face_learning_filename_counter) + '.png'
                                        imgCroped.save(face_learning_outputDir)
                                        # 配列を読み込み直す必要がある
                                        known_face_encodings, known_face_names = load_priset_image.load_priset_image(kaoninshoDir, priset_face_imagesDir)
                                        flag = True
                                    elif event=='いいえ':
                                        # もし同じ名前でface_learning_file_counterが\dだったらカウンターを戻して以降の処理を飛ばす
                                        if face_learning_filename_counter > how_many_face_learning_images:
                                            face_learning_filename_counter = 1
                                        else:
                                            face_learning_distance_matchs = heapq.nsmallest(10, face_distances)
                                            face_learning_match_index = [face_distances.tolist().index(x) for x in face_learning_distance_matchs]
                                            face_learning_names = [known_face_names[x] for x in face_learning_match_index]
                                            # face_learning_names = [x.split('_', maxsplit=1) for x in face_learning_names]
                                            face_learning_window_layout_3 = [
                                                [sg.Listbox(values=face_learning_names, size=(30, 10), key='listbox')],
                                                [sg.Button(button_text='選択'), sg.Button(button_text='この中にはない', button_color='red')]
                                                ]
                                            face_learning_window = sg.Window('光学補正', face_learning_window_layout_3)
                                            while True:             
                                                event, values = face_learning_window.read()
                                                if event in (sg.WIN_CLOSED, '選択', 'この中にはない'):
                                                    return
                                            face_learning_window.close()
                                            if not values['listbox'] == []:
                                                selected_name = values['listbox'][0].split('_', maxsplit=1)[0]
                                                face_learning_filename_counter += 1  ## str型にする直前にカウンターを加算する
                                                face_learning_outputDir = priset_face_imagesDir + selected_name + '_face_learning_' + str(face_learning_filename_counter) + '.png'
                                                imgCroped.save(face_learning_outputDir)
                                                # 配列を読み込み直す必要がある
                                                known_face_encodings, known_face_names = load_priset_image.load_priset_image(kaoninshoDir, priset_face_imagesDir)
                                                flag = True
                                            else:
                                                file_browse_window_layout = [
                                                    [sg.Text('正しいファイルを選んで下さい')],
                                                    [sg.InputText(key='input_browse_file'), 
                                                        sg.FileBrowse(
                                                            key='file_browse',
                                                            target='input_browse_file', 
                                                            file_types=(("ALL Files", "*default.png"), ), 
                                                            initial_folder=priset_face_imagesDir
                                                            )], 
                                                    [sg.Submit(), sg.Cancel()]
                                                ]
                                                window_from_file_browse = sg.Window('正しい顔画像ファイルの選択', file_browse_window_layout)
                                                event, values = window_from_file_browse.read()
                                                window_from_file_browse.close()
                                                # print(values['file_browse']);exit()
                                                if values['file_browse']:
                                                    # もし同じ名前でface_learning_file_counterが\dだったらカウンターを戻して以降の処理を飛ばす
                                                    if face_learning_filename_counter > how_many_face_learning_images:
                                                        face_learning_filename_counter = 1
                                                    face_learning_file = os.path.basename(values['file_browse'])
                                                    face_learning_file, _ = face_learning_file.split('_', maxsplit=1)
                                                    face_learning_outputDir = priset_face_imagesDir + face_learning_file + '_face_learning_' + str(face_learning_filename_counter) + '.png'
                                                    imgCroped.save(face_learning_outputDir)
                                                    face_learning_filename_counter += 1  ## str型にする直前にカウンターを加算する
                                                    # 配列を読み込み直す必要がある
                                                    known_face_encodings, known_face_names = load_priset_image.load_priset_image(kaoninshoDir, priset_face_imagesDir)
                                                    flag = True
                                                    # print(known_face_names);exit()
                                                    pass

                                        # # もし同じ名前でface_learning_file_counterが\dだったらカウンターを戻して以降の処理を飛ばす
                                        # if face_learning_filename_counter > how_many_face_learning_images:
                                        #     face_learning_filename_counter = 1
                                        # else:
                                        #     third_match = heapq.nsmallest(3, face_distances)[2]
                                        #     third_index = face_distances.tolist().index(third_match)
                                        #     third_match_name = known_face_names[third_index]
                                        #     third_match_name_only, _ = third_match_name.split('_',1)
                                        #     if os.path.exists(priset_face_imagesDir + third_match_name_only + '_default.png'):
                                        #         default_face_file_fullpath = priset_face_imagesDir + third_match_name_only + '_default.png'
                                        #         tempo_file_2 = 'output/temp_' + third_match_name_only + '.png'  ## PySimpleGUIではpng形式が表示できる jpgは不可
                                        #         imgCroped.save(tempo_file_2)
                                        #         face_learning_window_layout_2 = [
                                        #                 [sg.Text('第3候補です')],
                                        #                 [sg.Text('この人は'), sg.Text(third_match_name_only), sg.Text('さんですか？')],
                                        #                 [sg.Image(tempo_file_2, pad=(25,25)), sg.Image(default_face_file_fullpath, pad=(25,25))],
                                        #                 [sg.Text('参考顔距離: '), sg.Text(round(p, 3)), sg.Text(percentage_and_symbol)],
                                        #                 [sg.Button(button_text='はい'), sg.Button(button_text='いいえ', button_color='red')]
                                        #             ]
                                        #         face_learning_window = sg.Window('光学補正 第3候補', face_learning_window_layout_2)
                                        #         while True:             
                                        #             event, values = face_learning_window.read()
                                        #             if event in (sg.WIN_CLOSED, 'はい', 'いいえ'):
                                        #                 return
                                        #         face_learning_window.close()
                                        #         os.remove(tempo_file_2)
                                        #         if event=='はい':
                                        #             face_learning_outputDir = priset_face_imagesDir + third_match_name_only + '_face_learning_' + str(face_learning_filename_counter) + '.png'
                                        #             imgCroped.save(face_learning_outputDir)
                                        #             face_learning_filename_counter += 1  ## str型にする直前にカウンターを加算する
                                        #             # 配列を読み込み直す必要がある
                                        #             known_face_encodings, known_face_names = load_priset_image.load_priset_image(kaoninshoDir, priset_face_imagesDir)
                                        #             flag = True
                                        #         elif event=='いいえ':
