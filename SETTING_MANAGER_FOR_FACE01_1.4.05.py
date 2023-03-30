import configparser
import os
from tkinter.constants import TOP

import PySimpleGUI as sg

# テーマ
sg.theme('Reddit')

# 変数初期設定
parameters={}
http_value=None
file_browse_value=None

# ウィンドウレイアウト -------------------
layout=[
    [sg.Image('images/setting_manager_header.png', pad=((0,0),(0,0)), expand_x=True)],  # header

    [sg.Column([
        [
            sg.Frame('プリセット顔画像読み込み設定',
            [
                [sg.Text('顔認識精度', tooltip='ゆらぎ値(jitters)。ランダムにスケールアップや色の修正をして平均値を返す。デフォルト値：100')], 
                [
                    sg.Radio('標準', key='jitters_priset_image_jitters100', tooltip='jitters=100', group_id='priset_face_images_jitters', default=True),
                    sg.Radio('粗雑(fast)',key='jitters_priset_image_jitters10', tooltip='jitters=10' , group_id='priset_face_images_jitters')
                ],

                [sg.Text('プリセット画像の再データ化')],
                [
                    sg.Radio('再データ化しない', key='not_remake_npKnown', tooltip='今あるnpKnown.npzを用いる', group_id='remake_npKnown', default=True, enable_events=True),
                    sg.Radio('再データ化する', key='do_remake_npKnown', tooltip='再作成に時間がかかる', group_id='remake_npKnown', enable_events=True),
                ],
                [sg.Text('', key='remove_npKnown')],
                
                [sg.Image('images/plant.png')],  # イラストの挿入
                [sg.Text('FACE01 GRAPHICS ver.1.2.9対応')]  # 対応バージョン表示

            ], vertical_alignment=TOP, font=('BIZ-UDGOTHICB.TTC', 9),
            ),

            sg.Frame('入力映像読み込み設定',
                [
                    [sg.Text('映像入力元')],
                    [
                        sg.Radio('test.mp4', key='test.mp4', group_id='input_moving_image', tooltip='test.mp4を映像入力元に指定する', default=True), 
                        sg.Radio('Webカメラ', key='web_camera', group_id='input_moving_image', tooltip='Webカメラなどを映像入力元に指定する'), 
                        sg.Radio('HLS', key='HLS', group_id='input_moving_image', tooltip='HLSを映像入力元に指定する', enable_events=True),
                        sg.Radio('その他', key='other_input', group_id='input_moving_image', tooltip='その他動画を入力元に指定する', enable_events=True)
                    ],
                    [sg.Text('', relief='sunken', key='filename', font=('BIZ-UDGOTHICB.TTC', 9), size=(50,2), enable_events=True)],

                    [sg.Text('顔検出解像度', tooltip='検出対象の顔面積を指定。標準指定では80x80pxで顔探索をする。微細指定では40x40px。デフォルト値：標準')], 
                    [
                        sg.Radio('標準', key='upsampling_input_moving_image_80', tooltip='80x80ピクセル以上の大きさの顔探索を行う。処理速度は普通。', group_id='upsampling_input_moving_image', default=True), 
                        sg.Radio('微細(slow)', key='upsampling_input_moving_image_40', tooltip='40x40以上の大きさの顔探索を行う。処理速度は遅い', group_id='upsampling_input_moving_image'), 
                    ],

                    [sg.Text('フレームドロップ', tooltip='指定した数値だけフレームドロップする。-1を指定すると自動で処理を行う。デフォルト値：-1')], 
                    [sg.Slider(orientation='h', key='flame_skip', range=(-1, 20), default_value=-1, resolution=1 ,enable_events=True)],
                    
                    [sg.Text('エリア指定', tooltip='入力された映像データの指定したエリアのみを処理し指定エリア外は破棄される')],
                    [sg.Combo(('エリア指定なし','左上','右上','左下','右下','中央'), key='area', default_value='エリア指定なし', size=(20,1), enable_events=True)],

                    [sg.Text('映像幅リサイズ', tooltip='入力された映像データを指定された映像幅へリサイズする')],
                    [sg.Slider(orientation='h', key='set_width', range=(500, 800), default_value=550, resolution=50, enable_events=True)],

                    # output cropped image files ----
                    [sg.Text('顔画像ファイル出力', tooltip='I/O速度の影響を受ける。デフォルト値：OFF')], 
                    [
                        sg.Radio('ON', key='crop_face_image_ON',  group_id='crop_face_image'), 
                        sg.Radio('OFF', key='crop_face_image_OFF', group_id='crop_face_image', default=True), 
                    ],

                    [sg.Text('顔画像ファイル出力頻度', tooltip='フレームをいくつ飛ばしてoutputするか調節する。frame_skip値に影響される。デフォルト値：80')],
                    [
                        sg.Radio('標準', key='frequency_crop_image_80',  group_id='frequency_crop_image', default=True), 
                        sg.Radio('頻回', key='frequency_crop_image_1', group_id='frequency_crop_image'), 
                    ],
                    # ------------------------------

                ], vertical_alignment=TOP, font=('BIZ-UDGOTHICB.TTC', 9)
            ),

            sg.Frame('表現その他',
                [

                    [sg.Text('類似度(%)', tooltip='similar_percentage。この数値以上の類似度を同一人物と保証する。デフォルト値：99.0(%)')], 
                    [sg.Slider(orientation='h', key='similar_percentage', range=((98.0, 99.5)), default_value=99.0, resolution=0.1 ,enable_events=True)],

                    [sg.Text('顔枠形状・有無')],
                    [
                        sg.Radio('クラシック', key='rectangle', group_id='rectangle_group', tooltip='顔周囲に四角枠（直接描画）を描画するか否かを指定'), 
                        sg.Radio('標準', key='standard', group_id='rectangle_group', tooltip='顔周囲に四角枠（png画像）を描画するか否かを指定', default=True), 
                        sg.Radio('無し', key='nothing', group_id='rectangle_group', tooltip='四角枠を表示しない')
                    ],

                    [sg.Text('写真表示', tooltip='検出した顔領域横に認証された個人のイメージを表示する')],
                    [
                        sg.Radio('ON', key='default_face_image_draw_ON', group_id='default_face_image_draw', default=True),
                        sg.Radio('OFF', key='default_face_image_draw_OFF', group_id='default_face_image_draw')
                    ],

                    [sg.Text('半透明表示', tooltip='表示情報を全て班透明表示にする')],
                    [
                        sg.Radio('ON', key='show_overlay_ON', group_id='show_overlay', default=True),
                        sg.Radio('OFF', key='show_overlay_OFF', group_id='show_overlay')
                    ],

                    [sg.Text('個人名表示', tooltip='認証された個人名を表示する')],
                    [
                        sg.Radio('ON', key='show_name_ON', group_id='show_name', default=True),
                        sg.Radio('OFF', key='show_name_OFF', group_id='show_name')
                    ],

                    [sg.Text('パーセンテージ表示', tooltip='類似度をパーセンテージで表示する')],
                    [
                        sg.Radio('ON', key='show_percentage_ON', group_id='show_percentage', default=True),
                        sg.Radio('OFF', key='show_percentage_OFF', group_id='show_percentage')
                    ],

                    [sg.Text('複数人同時認証', tooltip='一度に複数人を認証を許可するかどうか指定する')],
                    [
                        sg.Radio('許可', key='multiple_faces_ON', group_id='multiple_faces', default=True),
                        sg.Radio('不許可', key='multiple_faces_OFF', group_id='multiple_faces')
                    ],

                    [sg.Text('下部エリア表示', tooltip='下部エリアを表示する。')],
                    [
                        sg.Radio('ON', key='bottom_area_ON', group_id='bottom_area'),
                        sg.Radio('OFF', key='bottom_area_OFF', group_id='bottom_area', default=True)
                    ],

                ], vertical_alignment=TOP, font=('BIZ-UDGOTHICB.TTC', 9)
            ),
        ],

            # Buttons
            [sg.Text('config_FACE01GRAPHICS129.iniに書き込みを行います', font=("BIZ-UDGOTHICB.TTC, 10"))],
            [
                sg.Button("iniファイルへ書き込み", font=('BIZ-UDGOTHICB.TTC, 16'), key='btn'), 
                sg.Button('終了', font=('BIZ-UDGOTHICB.TTC, 16'), key='shut_down', button_color='red')
            ],
            
            # footer
            [sg.Image('images/footer.png', pad=((0,0),(0,0)))]

        ], element_justification = "center", pad=((0,0),(0,0))
        )
    ]
]
# ----------------------------------------

window = sg.Window(
    'SETTING MANAGER', 
    layout, 
    element_justification="center", 
    margins=(0,0), 
    icon='images/icon.png', 
    grab_anywhere=True, 
)

while True:

    event, value = window.read()

    if event == None:
        break

    if event=='shut_down':  ## 終了ボタンを押した場合
        print('終了します')
        break
    
    if value['do_remake_npKnown']==True:
        if os.path.isfile('npKnown.npz'):
            # os.remove('npKnown.npz')
            os.rename('npKnown.npz', 'npKnown.npz_')
            window['remove_npKnown'].Update('npKnow.npzを再構築します\nこれには時間がかかります')
    elif value['not_remake_npKnown']==True:
        try:
            os.rename('npKnown.npz_', 'npKnown.npz')
            window['remove_npKnown'].Update('今あるnpKnow.npzをそのまま\n使用します')
        except:
            pass

    if value['HLS'] and http_value==None:
        http_window_layout = [
            [sg.Text('IPとポート番号を入力して下さい')],      
            [sg.InputText('http://localhost:8080/')],      
            [sg.Submit()]
        ]
        http_window = sg.Window('HLS入力', http_window_layout)    
        http_event, http_value = http_window.read()    
        http_window.close()
        window['filename'].Update(http_value[0])
    
    if value['other_input'] and file_browse_value==None:
        file_browse_layout = [
            [sg.Text("ファイル"), sg.InputText(), 
                sg.FileBrowse(
                    '選択', 
                    file_types=(('その他の入力映像ファイル', '*.mp4'),), 
                )
            ],
            [sg.Submit('決定'), sg.Cancel('キャンセル')],
        ]
        file_browse_window = sg.Window('入力動画を選択', file_browse_layout)
        file_browse_event, file_browse_value = file_browse_window.read()
        file_browse_window.close()
        window['filename'].Update(file_browse_value[0])

    # iniファイルへ書き込みボタンを押した時の処理
    if event == 'btn':

        # 顔認識精度
        if value['jitters_priset_image_jitters100']==True:
            parameters['priset_face_images_jitters']='100'
        elif value['jitters_priset_image_jitters10']==True:
            parameters['priset_face_images_jitters']='10'

        # プリセット画像の再データ化
        if value['test.mp4']==True:
            parameters['remake_npKnown']='test.mp4'
        elif value['do_remake_npKnown']==True:
            parameters['remake_npKnown']='do_remake_npKnown'

        # 映像入力元
        if value['test.mp4']==True:
            parameters['input_moving_image']='test.mp4'
        elif value['web_camera']==True:
            parameters['input_moving_image']='usb'
        elif http_value:
            parameters['input_moving_image']=http_value[0]
        elif file_browse_value:
            parameters['input_moving_image']=file_browse_value[0]
            
        # 顔検出解像度
        if value['upsampling_input_moving_image_80']==True:
            parameters['upsampling_input_moving_image']='0'
        elif value['upsampling_input_moving_image_40']==True:
            parameters['upsampling_input_moving_image']='1'

        # フレームドロップ
        parameters['flame_skip']=int(value['flame_skip'])

        # エリア指定
        if value['area']=='エリア指定なし':
            parameters['area']='NONE'
        elif value['area']=='左上':
            parameters['area']='TOP_LEFT'
        elif value['area']=='右上':
            parameters['area']='TOP_RIGHT'
        elif value['area']=='左下':
            parameters['area']='BOTTOM_LEFT'
        elif value['area']=='右下':
            parameters['area']='BOTTOM_RIGHT'
        elif value['area']=='中央':
            parameters['area']='CENTER'

        # 映像幅リサイズ
        parameters['set_width']=int(value['set_width'])

        # 顔画像ファイル出力
        if value['crop_face_image_ON']==True:
            parameters['crop_face_image']='True'
        elif value['crop_face_image_OFF']==True:
            parameters['crop_face_image']='False'

        # 顔画像ファイル出力頻度
        if value['frequency_crop_image_80']==True:
            parameters['frequency_crop_image']='80'
        elif value['frequency_crop_image_1']==True:
            parameters['frequency_crop_image']='1'

        # 類似度(%)
        parameters['similar_percentage']=value['similar_percentage']
        
        # 顔枠形状・有無
        if value['rectangle']==True:
            parameters['rectangle']='True'
            parameters['target_rectangle']='False'
        elif value['standard']==True:
            parameters['rectangle']='False'
            parameters['target_rectangle']='True'
        elif value['nothing']==True:
            parameters['rectangle']='False'
            parameters['target_rectangle']='False'

        # 写真表示
        if value['default_face_image_draw_ON']==True:
            parameters['default_face_image_draw']='True'
        elif value['default_face_image_draw_OFF']==True:
            parameters['default_face_image_draw']='False'

        # 半透明表示
        if value['show_overlay_ON']==True:
            parameters['show_overlay']='True'
        elif value['show_overlay_OFF']==True:
            parameters['show_overlay']='False'

        # 個人名表示
        if value['show_name_ON']==True:
            parameters['show_name']='True'
        elif value['show_name_OFF']==True:
            parameters['show_name']='False'

        # パーセンテージ表示
        if value['show_percentage_ON']==True:
            parameters['show_percentage']='True'
        elif value['show_percentage_OFF']==True:
            parameters['show_percentage']='False'

        # 複数人同時認証
        if value['multiple_faces_ON']==True:
            parameters['multiple_faces']='True'
        elif value['multiple_faces_OFF']==True:
            parameters['multiple_faces']='False'

        # 下部エリア表示
        if value['bottom_area_ON']==True:
            parameters['bottom_area']='True'
        elif value['bottom_area_OFF']==True:
            parameters['bottom_area']='False'

        # configファイルを書き換え ---------------
        conf=configparser.ConfigParser()

        DEFAULT='DEFAULT'
        
        conf.set(DEFAULT, 'similar_percentage', str(parameters['similar_percentage']))
        conf.set(DEFAULT, 'jitters', '0')
        conf.set(DEFAULT, 'priset_face_images_jitters', parameters['priset_face_images_jitters'])
        conf.set(DEFAULT, 'upsampling', parameters['upsampling_input_moving_image'])
        conf.set(DEFAULT, 'mode', 'cnn')
        conf.set(DEFAULT, 'frame_skip', str(parameters['flame_skip']))
        conf.set(DEFAULT, 'movie', parameters['input_moving_image'])
        conf.set(DEFAULT, 'set_area', parameters['area'])
        conf.set(DEFAULT, 'SET_WIDTH', str(parameters['set_width']))
        conf.set(DEFAULT, 'rectangle', parameters['rectangle'])
        conf.set(DEFAULT, 'target_rectangle', parameters['target_rectangle'])
        conf.set(DEFAULT, 'show_video', 'False')
        conf.set(DEFAULT, 'crop_face_image', parameters['crop_face_image'])
        conf.set(DEFAULT, 'frequency_crop_image', parameters['frequency_crop_image'])
        conf.set(DEFAULT, 'default_face_image_draw', parameters['default_face_image_draw'])
        conf.set(DEFAULT, 'show_overlay', parameters['show_overlay'])
        conf.set(DEFAULT, 'show_percentage', parameters['show_percentage'])
        conf.set(DEFAULT, 'show_name', parameters['show_name'])
        conf.set(DEFAULT, 'print_property', 'False')
        conf.set(DEFAULT, 'calculate_time', 'False')
        conf.set(DEFAULT, 'multiple_faces', parameters['multiple_faces'])
        conf.set(DEFAULT, 'bottom_area', parameters['bottom_area'])

        # # iniファイルへの書き込み
        with open('config_FACE01GRAPHICS129.ini', 'w') as ini_file:
            conf.write(ini_file)
        # ----------------------------------------

window.close()

