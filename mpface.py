import mediapipe as mp
    """mediapipe for python, see bellow
    https://github.com/google/mediapipe/tree/master/mediapipe/python
    """
from face01lib.video_capture import video_capture
import configparser
import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import datetime
import cProfile as pr
import time

# configファイル読み込み
conf = configparser.ConfigParser()
conf.read('config.ini', 'utf-8')
similar_percentage: float =         float(conf.get('DEFAULT','similar_percentage'))
jitters: int  =int(conf.get('DEFAULT','jitters'))
priset_face_images_jitters: int =int(conf.get('DEFAULT','priset_face_images_jitters'))
upsampling: int =int(conf.get('DEFAULT','upsampling'))
mode: str =                       conf.get('DEFAULT','mode')
frame_skip: int =int(conf.get('DEFAULT','frame_skip'))
movie: str =                      conf.get('DEFAULT','movie')
rectangle: bool = conf.getboolean('DEFAULT','rectangle')
target_rectangle: bool = conf.getboolean('DEFAULT','target_rectangle')
show_video: bool = conf.getboolean('DEFAULT','show_video')
frequency_crop_image: int =int(conf.get('DEFAULT','frequency_crop_image'))
set_area: str =                   conf.get('DEFAULT','set_area')
print_property: bool = conf.getboolean('DEFAULT','print_property')
calculate_time: bool = conf.getboolean('DEFAULT','calculate_time')
SET_WIDTH: int =int(conf.get('DEFAULT','SET_WIDTH'))
default_face_image_draw: bool = conf.getboolean('DEFAULT', 'default_face_image_draw')
show_overlay: bool = conf.getboolean('DEFAULT', 'show_overlay')
show_percentage: bool = conf.getboolean('DEFAULT', 'show_percentage')
crop_face_image: bool = conf.getboolean('DEFAULT', 'crop_face_image')
show_name: bool = conf.getboolean('DEFAULT', 'show_name')
should_not_be_multiple_faces: bool = conf.getboolean('DEFAULT', 'should_not_be_multiple_faces')
bottom_area: bool = conf.getboolean('DEFAULT', 'bottom_area')
draw_telop_and_logo: bool = conf.getboolean('DEFAULT', 'draw_telop_and_logo')

def resize_frame(SET_WIDTH, SET_HEIGHT, frame):
    small_frame: cv2.Mat = cv2.resize(frame, (SET_WIDTH, SET_HEIGHT))
    return small_frame


# ホームディレクトリ固定
def home() -> tuple:
    kaoninshoDir: str = os.path.dirname(__file__)
    priset_face_imagesDir: str = f'{os.path.dirname(__file__)}/priset_face_images/'
    return kaoninshoDir, priset_face_imagesDir


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
        sg.popup( '入力指定された映像データ幅が小さすぎます','width: {}px'.format(width), 'このまま処理を続けますが', '性能が発揮できない場合があります','OKを押すと続行します', title='警告', button_type = POPUP_BUTTONS_OK, modal = True, keep_on_top = True)
    # しかしながらパイプ処理の際フレームサイズがわからなくなるので決め打ちする
    SET_HEIGHT: int = int((SET_WIDTH * height) / width)
    return SET_WIDTH,fps,height,width,SET_HEIGHT


# Initialize variables, load images
def initialize(SET_WIDTH):
    kaoninshoDir, priset_face_imagesDir = home()
    os.chdir(kaoninshoDir)

    load_telop_image: bool = False
    load_logo_image: bool = False
    load_unregistered_face_image: bool = False

    # Load Telop image
    telop_image: cv2.Mat
    if not load_telop_image:
       telop_image = cv2.imread("images/telop.png", cv2.IMREAD_UNCHANGED)
       load_telop_image = True

    # Load Logo image
    logo_image: cv2.Mat
    if not load_logo_image:
        logo_image: cv2.Mat = cv2.imread("images/Logo.png", cv2.IMREAD_UNCHANGED)
        load_logo_image = True

    rect01_png:cv2.Mat = cv2.imread("images/rect01.png", cv2.IMREAD_UNCHANGED)

    # Make vcap object
    vcap = video_capture(kaoninshoDir, movie)

    # Get frame info (fps, height etc)
    if not 'height' in locals():
        SET_WIDTH, fps, height, width, SET_HEIGHT = return_movie_property(SET_WIDTH, vcap)

    # Load unregistered_face_image
    unregistered_face_image: cv2.Mat
    if not load_unregistered_face_image:
        unregistered_face_image = np.array(Image.open('./images/顔画像未登録.png'))
        unregistered_face_image = cv2.cvtColor(unregistered_face_image, cv2.COLOR_BGR2RGBA)
        load_unregistered_face_image = True

    # 日付時刻算出
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒

    return date, rect01_png, telop_image, logo_image, vcap, unregistered_face_image, SET_WIDTH, fps, height, width, SET_HEIGHT, kaoninshoDir, priset_face_imagesDir


date, rect01_png, telop_image, logo_image, vcap, unregistered_face_image, \
    SET_WIDTH, fps, height, width, SET_HEIGHT, kaoninshoDir, priset_face_imagesDir = \
        initialize(SET_WIDTH)


def mp_face_detection_func(small_frame):
    face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.4
    )
    """
    MediaPipe Face Detection.
    MediaPipe Face Detection processes an RGB image and returns a list of the
    detected face location data.
    Please refer to
    https://solutions.mediapipe.dev/face_detection#python-solution-api
    for usage examples.
    """    
    # 推論処理
    results = face.process(small_frame)
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
    return results

def test(vcap, SET_WIDTH, SET_HEIGHT):
    while True:
        ret, frame = vcap.read()
        small_frame = resize_frame(SET_WIDTH, SET_HEIGHT, frame)
        small_frame.flags.writeable = False
        results = mp_face_detection_func(small_frame)
        yield results, small_frame


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('終了', key='terminate', pad=(0,10))]
    ]
    window = sg.Window(
        'CALL_FACE01GRAPHICS', layout, alpha_channel = 1, margins=(0, 0),
        no_titlebar = True, grab_anywhere = True,
        location=(350,130), modal = True
    )

    exec_times: int = 20
    HANDLING_FRAME_TIME: float = 0.0
    HANDLING_FRAME_TIME_FRONT: float = 0.0
    HANDLING_FRAME_TIME_REAR: float = 0.0

    # h, w, c = sample_img.shape
    # print('width:  ', w)
    # print('height: ', h)

    # xleft = data.xmin*w
    # xleft = int(xleft)
    # xtop = data.ymin*h
    # xtop = int(xtop)
    # xright = data.width*w + xleft
    # xright = int(xright)
    # xbottom = data.height*h + xtop
    # xbottom = int(xbottom)

    def profile(exec_times):
        HANDLING_FRAME_TIME_FRONT = time.perf_counter()
        xs = test(vcap, SET_WIDTH, SET_HEIGHT)
        for result, small_frame in xs:
            if not result.detections:
                continue
            if  exec_times >= 0:
                exec_times = exec_times - 1
                print('\n------------')
                print(f'人数: {len(result.detections)}人')
                print(f'exec_times: {exec_times}')
                for detection in result.detections:
                    event, _ = window.read(timeout = 1)
                    xleft = int(detection.location_data.relative_bounding_box.xmin * SET_WIDTH)
                    xtop = int(detection.location_data.relative_bounding_box.ymin * SET_HEIGHT)
                    xright = int(detection.location_data.relative_bounding_box.width * SET_WIDTH + xleft)
                    xbottom = int(detection.location_data.relative_bounding_box.height * SET_HEIGHT + xtop)
                    """see bellow
                    https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
                    """
                    print(f'信頼度: {round(detection.score[0]*100, 2)}%')
                    print(f'座標: {(xleft,xtop,xright,xbottom)}')
                    # small_frame.flags.writeable = True
                    mp_drawing.draw_detection(small_frame, detection)
                    imgbytes = cv2.imencode(".png", small_frame)[1].tobytes()
                    window["display"].update(data = imgbytes)
                if event=='terminate':
                    break
            else:
                break
        window.close()
        print('終了します')
        HANDLING_FRAME_TIME_REAR = time.perf_counter()
        HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT) 
        print(f'profile()関数の処理時間合計: {round(HANDLING_FRAME_TIME , 3)}[秒]')

    pr.run('profile(exec_times)', 'restats')

"""detection.location_data

format: RELATIVE_BOUNDING_BOX
relative_bounding_box {
  xmin: 0.11177106201648712
  ymin: 0.3772536516189575
  width: 0.1854635626077652
  height: 0.3302779793739319
}
relative_keypoints {
  x: 0.17963822185993195
  y: 0.4780540466308594
}
relative_keypoints {
  x: 0.2510705888271332
  y: 0.48258280754089355
}
relative_keypoints {
  x: 0.22213803231716156
  y: 0.5638952851295471
}
relative_keypoints {
  x: 0.21596230566501617
  y: 0.6227309107780457
}
relative_keypoints {
  x: 0.1223718449473381
  y: 0.48676401376724243
}
relative_keypoints {
  x: 0.2783607244491577
  y: 0.4942516088485718
}
"""










