import logging
from datetime import datetime
from io import BytesIO
from time import perf_counter
from typing import List, Tuple

import cv2
import mediapipe as mp
import requests
from PIL import Image
from requests.auth import HTTPDigestAuth
import numpy as np
from traceback import format_exc


def mp_face_detection_func(frame, model_selection=0, min_detection_confidence=0.4):
    face = mp.solutions.face_detection.FaceDetection(
        model_selection = model_selection,
        min_detection_confidence = min_detection_confidence
    )
    inference = face.process(frame)
    return inference

def return_face_location_list(frame, set_width, set_height, model_selection, min_detection_confidence) -> Tuple:
    frame.flags.writeable = False
    face_location_list: List = []
    result = mp_face_detection_func(frame, model_selection, min_detection_confidence)
    if not result.detections:
        return face_location_list
    else:
        for detection in result.detections:
            xleft:int = int(detection.location_data.relative_bounding_box.xmin * set_width)
            xtop :int= int(detection.location_data.relative_bounding_box.ymin * set_height)
            xright:int = int(detection.location_data.relative_bounding_box.width * set_width + xleft)
            xbottom:int = int(detection.location_data.relative_bounding_box.height * set_height + xtop)
            if xleft <= 0 or xtop <= 0:  # xleft or xtop がマイナスになる場合があるとバグる
                continue
            face_location_list.append((xtop,xright,xbottom,xleft))  # faceapi order
    frame.flags.writeable = True
    return face_location_list

def resize_frame(frame, width):
    h, w = frame.shape[:2]
    height = round(h * (width / w))
    frame = cv2.resize(frame, dsize = (width, height))
    return frame, width, height

def pil_img_rgb_instance(frame):
    pil_img_obj_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    return  pil_img_obj_rgb

def make_crop_face_image(pil_img_obj_rgb, top, left, right, bottom):
    date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    # imgCroped = pil_img_obj_rgb.crop((left -20,top -20,right +20,bottom +20)).resize((200, 200))
    try:
        imgCroped = pil_img_obj_rgb.crop((left -150,top -150,right +150,bottom +150))
    except:
        pass
    filename = "output/%s.png" % (date)
    imgCroped.save(filename)
    return filename

def Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR):
    HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)
    return HANDLING_FRAME_TIME

def cal_specify_date() -> None:
    """指定日付計算"""
    limit_date = datetime(2022, 8, 1, 0, 0, 0)
    today = datetime.now()
    if today >= limit_date:
        print('指定日付を過ぎました')
        exit(0)

def finalize(vcap):
    vcap.release()
    cv2.destroyAllWindows()

def frame_generator(INPUT, vcap, logger):
    """INPUT source"""
    if 'http' in INPUT:
        while True:
            response = requests.get(INPUT, auth = HTTPDigestAuth("", ""))  # user, passwd
            if response.headers['Content-type'] == 'image/jpeg':
                img_bin = BytesIO(response.content)
                img_pil = Image.open(img_bin)
                img_np  = np.asarray(img_pil)
                frame  = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                ret = True
                yield frame
            else:
                logger.warning("INPUTが開けません")
                logger.warning(format_exc(limit=None, chain=True))
    else:
        while vcap.isOpened(): 
            ret, frame = vcap.read()
            if not ret:
                logger.warning("INPUTが開けません")
                logger.warning("以下のエラーをシステム管理者へお伝えください")
                logger.warning("-" * 20)
                logger.warning(format_exc(limit=None, chain=True))
                logger.warning("-" * 20)
                finalize(vcap)
                break
            yield frame


def example(LEVEL, INPUT, SET_WIDTH):
    """Logging"""
    logger = logging.getLogger(__name__)
    if LEVEL == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')
    file_handler = logging.FileHandler('example.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    """Limit"""
    cal_specify_date()

    vcap = cv2.VideoCapture(INPUT, cv2.CAP_FFMPEG)
    while True:
        frame_generator_obj = frame_generator(INPUT, vcap, logger)
        frame = frame_generator_obj.__next__()
        frame, set_width, set_height = resize_frame(frame, SET_WIDTH)
        if LEVEL == "DEBUG":
            cv2.imshow("DEBUG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        face_location_list = return_face_location_list(frame, set_width, set_height, model_selection=0, min_detection_confidence=0.4)
        pil_img_obj_rgb = pil_img_rgb_instance(frame)
        for location in face_location_list:
            HANDLING_FRAME_TIME_FRONT = perf_counter()
            filename = make_crop_face_image(pil_img_obj_rgb, location[0], location[3], location[1], location[2])
            HANDLING_FRAME_TIME_REAR = perf_counter()
            HANDLING_FRAME_TIME = Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR)

            logger.debug((face_location_list, filename, HANDLING_FRAME_TIME))

if __name__ == '__main__':
    # 仮
    INPUT = "test.mp4"
    INPUT = "some_people.mp4"
    INPUT = "顔無し区間を含んだテスト動画.mp4"
    INPUT = "http://175.210.52.167:84/SnapshotJPEG?Resolution=640x480"
    SET_WIDTH = 750

    example("DEBUG", INPUT, SET_WIDTH)
