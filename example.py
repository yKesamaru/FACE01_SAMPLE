import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import datetime
from PIL import Image
from time import perf_counter

# セットアップ
# INPUT = "test.mp4"
INPUT = "some_people.mp4"
# INPUT = "顔無し区間を含んだテスト動画.mp4"
SET_WIDTH = 750

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
    date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
    # imgCroped = pil_img_obj_rgb.crop((left -20,top -20,right +20,bottom +20)).resize((200, 200))
    try:
        imgCroped = pil_img_obj_rgb.crop((left -150,top -150,right +150,bottom +150))
    except:
        pass
    filename = "output/%s.png" % (date)
    imgCroped.save(filename)
    return filename

def Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR):
    HANDLING_FRAME_TIME = (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)  ## 小数点以下がミリ秒
    return HANDLING_FRAME_TIME

vcap = cv2.VideoCapture(INPUT)
while True:
    ret, frame = vcap.read()
    frame, set_width, set_height = resize_frame(frame, SET_WIDTH)
    # """DEBUG
    cv2.imshow("DEBUG", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    # """
    if ret:
        face_location_list = return_face_location_list(frame, set_width, set_height, model_selection=0, min_detection_confidence=0.4)
        pil_img_obj_rgb = pil_img_rgb_instance(frame)
        for location in face_location_list:
            HANDLING_FRAME_TIME_FRONT = perf_counter()
            filename = make_crop_face_image(pil_img_obj_rgb, location[0], location[3], location[1], location[2])
            HANDLING_FRAME_TIME_REAR = perf_counter()
            HANDLING_FRAME_TIME = Measure_processing_time(HANDLING_FRAME_TIME_FRONT,HANDLING_FRAME_TIME_REAR)

            print(face_location_list, filename, HANDLING_FRAME_TIME)