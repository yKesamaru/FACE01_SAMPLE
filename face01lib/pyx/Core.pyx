"""mediapipe for python, see bellow
https://github.com/google/mediapipe/tree/master/mediapipe/python
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
Dlib_api(): (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
"""
__model__ = 'Original model create by Prokofev Kirill, modified by PINT'
__URL__ = 'https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3'

from datetime import datetime
from platform import system
from traceback import format_exc

import cv2
# from asyncio.log import logger
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True

from face01lib.api import Dlib_api
from face01lib.Calc import Cal

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os.path import dirname, exists
from shutil import move

import onnxruntime

from face01lib import face_recognition_models
from face01lib.logger import Logger

anti_spoof_model = face_recognition_models.anti_spoof_model_location()
onnx_session = onnxruntime.InferenceSession(anti_spoof_model)

name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir)
Cal().cal_specify_date(logger)

class Core:
    def __init__(self) -> None:
        pass

    def mp_face_detection_func(self, resized_frame, model_selection=0, min_detection_confidence=0.4):
        face = mp.solutions.face_detection.FaceDetection(  # type: ignore
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

    def return_face_location_list(self, resized_frame, set_width, set_height, model_selection, min_detection_confidence) -> list:
        """
        return: face_location_list
        """
        self.resized_frame = resized_frame
        self.set_width = set_width
        self.set_height = set_height
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.resized_frame.flags.writeable = False
        face_location_list: list = []
        person_frame = np.empty((2,0), dtype = np.float64)
        result = self.mp_face_detection_func(self.resized_frame, self.model_selection, self.min_detection_confidence)
        if not result.detections:
            return face_location_list
        else:
            for detection in result.detections:
                xleft:int = int(detection.location_data.relative_bounding_box.xmin * self.set_width)
                xtop :int= int(detection.location_data.relative_bounding_box.ymin * self.set_height)
                xright:int = int(detection.location_data.relative_bounding_box.width * self.set_width + xleft)
                xbottom:int = int(detection.location_data.relative_bounding_box.height * self.set_height + xtop)
                # see bellow
                # https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
                
                if xleft <= 0 or xtop <= 0:  # xleft or xtop がマイナスになる場合があるとバグる
                    continue
                face_location_list.append((xtop,xright,xbottom,xleft))  # Dlib_api() order

        self.resized_frame.flags.writeable = True
        return face_location_list

    def draw_telop(self, logger, cal_resized_telop_nums, frame: np.ndarray) -> np.ndarray:
        self.logger = logger
        self.cal_resized_telop_nums = cal_resized_telop_nums
        self.frame = frame
        x1, y1, x2, y2, a, b = self.cal_resized_telop_nums
        try:
            frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2] * a + b
        except:
            self.logger.debug("telopが描画できません")
            # TODO np.clip
        return  frame

    def draw_logo(self, logger, cal_resized_logo_nums, frame) -> np.ndarray:
        self.logger = logger
        self.cal_resized_logo_nums = cal_resized_logo_nums
        self.frame = frame
        x1, y1, x2, y2, a, b = self.cal_resized_logo_nums
        try:
            frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2] * a + b
        except:
            self.logger.debug("logoが描画できません")
            # TODO np.clip
        return frame

    def check_compare_faces(self, logger, known_face_encodings, face_encoding, tolerance) -> list:
        self.logger = logger
        self.known_face_encodings = known_face_encodings
        self.face_encoding = face_encoding
        self.tolerance = tolerance
        try:
            matches = Dlib_api().compare_faces(self.known_face_encodings, self.face_encoding, self.tolerance)
            return matches
        except:
            self.logger.warning("DEBUG: npKnown.npzが壊れているか予期しないエラーが発生しました。")
            self.logger.warning("npKnown.npzの自動削除は行われません。原因を特定の上、必要な場合npKnown.npzを削除して下さい。")
            self.logger.warning("処理を終了します。FACE01を再起動して下さい。")
            self.logger.warning("以下のエラーをシステム管理者様へお伝えください")
            self.logger.warning("-" * 20)
            self.logger.warning(format_exc(limit=None, chain=True))
            self.logger.warning("-" * 20)
            exit(0)

    def make_frame_datas_array(self, overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,resized_frame):
        """データ構造(frame_datas_list)を返す

        person_data:
            {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
            
            person_data内のlocationは個々人の顔座標です。個々人を特定しない場合の顔座標はframe_detas['face_location_list']を使ってください。
        
        person_data_list: 
            person_data_list.append(person_data)
        
        frame_datas:
            {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}

        frame_datas_list: 
            frame_datas_array.append(frame_datas)

        return: frame_datas_list
        """
        self.overlay = overlay
        self.face_location_list = face_location_list
        self.name = name
        self.filename = filename
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        self.percentage_and_symbol = percentage_and_symbol
        self.person_data_list = person_data_list
        self.frame_datas_array = frame_datas_array
        self.resized_frame = resized_frame
        date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
        person_data = {'name': self.name, 'pict':self.filename,  'date':date, 'location':(self.top,self.right,self.bottom,self.left), 'percentage_and_symbol': self.percentage_and_symbol}
        self.person_data_list.append(person_data)
        frame_datas = {'img':self.resized_frame, 'face_location_list':self.face_location_list, 'overlay': self.overlay, 'person_data_list': self.person_data_list}
        self.frame_datas_array.append(frame_datas)
        return self.frame_datas_array

    # pil_img_rgbオブジェクトを生成
    def pil_img_rgb_instance(self, frame):
        self.frame = frame
        pil_img_obj_rgb = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA))
        return  pil_img_obj_rgb

    # 顔部分の領域をクロップ画像ファイルとして出力
    def make_crop_face_image(self, name, dis, pil_img_obj_rgb, top, left, right, bottom, number_of_crops, frequency_crop_image):
        self.name = name
        self.dis = dis
        self.pil_img_obj_rgb = pil_img_obj_rgb
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
        self.number_of_crops = number_of_crops
        self.frequency_crop_image = frequency_crop_image
        date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")
        imgCroped = self.pil_img_obj_rgb.crop((self.left -20,self.top -20,self.right +20,self.bottom +20)).resize((200, 200))
        filename = "output/%s_%s_%s.png" % (self.name, date, self.dis)
        imgCroped.save(filename)
        return filename,self.number_of_crops, self.frequency_crop_image

    # Get face_names
    def return_face_names(self, args_dict, face_names, face_encoding, matches, name):
        self.args_dict = args_dict
        self.face_names = face_names
        self.face_encoding = face_encoding
        self.matches = matches
        self.name = name
        # 各プリセット顔画像のエンコーディングと動画中の顔画像エンコーディングとの各顔距離を要素としたアレイを算出
        face_distances = Dlib_api().face_distance(self.args_dict["known_face_encodings"], self.face_encoding)  ## face_distances -> shape:(677,), face_encoding -> shape:(128,)
        # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素のインデックスを求める
        best_match_index = np.argmin(face_distances)
        # プリセット顔画像と動画中顔画像との各顔距離を要素とした配列に含まれる要素のうち、最小の要素の値を求める
        min_face_distance: str = str(min(face_distances))  # あとでファイル名として文字列として加工するので予めstr型にしておく
        # アレイ中のインデックスからknown_face_names中の同インデックスの要素を算出
        if self.matches[best_match_index]:  # tolerance以下の人物しかここは通らない。
            file_name = self.args_dict["known_face_names"][best_match_index]
            self.name = file_name + ':' + min_face_distance
        self.face_names.append(self.name)
        return self.face_names

    def return_concatenate_location_and_frame(self, resized_frame, face_location_list):
        self.resized_frame = resized_frame
        self.face_location_list = face_location_list
        """face_location_listはresized_frame上の顔座標"""
        finally_height_size:int = 150
        concatenate_face_location_list = []
        detection_counter:int = 0
        person_frame_list: list = []
        concatenate_person_frame: np.ndarray = np.empty((0,0), dtype=float)
        for xtop,xright,xbottom,xleft in self.face_location_list:
            person_frame = self.resized_frame[xtop:xbottom, xleft:xright]
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

            concatenate_face_location_list.append((concatenated_xtop,concatenated_xright,concatenated_xbottom,concatenated_xleft))  # Dlib_api() order
            detection_counter += 1
            """about coordinate order
            dlib: (Left, Top, Right, Bottom,)
            Dlib_api(): (top, right, bottom, left)
            see bellow
            https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
            """

            concatenate_person_frame = np.hstack(person_frame_list)
            """DEBUG
            cv2.imshow("face_encodings", concatenate_person_frame)
            cv2.moveWindow("face_encodings", 800,600)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            exit(0)
            print("---------------------------------")
            print(f'concatenate_face_location_list: {concatenate_face_location_list}')
            print("---------------------------------")
            """
        return concatenate_face_location_list, concatenate_person_frame

    # フレーム前処理
    def frame_pre_processing(self, logger, args_dict, resized_frame):
        self.logger = logger
        self.args_dict = args_dict
        self.resized_frame = resized_frame
        person_data_list = [{}]
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
        if  self.args_dict["headless"] == False:
            # 半透明処理（前半）
            if self.args_dict["show_overlay"]==True:
                overlay: cv2.Mat = self.resized_frame.copy()

            # テロップとロゴマークの合成
            if self.args_dict["draw_telop_and_logo"] == True:
                self.resized_frame =  self.draw_telop(self.logger, self.args_dict["cal_resized_telop_nums"], self.resized_frame)
                self.resized_frame = self.draw_logo(self.logger, self.args_dict["cal_resized_logo_nums"], self.resized_frame)

        # 顔座標算出
        if self.args_dict["use_pipe"] == True:
            face_location_list = self.return_face_location_list(self.resized_frame, self.args_dict["set_width"], self.args_dict["set_height"], self.args_dict["model_selection"], self.args_dict["min_detection_confidence"])
        else:
            face_location_list = Dlib_api().face_locations(self.resized_frame, self.args_dict["upsampling"], self.args_dict["mode"])
        """face_location_list
        [(144, 197, 242, 99), (97, 489, 215, 371)]
        """

        # 顔がなかったら以降のエンコード処理を行わない
        if len(face_location_list) == 0:
            frame_datas_array = self.make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
            return frame_datas_array

        # 顔が一定数以上なら以降のエンコード処理を行わない
        if len(face_location_list) >= self.args_dict["number_of_people"]:
            self.logger.info(f'{self.args_dict["number_of_people"]}人以上を検出しました')
            frame_datas_array = self.make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
            return frame_datas_array

        frame_datas_array = self.make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
        return frame_datas_array

    # 顔のエンコーディング
    # @profile()
    def face_encoding_process(self, logger, args_dict, frame_datas_array):
        """frame_datas_arrayの定義
        person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
        person_data_list.append(person_data)
        frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
        frame_datas_array.append(frame_datas)
        """
        self.logger = logger
        self.args_dict = args_dict
        self.frame_datas_array = frame_datas_array
        face_encodings = []
        for frame_data in self.frame_datas_array:
            resized_frame = frame_data["img"]
            face_location_list = frame_data["face_location_list"]  # [(139, 190, 257, 72)]
            if len(face_location_list) == 0:
                return face_encodings, self.frame_datas_array
            elif len(face_location_list) > 0:
                # 顔ロケーションからエンコーディングを求める
                if self.args_dict["use_pipe"] == True and  self.args_dict["person_frame_face_encoding"] == True:
                    """FIX
                    人数分を繰り返し処理しているので時間がかかる。
                    dlibは一つの画像に複数の座標を与えて一度に処理をする。
                    なので各person_frameをくっつけて一つの画像にすれば処理時間は短くなる。
                        numpy.hstack(tup)[source]
                        Stack arrays in sequence horizontally (column wise).
                        https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
                    """
                    concatenate_face_location_list, concatenate_person_frame = \
                        self.return_concatenate_location_and_frame(resized_frame, face_location_list)
                    face_encodings = Dlib_api().face_encodings(concatenate_person_frame, concatenate_face_location_list, self.args_dict["jitters"], self.args_dict["model"])
                elif self.args_dict["use_pipe"] == True and  self.args_dict["person_frame_face_encoding"] == False:
                    face_encodings = Dlib_api().face_encodings(resized_frame, face_location_list, self.args_dict["jitters"], self.args_dict["model"])
                elif self.args_dict["use_pipe"] == False and  self.args_dict["person_frame_face_encoding"] == True:
                    self.logger.warning("config.ini:")
                    self.logger.warning("mediapipe = False  の場合 person_frame_face_encoding = True  には出来ません")
                    self.logger.warning("システム管理者様へ連絡の後、設定を変更してください")
                    self.logger.warning("-" * 20)
                    self.logger.warning(format_exc(limit=None, chain=True))
                    self.logger.warning("-" * 20)
                    self.logger.warning("処理を終了します")
                    exit(0)
                elif self.args_dict["use_pipe"] == False and self.args_dict["person_frame_face_encoding"] == False:
                    face_encodings = Dlib_api().face_encodings(resized_frame, face_location_list, self.args_dict["jitters"], self.args_dict["model"])
            return face_encodings, self.frame_datas_array

    # 顔の生データ(np.ndarray)を返す
    def return_face_image(self, resized_frame, face_location):
        self.resized_frame = resized_frame
        if len(self.face_location) > 0:
            top = face_location[0]
            right = face_location[1]
            bottom = face_location[2]
            left = face_location[3]
            face_image = self.resized_frame[top:bottom, left:right]
            """How to slice
            face_location order: top, right, bottom, left
            how to slice: img[top:bottom, left:right]
            """
            """DEBUG
            from face01lib.video_capture import VidCap
            VidCap().frame_imshow_for_debug(face_image)
            VidCap().frame_imshow_for_debug(self.resized_frame)
            """
            return face_image
        else:
            return []

    # フレーム後処理
    # @profile()
    def frame_post_processing(self, logger, args_dict, face_encodings, frame_datas_array, GLOBAL_MEMORY):
        """frame_datas_arrayの定義
        person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
        person_data_list.append(person_data)
        frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
        frame_datas_array.append(frame_datas)
        """
        self.logger = logger
        self.args_dict = args_dict
        self.face_encodings = face_encodings
        self.frame_datas_array = frame_datas_array
        self.GLOBAL_MEMORY = GLOBAL_MEMORY
        face_names = []
        face_location_list = []
        filename = ''
        debug_frame_turn_count = 0
        modified_frame_list = []

        for frame_data in self.frame_datas_array:
            if "face_location_list" not in frame_data:
                if self.args_dict["headless"] == False:
                    # 半透明処理（後半）_1frameに対して1回
                    if self.args_dict["show_overlay"]==True:
                        cv2.addWeighted(frame_data["overlay"], self.GLOBAL_MEMORY["alpha"], frame_data["img"], 1-self.GLOBAL_MEMORY["alpha"], 0, frame_data["img"])
                continue

            resized_frame = frame_data["img"]
            face_location_list = frame_data["face_location_list"]
            overlay = frame_data["overlay"]
            person_data_list = frame_data["person_data_list"]
            date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")

            # 名前リスト作成
            for face_encoding in self.face_encodings:
                # Initialize name, matches (Inner frame)
                name = "Unknown"
                matches = self.check_compare_faces(self.logger, self.args_dict["known_face_encodings"], face_encoding, self.args_dict["tolerance"])
                # 名前リスト(face_names)生成
                face_names = self.return_face_names(self.args_dict, face_names, face_encoding,  matches, name)

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
                    percentage = Cal().return_percentage(p)
                    percentage = round(percentage, 1)
                    percentage_and_symbol = str(percentage) + '%'
                    # ファイル名を最初のアンダーバーで区切る（アンダーバーは複数なのでmaxsplit = 1）
                    try:
                        name, _ = name.split('_', maxsplit = 1)
                    except:
                        self.logger.warning('ファイル名に異常が見つかりました',name,'NAME_default.png あるいはNAME_001.png (001部分は001からはじまる連番)にしてください','noFaceフォルダに移動します')
                        move(name, './noFace/')
                        return

                # クロップ画像保存
                if self.args_dict["crop_face_image"]==True:
                    if self.args_dict["frequency_crop_image"] < self.GLOBAL_MEMORY['number_of_crops']:
                        pil_img_obj_rgb = self.pil_img_rgb_instance(resized_frame)
                        if self.args_dict["crop_with_multithreading"] == True:
                            # """1.3.08 multithreading 9.05s
                            with ThreadPoolExecutor() as executor:
                                future = executor.submit(self.make_crop_face_image, name, dis, pil_img_obj_rgb, top, left, right, bottom, self.GLOBAL_MEMORY['number_of_crops'], self.args_dict["frequency_crop_image"])
                                filename,number_of_crops, frequency_crop_image = future.result()
                            # """
                        else:
                            # """ORIGINAL: 1.3.08で変更 8.69s
                            filename,number_of_crops, frequency_crop_image = \
                                self.make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, self.GLOBAL_MEMORY['number_of_crops'], self.args_dict["frequency_crop_image"])
                            # """
                        self.GLOBAL_MEMORY['number_of_crops'] = 0
                    else:
                        self.GLOBAL_MEMORY['number_of_crops'] += 1

                # 描画系
                if self.args_dict["headless"] == False:
                    # デフォルト顔画像の描画
                    if p <= self.args_dict["tolerance"]:  # ディスタンスpがtolerance以下の場合
                        if self.args_dict["default_face_image_draw"] == True:
                            resized_frame = self.draw_default_face(self.logger, self.args_dict, name, resized_frame, number_of_people)
                            number_of_people += 1  # 何人目か
                            """DEBUG"""
                            # frame_imshow_for_debug(resized_frame)

                    # ピンクまたは白の四角形描画
                    if self.args_dict["rectangle"] == True:
                        if name == 'Unknown':  # プリセット顔画像に対応する顔画像がなかった場合
                            resized_frame = self.draw_pink_rectangle(resized_frame, top,bottom,left,right)
                        else:  # プリセット顔画像に対応する顔画像があった場合
                            resized_frame = self.draw_white_rectangle(self.args_dict["rectangle"], resized_frame, top, left, right, bottom)
                        
                    # パーセンテージ描画
                    if self.args_dict["show_percentage"]==True:
                        resized_frame = self.display_percentage(percentage_and_symbol,resized_frame, p, left, right, bottom, self.args_dict["tolerance"])
                        """DEBUG"""
                        # frame_imshow_for_debug(resized_frame)

                    # 名前表示と名前用四角形の描画
                    if self.args_dict["show_name"]==True:
                        resized_frame = self.draw_rectangle_for_name(name,resized_frame, left, right,bottom)
                        pil_img_obj= Image.fromarray(resized_frame)
                        resized_frame = self.draw_text_for_name(self.logger, left,right,bottom,name, p,self.args_dict["tolerance"],pil_img_obj)
                        """DEBUG"""
                        # frame_imshow_for_debug(resized_frame)

                    # target_rectangleの描画
                    if self.args_dict["target_rectangle"] == True:
                        resized_frame = self.draw_target_rectangle(self.args_dict["anti_spoof"], self.args_dict["rect01_png"], self.args_dict["rect01_NG_png"], resized_frame,top,bottom,left,right,name)
                        """DEBUG
                        frame_imshow_for_debug(resized_frame)
                        """

                person_data = {'name': name, 'pict':filename,  'date':date, 'location':(top,right,bottom,left), 'percentage_and_symbol': percentage_and_symbol}
                person_data_list.append(person_data)
            # End for (top, right, bottom, left), name in zip(face_location_list, face_names)

            # _1frameに対して1回
            if self.args_dict["headless"] == False:
                frame_datas = {'img':resized_frame, 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}
                """DEBUG
                frame_imshow_for_debug(resized_frame)
                self.frame_datas_array.append(frame_datas)
                """
                modified_frame_list.append(frame_datas)

            elif self.args_dict["headless"] == True:
                frame_datas = {'img':'no-data_img', 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': person_data_list}  # TypeError: list indices must be integers or slices, not str -> img
                # self.frame_datas_array.append(frame_datas)
                modified_frame_list.append(frame_datas)
            else:
                frame_datas = {'img':'no-data_img', 'face_location_list':face_location_list, 'overlay': overlay, 'person_data_list': 'no-data_person_data_list'} 
                # self.frame_datas_array.append(frame_datas)
                modified_frame_list.append(frame_datas)

            if self.args_dict["headless"] == False:
                # 半透明処理（後半）_1frameに対して1回
                if self.args_dict["show_overlay"]==True:
                    # cv2.addWeighted(overlay, self.GLOBAL_MEMORY["alpha"], resized_frame, 1-self.GLOBAL_MEMORY["alpha"], 0, resized_frame)
                    for modified_frame in modified_frame_list:
                        cv2.addWeighted(modified_frame["overlay"], self.GLOBAL_MEMORY["alpha"], modified_frame["img"], 1-self.GLOBAL_MEMORY["alpha"], 0, modified_frame["img"])
                    # """DEBUG"""
                    # frame_imshow_for_debug(resized_frame)
            
        # return frame_datas
        """DEBUG
        print(f"modified_frame_list.__sizeof__(): {modified_frame_list.__sizeof__()}MB")
        """
        return modified_frame_list

    def return_anti_spoof(self, frame, face_location):
        self.frame = frame
        self.face_location = face_location
        face_image = self.return_face_image(self.frame, self.face_location)
        # VidCap_obj.frame_imshow_for_debug(face_image)  # DEBUG

        # 定形処理:リサイズ, 標準化, 成形, float32キャスト, 推論, 後処理
        input_image = cv2.resize(face_image, dsize=(128, 128))
        """DEBUG"""
        # VidCap_obj.frame_imshow_for_debug(input_image)

        input_image = input_image.transpose(2, 0, 1).astype('float32')
        input_image = input_image.reshape(-1, 3, 128, 128)

        # 推論
        input_name = onnx_session.get_inputs()[0].name
        result = onnx_session.run(None, {input_name: input_image})

        # 後処理
        result = np.array(result)
        result = np.squeeze(result)

        as_index = np.argmax(result)

        score: float = 0.0
        if result[0] > result[1]:
            score = result[0] - result[1]
        else:
            score = result[1] - result[0]
        score = round(score, 2)
        # ELE: Equally Likely Events
        ELE: bool = False
        if score < 0.3:
            ELE = True

        spoof_or_not: str = ''
        if as_index == 0:  # (255, 0, 0)
            spoof_or_not = 'spoof'
            return spoof_or_not, score, ELE
        if as_index == 1:
            spoof_or_not = 'not_spoof'
            return spoof_or_not, score, ELE

# 以下、元libdraw.LibDraw
    def draw_pink_rectangle(self, resized_frame, top,bottom,left,right) -> np.ndarray:
        self.resized_frame = resized_frame
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        cv2.rectangle(self.resized_frame, (self.left, self.top), (self.right, self.bottom), (255, 87, 243), 2) # pink
        return self.resized_frame
        
    def draw_white_rectangle(self, rectangle, resized_frame, top, left, right, bottom) -> np.ndarray:
        self.rectangle = rectangle
        self.resized_frame = resized_frame
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
        cv2.rectangle(self.resized_frame, (self.left-18, self.top-18), (self.right+18, self.bottom+18), (175, 175, 175), 2) # 灰色内枠
        cv2.rectangle(self.resized_frame, (self.left-20, self.top-20), (self.right+20, self.bottom+20), (255,255,255), 2) # 白色外枠
        return self.resized_frame

    # パーセンテージ表示
    def display_percentage(self, percentage_and_symbol,resized_frame, p, left, right, bottom, tolerance) -> np.ndarray:
        self.percentage_and_symbol = percentage_and_symbol
        self.resized_frame = resized_frame
        self.p = p
        self.left = left
        self.right = right
        self.bottom = bottom
        self.tolerance = tolerance
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # パーセンテージ表示用の灰色に塗りつぶされた四角形の描画
        cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 75), (self.right+25, self.bottom+50), (30,30,30), cv2.FILLED) # 灰色
        # テキスト表示位置
        fontsize = 14
        putText_center = int((self.left-25 + self.right+25)/2)
        putText_chaCenter = int(5/2)
        putText_pos = putText_center - (putText_chaCenter*fontsize) - int(fontsize/2)
        putText_position = (putText_pos, self.bottom + 75 - int(fontsize / 2))
        # toleranceの値によってフォント色を変える
        if self.p < self.tolerance:
            # パーセンテージを白文字表示
            self.resized_frame = cv2.putText(self.resized_frame, self.percentage_and_symbol, putText_position, font, 1, (255,255,255), 1, cv2.LINE_AA)
        else:
            # パーセンテージをピンク表示
            self.resized_frame = cv2.putText(self.resized_frame, self.percentage_and_symbol, putText_position, font, 1, (255, 87, 243), 1, cv2.LINE_AA)
        return self.resized_frame

    # デフォルト顔画像の描画処理
    def draw_default_face_image(self, logger, resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width):
        self.resized_frame = resized_frame
        self.default_face_small_image = default_face_small_image
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.number_of_people = number_of_people
        self.face_image_width = face_image_width
        try:
            self.x1 = self.x1 + (self.number_of_people * self.face_image_width)
            self.x2 = self.x2 + (self.number_of_people * self.face_image_width)
            self.resized_frame[self.y1:self.y2, self.x1:self.x2] = self.resized_frame[self.y1:self.y2, self.x1:self.x2] * (1 - self.default_face_small_image[:,:,3:] / 255) + self.default_face_small_image[:,:,:3] * (default_face_small_image[:,:,3:] / 255)
            # resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * a + b  # ValueError: assignment destination is read-only
            """DEBUG"""
            # frame_imshow_for_debug(resized_frame)
        except:
            logger.info('デフォルト顔画像の描画が出来ません')
            logger.info('描画面積が足りないか他に問題があります')
        return self.resized_frame

    # デフォルト顔画像の表示面積調整
    def adjust_display_area(self, args_dict, default_face_image) -> tuple:
        self.args_dict = args_dict
        self.default_face_image = default_face_image
        """TODO
        繰り返し計算させないようリファクタリング"""
        face_image_width = int(self.args_dict["set_width"] / 15)
        default_face_small_image = cv2.resize(self.default_face_image, dsize=(face_image_width, face_image_width))  # 幅・高さともに同じとする
        # 高さ = face_image_width
        x1, y1, x2, y2 = 0, self.args_dict["set_height"] - face_image_width - 10, face_image_width, self.args_dict["set_height"] - 10
        return x1, y1, x2, y2, default_face_small_image, face_image_width

    def draw_default_face(self, logger, args_dict, name, resized_frame, number_of_people):
        self.logger = logger
        self.args_dict = args_dict
        self.name = name
        self.resized_frame = resized_frame
        self.number_of_people = number_of_people
        default_face_image_dict = self.args_dict["default_face_image_dict"]

        default_name_png = self.name + '_default.png'
        default_face_image_name_png = './priset_face_images/' + default_name_png
        if not self.name in default_face_image_dict:  # default_face_image_dictにnameが存在しなかった場合
            # 各人物のデフォルト顔画像ファイルの読み込み
            if exists(default_face_image_name_png):
                # WINDOWSのopencv-python4.2.0.32ではcv2.imread()でpng画像を読み込めないバグが
                # 存在する可能性があると思う。そこでPNG画像の読み込みにはpillowを用いることにする
                default_face_image = np.array(Image.open(default_face_image_name_png))
                """DEBUG
                frame_imshow_for_debug(default_face_image)
                """
                # BGAをRGBへ変換
                default_face_image = cv2.cvtColor(default_face_image, cv2.COLOR_BGR2RGBA)
                """DEBUG
                frame_imshow_for_debug(default_face_image)
                """
                # if default_face_image.ndim == 3:  # RGBならアルファチャンネル追加 resized_frameがアルファチャンネルを持っているから。
                # default_face_imageをメモリに保持
                default_face_image_dict[self.name] = default_face_image  # キーnameと値default_face_imageの組み合わせを挿入する
            else:
                self.logger.info(f'{self.name}さんのデフォルト顔画像ファイルがpriset_face_imagesフォルダに存在しません')
                self.logger.info(f'{self.name}さんのデフォルト顔画像ファイルをpriset_face_imagesフォルダに用意してください')
        else:  # default_face_image_dictにnameが存在した場合
            default_face_image = default_face_image_dict[self.name]  # キーnameに対応する値をdefault_face_imageへ格納する
            """DEBUG
            frame_imshow_for_debug(default_face_image)  # OK
            """
            x1, y1, x2, y2 , default_face_small_image, face_image_width = self.adjust_display_area(args_dict, default_face_image)
            resized_frame = self.draw_default_face_image(logger, resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width)
        return resized_frame

    def draw_rectangle_for_name(self, name,resized_frame, left, right,bottom):
        self.name = name
        self.resized_frame = resized_frame
        self.left = left
        self.right = right
        self.bottom = bottom
        if self.name == 'Unknown':   # nameがUnknownだった場合
            self.resized_frame = cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 25), (self.right+25, self.bottom+50), (255, 87, 243), cv2.FILLED) # pink
        else:                   # nameが既知だった場合
            # cv2.rectangle(resized_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), thickness = 1) # 濃い水色の線
            self.resized_frame = cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 25), (self.right+25, self.bottom+50), (211, 173, 54), cv2.FILLED) # 濃い水色
        return self.resized_frame

    # 帯状四角形（ピンク）の描画
    def draw_error_messg_rectangle(self, resized_frame, set_height, set_width):
        """廃止予定
        """        
        self.resized_frame = resized_frame
        self.set_height = set_height
        self.set_width = set_width
        error_messg_rectangle_top: int  = int((self.set_height + 20) / 2)
        error_messg_rectangle_bottom : int = int((self.set_height + 120) / 2)
        error_messg_rectangle_left: int  = 0
        error_messg_rectangle_right : int = self.set_width
        cv2.rectangle(self.resized_frame, (error_messg_rectangle_left, error_messg_rectangle_top), (error_messg_rectangle_right, error_messg_rectangle_bottom), (255, 87, 243), cv2.FILLED)  # pink
        return error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom

    # drawオブジェクトを生成
    def  make_draw_object(self, frame):
        self.frame = frame
        draw = ImageDraw.Draw(Image.fromarray(self.frame))
        return draw

    def draw_error_messg_rectangle_messg(self, draw, error_messg_rectangle_position, error_messg_rectangle_messg, error_messg_rectangle_font):
        """廃止予定
        """
        self.draw = draw
        self.error_messg_rectangle_position = error_messg_rectangle_position
        self.error_messg_rectangle_messg = error_messg_rectangle_messg
        self.error_messg_rectangle_font = error_messg_rectangle_font
        draw.text(self.error_messg_rectangle_position, self.error_messg_rectangle_messg, fill=(255, 255, 255, 255), font = self.error_messg_rectangle_font)

    def return_fontpath(self, logger):
        # フォントの設定(フォントファイルのパスと文字の大きさ)
        operating_system: str  = system()
        fontpath: str = ''
        if (operating_system == 'Linux'):
            fontpath = "/usr/share/fonts/truetype/mplus/mplus-1mn-bold.ttf"
        elif (operating_system == 'Windows'):
                        # fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICR.TTC"
            fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICB.TTC"  ## bold体
        else:
            logger.info('オペレーティングシステムの確認が出来ません。システム管理者にご連絡ください')
        return fontpath

    def calculate_text_position(self, left,right,name,fontsize,bottom):
        self.left = left
        self.right = right
        self.name = name
        self.fontsize = fontsize
        self.bottom = bottom
        center = int((self.left + self.right)/2)
        chaCenter = int(len(self.name)/2)
        pos = center - (chaCenter* self.fontsize) - int(self.fontsize/2)
        position = (pos, self.bottom + (self.fontsize * 2))
        Unknown_position = (pos + self.fontsize, self.bottom + (self.fontsize * 2))
        return position, Unknown_position

    def draw_name(self, name,pil_img_obj, Unknown_position, font, p, tolerance, position):
        self.name = name
        self.pil_img_obj = pil_img_obj
        self.Unknown_position = Unknown_position
        self.font = font
        self.p = p
        self.tolerance = tolerance
        self.position = position
        local_draw_obj = ImageDraw.Draw(self.pil_img_obj)
        if self.name == 'Unknown':  ## nameがUnknownだった場合
            # draw.text(Unknown_position, '照合不一致', fill=(255, 255, 255, 255), font = font)
            local_draw_obj.text(self.Unknown_position, '　未登録', fill=(255, 255, 255, 255), font = self.font)
        else:  ## nameが既知の場合
            # if percentage > 99.0:
            if self.p < self.tolerance:
                # nameの描画
                local_draw_obj.text(self.position, self.name, fill=(255, 255, 255, 255), font = self.font)
            else:
                local_draw_obj.text(self.position, self.name, fill=(255, 87, 243, 255), font = self.font)
        return self.pil_img_obj

    # pil_img_objをnumpy配列に変換
    def convert_pil_img_to_ndarray(self, pil_img_obj):
        self.pil_img_obj = pil_img_obj
        frame = np.array(pil_img_obj)
        return frame

    def draw_text_for_name(self, logger, left,right,bottom,name, p,tolerance,pil_img_obj):
        self.logger = logger
        self.left = left
        self.right = right
        self.bottom = bottom
        self.name = name
        self.p = p
        self.tolerance = tolerance
        self.pil_img_obj = pil_img_obj
        fontpath = self.return_fontpath(logger)
        """TODO FONTSIZEハードコーティング訂正"""
        fontsize = 14
        font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
        # テキスト表示位置決定
        position, Unknown_position = self.calculate_text_position(self.left,self.right,self.name,fontsize,self.bottom)
        # nameの描画
        self.pil_img_obj = self.draw_name(self.name,self.pil_img_obj, Unknown_position, font, self.p, self.tolerance, position)
        # pil_img_objをnumpy配列に変換
        resized_frame = self.convert_pil_img_to_ndarray(self.pil_img_obj)
        return resized_frame

    # target_rectangleの描画
    def draw_target_rectangle(self, anti_spoof, rect01_png, rect01_NG_png, resized_frame,top,bottom,left,right,name):
        self.anti_spoof = anti_spoof,
        self.rect01_png = rect01_png
        self.rect01_NG_png = rect01_NG_png
        self.resized_frame = resized_frame
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        face_location = [self.top, self.right, self.bottom, self.left]
        self.name = name
        width_ratio: float
        height_ratio: float
        face_width: int
        face_height: int
        result = 'not_spoof'
        if self.anti_spoof[0] == True:
            # ELE: Equally Likely Events
            result, score, ELE = self.return_anti_spoof(self.resized_frame, face_location)
        if not self.name == 'Unknown' and result == 'not_spoof' and ELE is False:  ## self.nameが既知の場合
            face_width: int = self.right - self.left
            face_height: int = self.bottom - self.top
            orgHeight, orgWidth = self.rect01_png.shape[:2]
            width_ratio = 1.0 * (face_width / orgWidth)
            height_ratio = 1.0 * (face_height / orgHeight)
            self.rect01_png = cv2.resize(self.rect01_png, None, fx = width_ratio, fy = height_ratio)
            x1, y1, x2, y2 = self.left, self.top, self.left + self.rect01_png.shape[1], self.top + self.rect01_png.shape[0]
            try:
                self.resized_frame[y1:y2, x1:x2] = self.resized_frame[y1:y2, x1:x2] * (1 - self.rect01_png[:,:,3:] / 255) + \
                            self.rect01_png[:,:,:3] * (self.rect01_png[:,:,3:] / 255)
            except:
                # TODO: np.clip
                pass
        else:  ## self.nameがUnknownだった場合
            fx: float = 0.0
            face_width = self.right - self.left
            face_height = self.bottom - self.top
            # rect01_NG_png←ピンクのtarget_rectangle
            # rect01_NG_png: cv2.Mat = cv2.imread("images/rect01_NG.png", cv2.IMREAD_UNCHANGED)
            orgHeight, orgWidth = self.rect01_NG_png.shape[:2]
            width_ratio = float(1.0 * (face_width / orgWidth))
            height_ratio = 1.0 * (face_height / orgHeight)
            self.rect01_NG_png = cv2.resize(self.rect01_NG_png, None, fx = width_ratio, fy = height_ratio)
            x1, y1, x2, y2 = self.left, self.top, self.left + self.rect01_NG_png.shape[1], self.top + self.rect01_NG_png.shape[0]
            try:
                self.resized_frame[y1:y2, x1:x2] = self.resized_frame[y1:y2, x1:x2] * (1 - self.rect01_NG_png[:,:,3:] / 255) + \
                            self.rect01_NG_png[:,:,:3] * (self.rect01_NG_png[:,:,3:] / int(255))
            except:
                pass
        return self.resized_frame
