"""mediapipe for python, see bellow
https://github.com/google/mediapipe/tree/master/mediapipe/python
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
faceapi: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
"""
# from asyncio.log import logger
import mediapipe as mp
import numpy as np
import face01lib.api as faceapi
from traceback import format_exc
from datetime import datetime
from PIL import Image, ImageFile

import cv2
from collections import defaultdict
from face01lib.Calc import Cal
from face01lib.libdraw import LibDraw
from shutil import move
from concurrent.futures import ThreadPoolExecutor
from os.path import dirname
from face01lib.logger import Logger

name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir)
Cal().cal_specify_date(logger)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Core:
    def __init__(self) -> None:
        pass

    def mp_face_detection_func(self, resized_frame, model_selection=0, min_detection_confidence=0.4):
        face = mp.solutions.face_detection.FaceDetection(
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
                face_location_list.append((xtop,xright,xbottom,xleft))  # faceapi order

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
        return frame

    def check_compare_faces(self, logger, known_face_encodings, face_encoding, tolerance) -> list:
        self.logger = logger
        self.known_face_encodings = known_face_encodings
        self.face_encoding = face_encoding
        self.tolerance = tolerance
        try:
            matches = faceapi.compare_faces(self.known_face_encodings, self.face_encoding, self.tolerance)
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
        face_distances = faceapi.face_distance(self.args_dict["known_face_encodings"], self.face_encoding)  ## face_distances -> shape:(677,), face_encoding -> shape:(128,)
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

            concatenate_face_location_list.append((concatenated_xtop,concatenated_xright,concatenated_xbottom,concatenated_xleft))  # faceapi order
            detection_counter += 1
            """about coordinate order
            dlib: (Left, Top, Right, Bottom,)
            faceapi: (top, right, bottom, left)
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
        person_data_list = []
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
            """1.3.06でボトムエリア描画は廃止予定
            # bottom area描画
            if self.args_dict["bottom_area"]==True:
                # resized_frameの下方向に余白をつける
                self.resized_frame = cv2.copyMakeBorder(self.resized_frame, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
            """

            # 半透明処理（前半）
            if self.args_dict["show_overlay"]==True:
                overlay: cv2.Mat = self.resized_frame.copy()

            # テロップとロゴマークの合成
            if self.args_dict["draw_telop_and_logo"] == True:
                self.resized_frame =  Core().draw_telop(self.logger, self.args_dict["cal_resized_telop_nums"], self.resized_frame)
                self.resized_frame = Core().draw_logo(self.logger, self.args_dict["cal_resized_logo_nums"], self.resized_frame)

        # 顔座標算出
        if self.args_dict["use_pipe"] == True:
            face_location_list = Core().return_face_location_list(self.resized_frame, self.args_dict["set_width"], self.args_dict["set_height"], self.args_dict["model_selection"], self.args_dict["min_detection_confidence"])
        else:
            face_location_list = faceapi.face_locations(self.resized_frame, self.args_dict["upsampling"], self.args_dict["mode"])
        """face_location_list
        [(144, 197, 242, 99), (97, 489, 215, 371)]
        """

        # 顔がなかったら以降のエンコード処理を行わない
        if len(face_location_list) == 0:
            frame_datas_array = Core().make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
            return frame_datas_array

        # 顔が一定数以上なら以降のエンコード処理を行わない
        if len(face_location_list) >= self.args_dict["number_of_people"]:
            self.logger.info(f'{self.args_dict["number_of_people"]}人以上を検出しました')
            frame_datas_array = Core().make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
            return frame_datas_array

        frame_datas_array = Core().make_frame_datas_array(overlay, face_location_list, name,filename, top,right,bottom,left,percentage_and_symbol,person_data_list,frame_datas_array,self.resized_frame)
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
                        Core().return_concatenate_location_and_frame(resized_frame, face_location_list)
                    face_encodings = faceapi.face_encodings(concatenate_person_frame, concatenate_face_location_list, self.args_dict["jitters"], self.args_dict["model"])
                elif self.args_dict["use_pipe"] == True and  self.args_dict["person_frame_face_encoding"] == False:
                    face_encodings = faceapi.face_encodings(resized_frame, face_location_list, self.args_dict["jitters"], self.args_dict["model"])
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
                    face_encodings = faceapi.face_encodings(resized_frame, face_location_list, self.args_dict["jitters"], self.args_dict["model"])
            return face_encodings, self.frame_datas_array

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
                matches = Core().check_compare_faces(self.logger, self.args_dict["known_face_encodings"], face_encoding, self.args_dict["tolerance"])
                # 名前リスト(face_names)生成
                face_names = Core().return_face_names(self.args_dict, face_names, face_encoding,  matches, name)

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
                        pil_img_obj_rgb = Core().pil_img_rgb_instance(resized_frame)
                        if self.args_dict["crop_with_multithreading"] == True:
                            # """1.3.08 multithreading 9.05s
                            with ThreadPoolExecutor() as executor:
                                future = executor.submit(Core().make_crop_face_image, name, dis, pil_img_obj_rgb, top, left, right, bottom, self.GLOBAL_MEMORY['number_of_crops'], self.args_dict["frequency_crop_image"])
                                filename,number_of_crops, frequency_crop_image = future.result()
                            # """
                        else:
                            # """ORIGINAL: 1.3.08で変更 8.69s
                            filename,number_of_crops, frequency_crop_image = \
                                Core().make_crop_face_image(name, dis, pil_img_obj_rgb, top, left, right, bottom, self.GLOBAL_MEMORY['number_of_crops'], self.args_dict["frequency_crop_image"])
                            # """
                        self.GLOBAL_MEMORY['number_of_crops'] = 0
                    else:
                        self.GLOBAL_MEMORY['number_of_crops'] += 1

                # 描画系
                if self.args_dict["headless"] == False:
                    # デフォルト顔画像の描画
                    if p <= self.args_dict["tolerance"]:  # ディスタンスpがtolerance以下の場合
                        if self.args_dict["default_face_image_draw"] == True:
                            resized_frame = LibDraw().draw_default_face(self.logger, self.args_dict, name, resized_frame, number_of_people)
                            number_of_people += 1  # 何人目か
                            """DEBUG"""
                            # frame_imshow_for_debug(resized_frame)

                    # ピンクまたは白の四角形描画
                    if self.args_dict["rectangle"] == True:
                        if name == 'Unknown':  # プリセット顔画像に対応する顔画像がなかった場合
                            resized_frame = LibDraw().draw_pink_rectangle(resized_frame, top,bottom,left,right)
                        else:  # プリセット顔画像に対応する顔画像があった場合
                            resized_frame = LibDraw().draw_white_rectangle(self.args_dict["rectangle"], resized_frame, top, left, right, bottom)
                        
                    # パーセンテージ描画
                    if self.args_dict["show_percentage"]==True:
                        resized_frame = LibDraw().display_percentage(percentage_and_symbol,resized_frame, p, left, right, bottom, self.args_dict["tolerance"])
                        """DEBUG"""
                        # frame_imshow_for_debug(resized_frame)

                    # 名前表示と名前用四角形の描画
                    if self.args_dict["show_name"]==True:
                        resized_frame = LibDraw().draw_rectangle_for_name(name,resized_frame, left, right,bottom)
                        pil_img_obj= Image.fromarray(resized_frame)
                        resized_frame = LibDraw().draw_text_for_name(self.logger, left,right,bottom,name, p,self.args_dict["tolerance"],pil_img_obj)
                        """DEBUG"""
                        # frame_imshow_for_debug(resized_frame)

                    # target_rectangleの描画
                    if self.args_dict["target_rectangle"] == True:
                        resized_frame = LibDraw().draw_target_rectangle(self.args_dict["rect01_png"], resized_frame,top,bottom,left,right,name)
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
