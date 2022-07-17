"""mediapipe for python, see bellow
https://github.com/google/mediapipe/tree/master/mediapipe/python
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
faceapi: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/faceapi.py
"""
import mediapipe as mp
import numpy as np
import face01lib.api as faceapi
from traceback import format_exc
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