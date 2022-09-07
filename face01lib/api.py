"""to refer, see bellow
https://github.com/davisking/dlib
https://github.com/davisking/dlib-models
https://github.com/ageitgey/face_recognition
"""
"""copyright
This code is based on 'face_recognition' written by Adam Geitgey,
and modified by Yoshitsugu Kesamaru.
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
face_recognition: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
"""


"""DEBUG: MEMORY LEAK
from face01lib.memory_leak import Memory_leak
m = Memory_leak(limit=2, key_type='traceback', nframe=20)
m.memory_leak_analyze_start()
"""

import cython
import numpy as np
from typing import List, Tuple, Union
import numpy.typing as npt  # See [](https://discuss.python.org/t/how-to-type-annotate-mathematical-operations-that-supports-built-in-numerics-collections-and-numpy-arrays/13509)
import dlib
from PIL import ImageFile
from PIL.Image import open

ImageFile.LOAD_TRUNCATED_IMAGES = True
import traceback
from os.path import dirname
from sys import exit
from traceback import format_exc

from memory_profiler import profile  # @profile()

from face01lib.logger import Logger


name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir, None)


try:
    from face01lib.models import Dlib_models
except Exception:
    logger.error("modelのimportに失敗しました")
    logger.error("-" * 20)
    logger.error(format_exc(limit=None, chain=True))
    logger.error("-" * 20)
    exit(0)


face_detector = dlib.get_frontal_face_detector()  # type: ignore

predictor_5_point_model = Dlib_models().pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)  # type: ignore

cnn_face_detection_model = Dlib_models().cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)  # type: ignore

face_recognition_model = Dlib_models().face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)  # type: ignore


class Dlib_api:
    __author__ = 'Original code written by Adam Geitgey, modified by YOSHITSUGU KESAMARU'
    __email__ = 'y.kesamaru@tokai-kaoninsho.com'
    __version__ = 'v0.0.1'

    # def __init__(self) -> None:

    def _rect_to_css(self, rect: object) -> Tuple[int,int,int,int]:
        self.rect = rect
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order.
        This method used only `use_pipe = False`.

        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left()


    def _css_to_rect(self, css: Tuple[int,int,int,int]) -> dlib.rectangle:  # type: ignore
        self.css: Tuple[int,int,int,int] = css
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: <class '_dlib_pybind11.rectangle'>
        """
        return dlib.rectangle(self.css[3], self.css[0], self.css[1], self.css[2])  # type: ignore


    def _trim_css_to_bounds(
            self,
            css: Tuple[int,int,int,int],
            image_shape: Tuple[int,int,int]
        ) -> Tuple[int,int,int,int]:
        self._trim_css_to_bounds_css: Tuple[int,int,int,int] = css
        self.image_shape: Tuple[int, int, int] = image_shape
        """
        cssを境界に沿ってtrimする
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
        This method used only `use_pipe = False`.

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return (
            max(self._trim_css_to_bounds_css[0], 0),
            min(self._trim_css_to_bounds_css[1],self.image_shape[1]),
            min(self._trim_css_to_bounds_css[2], self.image_shape[0]),
            max(self._trim_css_to_bounds_css[3], 0)
        )


    """NOT USE"""
    # def load_image_file(self, file, mode='RGB'):
    #     self.file = file
    #     self.mode = mode
    #     """
    #     Loads an image file (.jpg, .png, etc) into a numpy array

    #     :param file: image file name or file object to load
    #     :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    #     :return: image contents as numpy array
    #     """
    #     im = open(self.file)
    #     if self.mode:
    #         im = im.convert(self.mode)
    #     return np.array(im)


    def _raw_face_locations(
            self,
            img: np.ndarray,
            number_of_times_to_upsample:int = 0,
            model: str="cnn"
        ):
        self.img: np.ndarray = img
        self.number_of_times_to_upsample: int = number_of_times_to_upsample
        self.model: str = model
        """
        Returns an array of bounding boxes of human faces in a image.
        This method used only `use_pipe = False`.

        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of dlib 'rect' objects of found face locations
        """
        if self.model == "cnn":
            return cnn_face_detector(self.img, self.number_of_times_to_upsample)
        else:
            return face_detector(self.img, self.number_of_times_to_upsample)


    # @profile()
    def face_locations(
        self,
        img: np.ndarray,
        number_of_times_to_upsample: int = 0,
        model: str="hog"
        ) -> list:
        
        self.img = img
        self.number_of_times_to_upsample = number_of_times_to_upsample
        self.model = model
        """
        Returns an array of bounding boxes of human faces in a image.
        This method used only `use_pipe = False`.

        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        if self.model == "cnn":
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), self.img.shape) for face in self._raw_face_locations(self.img, self.number_of_times_to_upsample, "cnn")]
        else:
            return [self._trim_css_to_bounds(self._rect_to_css(face), self.img.shape) for face in self._raw_face_locations(self.img, self.number_of_times_to_upsample, self.model)]


    def _return_raw_face_landmarks(
        self,
        resized_frame: np.ndarray,
        face_location_list: List = [],
        model: str ="small"
    ) -> List:
        self.resized_frame: np.ndarray = resized_frame
        self.raw_face_location_list: List = face_location_list
        self. model = model

        new_face_location_list: List[dlib.rectangle[int,int,int,int]] = []  # type: ignore
        raw_face_location: Tuple[int,int,int,int]

        for raw_face_location in self.raw_face_location_list:
            new_face_location_list.append(self._css_to_rect(raw_face_location))
        
        raw_face_landmarks: List[dlib.rectangle] = []  # type: ignore
        new_face_location: dlib.rectangle  # type: ignore

        for new_face_location in new_face_location_list:
            raw_face_landmarks.append(
                pose_predictor_5_point(self.resized_frame, new_face_location)
            )
        
        return raw_face_landmarks


    # @profile()
    def face_encodings(
        self,
        resized_frame: np.ndarray,
        face_location_list: List = [],  # face_location_listの初期値は[]とする。
        num_jitters: int = 0,
        model: str = "small"
    ) -> List[np.ndarray]:

        """
        Given an image, return the 128-dimension face encoding for each face in the image.

        :param resized_frame: The image that contains one or more faces (=small_frame)
        :param face_location_list: Optional - the bounding boxes of each face if you already know them. (=face_location_list)
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        
        Image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it.
        That is, centered and scaled essentially the same way.
        
        About coordinate order
        dlib: (Left, Top, Right, Bottom,)
        face_recognition: (top, right, bottom, left)
        see bellow
        https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
        """
        self.face_encodings_resized_frame: np.ndarray = resized_frame
        self.face_location_list: List  = face_location_list
        self.num_jitters: int =  num_jitters
        self.face_encodings_model: str = model
        _PADDING: float = 0.25
        face_encodings: List[np.ndarray] = []

        if len(self.face_location_list) > 0:
            raw_face_landmarks: List = self._return_raw_face_landmarks(
                self.face_encodings_resized_frame,
                self.face_location_list,
                self.face_encodings_model
            )

            raw_face_landmark: dlib.rectangle  # type: ignore

            for raw_face_landmark in raw_face_landmarks:
                face_landmark_ndarray: np.ndarray = np.array(
                    face_encoder.compute_face_descriptor(
                        self.face_encodings_resized_frame,
                        raw_face_landmark,
                        self.num_jitters,
                        _PADDING
                    )
                )
                face_encodings.append(face_landmark_ndarray)

        return face_encodings
        """
        [compute_face_descriptor](https://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html?m=0&commentPage=2)
        Davis King said...
        The landmarks are only used to align the face before the DNN extracts 
        the face descriptor. How many landmarks you use doesn't really matter.
        """

        # TODO: 余白の0.25 (PADDING)
        # return [np.array(face_encoder.compute_face_descriptor(self.face_encodings_resized_frame, raw_landmark_set, self.num_jitters, 0.25)) for raw_landmark_set in raw_landmarks]
        # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
        # be closely cropped around the face. Setting larger padding values will result a looser cropping.
        # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
        # would triple it, and so forth.
        # see bellow
        # http://dlib.net/face_recognition.py.html
        """マルチスレッド化
        pool = ThreadPoolExecutor()
        # pool = ProcessPoolExecutor(max_workers=1)  # Error while calling cudaGetDevice(&the_device_id) in file /tmp/pip-install-983gqknr/dlib_66282e4ffadf4aa6965801c6f7ff7671/dlib/cuda/gpu_data.cpp:204. code: 3, reason: initialization error
        return [pool.submit(multithread, raw_landmark_set, self.face_encodings_resized_frame, self.num_jitters).result() for raw_landmark_set in raw_landmarks]
        """


    """NOT USED"""
    # def multithread(self, raw_landmark_set, resized_frame, num_jitters):
    #     self.raw_landmark_set = raw_landmark_set
    #     self.resized_frame = resized_frame
    #     self.num_jitters = num_jitters
    #     return np.array(face_encoder.compute_face_descriptor(self.resized_frame, self.raw_landmark_set, self.num_jitters))


    # @profile()
    def face_distance(
            self,
            face_encodings: List[npt.NDArray[np.float64]],
            face_to_compare: npt.NDArray[np.float64]
            # face_encodings: List[np.ndarray],
            # face_to_compare: np.ndarray
        ) -> np.ndarray:
        """
        Given a list of face encodings, compare them to a known face encoding and get a 
        euclidean distance for each comparison face.
        The distance tells you how similar the faces are.

        :param face_encodings: List of face encodings to compare (=small_frame)
        :param face_to_compare: A face encoding to compare against (=face_location_list)
        :return: A numpy ndarray with the distance for each face 
                          in the same order as the 'faces' array
        """
        # self.face_encodings = face_encodings
        # self.face_to_compare = face_to_compare

        if len(face_encodings) == 0:
            return np.empty((2,2,3), dtype=np.uint8)
        
        return np.linalg.norm(x=(face_encodings - face_to_compare), axis=1)


    # @profile()
    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):
        self.known_face_encodings = known_face_encodings
        self.face_encoding_to_check = face_encoding_to_check
        self.tolerance = tolerance
        """TODO
        compare_facesとreturn_face_namesに冗長がある"""
        """
        Compare a list of face encodings against a candidate encoding to see if they match.

        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        face_distance_list = list(self.face_distance(self.known_face_encodings, self.face_encoding_to_check))
        _min = min(face_distance_list)
        if _min <= self.tolerance:
            return [True if i == _min else False for i in face_distance_list]
        else:
            return [False] * len(face_distance_list)


"""DEBUG: MEMORY LEAK
m.memory_leak_analyze_stop()
"""


"""Reference
- Max-Margin Object Detection(MMOD)
    - https://blog.chowagiken.co.jp/entry/2019/06/28/OpenCV%E3%81%A8dlib%E3%81%AE%E9%A1%94%E6%A4%9C%E5%87%BA%E6%A9%9F%E8%83%BD%E3%81%AE%E6%AF%94%E8%BC%83
    - https://github.com/davisking/dlib-models
"""