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
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import dlib
import numpy as np
from PIL.Image import open
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from face01lib.logger import Logger
from traceback import format_exc
from sys import exit
from os.path import dirname

name = __name__
dir = dirname(__file__)
logger = Logger().logger(name, dir)
try:
    from face01lib import face_recognition_models
except Exception:
    logger.warning("modelのimportに失敗しました")
    logger.warning("システム管理者にお問い合わせください")
    logger.warning("-" * 20)
    logger.warning(format_exc(limit=None, chain=True))
    logger.warning("-" * 20)
    exit(0)


face_detector = dlib.get_frontal_face_detector()

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

class Dlib_api:
    def __init__(self) -> None:
        __author__ = 'Original code written by Adam Geitgey, modified by YOSHITSUGU KESAMARU'
        __email__ = 'y.kesamaru@tokai-kaoninsho.com'
        __version__ = '1.4.02'

    def _rect_to_css(self, rect: object) -> tuple:
        self.rect = rect
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return self.rect.top(), self.rect.right(), self.rect.bottom(), self.rect.left()


    def _css_to_rect(self, css):
        self.css = css
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(self.css[3], self.css[0], self.css[1], self.css[2])


    def _trim_css_to_bounds(self, css, image_shape):
        self.css = css
        self.image_shape = image_shape
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(self.css[0], 0), min(self.css[1], self.image_shape[1]), min(self.css[2], self.image_shape[0]), max(self.css[3], 0)


    def load_image_file(self, file, mode='RGB'):
        self.file = file
        self.mode = mode
        """
        Loads an image file (.jpg, .png, etc) into a numpy array

        :param file: image file name or file object to load
        :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
        :return: image contents as numpy array
        """
        im = open(self.file)
        if self.mode:
            im = im.convert(self.mode)
        return np.array(im)


    def _raw_face_locations(self, img: np.ndarray, number_of_times_to_upsample:int =0, model: str="cnn"):
        self.img = img
        self.number_of_times_to_upsample = number_of_times_to_upsample
        self.model = model
        """
        Returns an array of bounding boxes of human faces in a image

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


    def face_locations(self, img: np.ndarray, number_of_times_to_upsample: int=0, model: str="hog") -> list:
        self.img = img
        self.number_of_times_to_upsample = number_of_times_to_upsample
        self.model = model
        """
        Returns an array of bounding boxes of human faces in a image

        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        if self.model == "cnn":
            return [Dlib_api()._trim_css_to_bounds(Dlib_api()._rect_to_css(face.rect), self.img.shape) for face in Dlib_api()._raw_face_locations(self.img, self.number_of_times_to_upsample, "cnn")]
        else:
            return [Dlib_api()._trim_css_to_bounds(Dlib_api()._rect_to_css(face), self.img.shape) for face in Dlib_api()._raw_face_locations(self.img, self.number_of_times_to_upsample, self.model)]


    def _raw_face_landmarks(self, face_image, face_locations=None, model="small"):
        self.face_image = face_image
        self.face_locations = face_locations
        self. model = model
        if self.face_locations is None:
            self.face_locations = Dlib_api()._raw_face_locations(self.face_image)
        else:
            self.face_locations = [Dlib_api()._css_to_rect(face_location) for face_location in self.face_locations]
        return [pose_predictor_5_point(self.face_image, face_location) for face_location in self.face_locations]


    def face_encodings(self, face_image, known_face_locations=None, num_jitters=0, model="small") ->list:
        self.face_image = face_image
        self.known_face_locations = known_face_locations
        self.num_jitters = num_jitters
        self.model = model
        """
        Given an image, return the 128-dimension face encoding for each face in the image.

        :param face_image: The image that contains one or more faces (=small_frame)
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them. (=face_location_list)
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        """image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it.
        That is, centered and scaled essentially the same way."""
        """about coordinate order
        dlib: (Left, Top, Right, Bottom,)
        face_recognition: (top, right, bottom, left)
        see bellow
        https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
        """
        raw_landmarks = Dlib_api()._raw_face_landmarks(self.face_image, self.known_face_locations, self.model)
        # face_list = []
        # for raw_landmark_set in raw_landmarks:
        #     a = np.array(face_encoder.compute_face_descriptor(self.face_image, raw_landmark_set, self.num_jitters, 0.25))
        #     face_list.append(a)
        #     return face_list
        return [np.array(face_encoder.compute_face_descriptor(self.face_image, raw_landmark_set, self.num_jitters, 0.25)) for raw_landmark_set in raw_landmarks]
        # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
        # be closely cropped around the face. Setting larger padding values will result a looser cropping.
        # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
        # would triple it, and so forth.
        # see bellow
        # http://dlib.net/face_recognition.py.html
        """マルチスレッド化
        pool = ThreadPoolExecutor()
        # pool = ProcessPoolExecutor(max_workers=1)  # Error while calling cudaGetDevice(&the_device_id) in file /tmp/pip-install-983gqknr/dlib_66282e4ffadf4aa6965801c6f7ff7671/dlib/cuda/gpu_data.cpp:204. code: 3, reason: initialization error
        return [pool.submit(multithread, raw_landmark_set, self.face_image, self.num_jitters).result() for raw_landmark_set in raw_landmarks]
        """

    def multithread(self, raw_landmark_set, face_image, num_jitters):
        self.raw_landmark_set = raw_landmark_set
        self.face_image = face_image
        self.num_jitters = num_jitters
        return np.array(face_encoder.compute_face_descriptor(self.face_image, self.raw_landmark_set, self.num_jitters))


    def face_distance(self, face_encodings, face_to_compare):
        self.face_encodings = face_encodings
        self.face_to_compare = face_to_compare
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.

        :param faces: List of face encodings to compare (=small_frame)
        :param face_to_compare: A face encoding to compare against (=face_location_list)
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(self.face_encodings) == 0:
            return np.empty((2,0), dtype=np.float64)
        return np.linalg.norm(x=(self.face_encodings - self.face_to_compare), axis=1)


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
        face_distance_list = list(Dlib_api().face_distance(self.known_face_encodings, self.face_encoding_to_check))
        _min = min(face_distance_list)
        if _min <= self.tolerance:
            return [True if i == _min else False for i in face_distance_list]
        else:
            return [False] * len(face_distance_list)
