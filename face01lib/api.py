import inspect
from functools import lru_cache
from typing import Dict, List, Tuple

import dlib
import numpy as np
import PIL.Image
from numba import f8, i8, njit, typeof
from numba.extending import overload
from numba.typed import List
from PIL import ImageFile

NUMBA_CAPTURED_ERRORS="new_style"
try:
    import face_recognition_models
except Exception:
    print("例外エラーが発生しました:<api.py>\n")
    print("システム管理者にお問い合わせください")
    quit()
"""to refer, see bellow
https://github.com/davisking/dlib
https://github.com/davisking/dlib-models
https://github.com/ageitgey/face_recognition
"""
"""about coordinate order
dlib: (Left, Top, Right, Bottom,)
face_recognition: (top, right, bottom, left)
see bellow
https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
"""

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

# predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
# pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


# @lru_cache(maxsize = 128)
# @njit(cache=True)
def _rect_to_css(rect: object) ->Tuple[int,int,int,int]:
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


# @njit(cache=True)
# @lru_cache(maxsize = 128)
def _raw_face_locations(img: np.ndarray, number_of_times_to_upsample:int =0, model: str="cnn"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)


# @njit(cache=True)
def face_locations(img: np.ndarray, number_of_times_to_upsample: int=0, model: str="hog") -> List[Tuple]:
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]


def _raw_face_locations_batched(images, number_of_times_to_upsample=1, batch_size=128):
    """
    Returns an 2d array of dlib rects of human faces in a image using the cnn face detector

    :param img: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)


"""NOT USE
def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
    def convert_cnn_detections_to_css(detections):
        return [_trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape) for face in detections]

    raw_detections_batched = _raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

    return list(map(convert_cnn_detections_to_css, raw_detections_batched))
"""

# @njit()
def _raw_face_landmarks(face_image, face_locations=None, model="small"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    # pose_predictor = pose_predictor_5_point
    return [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]


"""NO USE
def face_landmarks(face_image, face_locations=None, model="large"):
    # Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
    # :param face_image: image to search
    # :param face_locations: Optionally provide a list of face locations to check.
    # :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    # :return: A list of dicts of face feature locations (eyes, nose, etc)
    
    landmarks = _raw_face_landmarks(face_image, face_locations, model)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    if model == 'large':
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]
    elif model == 'small':
        return [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks_as_tuples]
    else:
        raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")
"""


# compu_face = face_encoder.compute_face_descriptor()
# @njit()
def face_encodings( person_frame_face_encoding, face_image, known_face_locations=None, num_jitters=0, model="small") ->List[np.ndarray]:
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces (=small_frame)
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them. (=face_location_list)
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    if  person_frame_face_encoding == True:
        """image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it.
        That is, centered and scaled essentially the same way."""
        """about coordinate order
        dlib: (Left, Top, Right, Bottom,)
        face_recognition: (top, right, bottom, left)
        see bellow
        https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
        """
        # known_face_locations=[(0, face_image.shape[0], face_image.shape[1], 0)]  # face_recognition order
        # known_face_locations=[(0, 0, face_image.shape[0], face_image.shape[1])]
        raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
        return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
    else:
        raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
        return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


# @njit(cache=False)
# @njit
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare (=small_frame)
    :param face_to_compare: A face encoding to compare against (=face_location_list)
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((2,0), dtype=np.float64)
    return np.linalg.norm(x=(face_encodings - face_to_compare), axis=1)


# @njit('reflected list(Tuple(float64, bool))<iv=None>')
# @njit(list(bool)((f8[:,:], f8[:,:], f8)), cache=False)
# @njit
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    face_distance_list = list(face_distance(known_face_encodings, face_encoding_to_check))
    _min = min(face_distance_list)
    if _min <= tolerance:
        return [True if i == _min else False for i in face_distance_list]
    else:
        return [False] * len(face_distance_list)
    # return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
