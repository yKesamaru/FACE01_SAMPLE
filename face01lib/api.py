from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple

import dlib
import numpy as np
import PIL.Image
from PIL import ImageFile

# with ThreadPoolExecutor() as th:
#     th.submit(main())


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

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


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


def _raw_face_landmarks(face_image, face_locations=None, model="small"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=0, model="small") ->List[np.ndarray]:
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
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    # face_list = []
    # for raw_landmark_set in raw_landmarks:
    #     a = np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters, 0.25))
    #     face_list.append(a)
    #     return face_list
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters, 0.25)) for raw_landmark_set in raw_landmarks]
    # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
    # be closely cropped around the face. Setting larger padding values will result a looser cropping.
    # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
    # would triple it, and so forth.
    # see bellow
    # http://dlib.net/face_recognition.py.html
    """マルチスレッド化
    pool = ThreadPoolExecutor()
    # pool = ProcessPoolExecutor(max_workers=1)  # Error while calling cudaGetDevice(&the_device_id) in file /tmp/pip-install-983gqknr/dlib_66282e4ffadf4aa6965801c6f7ff7671/dlib/cuda/gpu_data.cpp:204. code: 3, reason: initialization error
    return [pool.submit(multithread, raw_landmark_set, face_image, num_jitters).result() for raw_landmark_set in raw_landmarks]
    """

def multithread(raw_landmark_set, face_image, num_jitters):
    return np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))


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


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """TODO
    compare_facesとreturn_face_namesに冗長がある"""
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
