__author__ = 'Original code written by Adam Geitgey, modified by YOSHITSUGU KESAMARU'
__email__ = 'y.kesamaru@tokai-kaoninsho.com'
__version__ = '1.4.04'

from pkg_resources import resource_filename

class Dlib_models:
    # DATA_PATH = pkg_resources.resource_filename('<package name>', 'data/')
    def pose_predictor_five_point_model_location(self):
        return resource_filename(__name__, "shape_predictor_5_face_landmarks.dat")

    def face_recognition_model_location(self):
        return resource_filename(__name__, "dlib_face_recognition_resnet_model_v1.dat")

    def cnn_face_detector_model_location(self):
        return resource_filename(__name__, "mmod_human_face_detector.dat")

    def anti_spoof_model_location(self):
        return resource_filename(__name__, "model_float32.onnx")