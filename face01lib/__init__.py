__author__ = 'Original code written by Adam Geitgey, modified by YOSHITSUGU KESAMARU'
__email__ = 'y.kesamaru@tokai-kaoninsho.com'
__version__ = '1.3.10'

from .api import load_image_file, face_locations, face_encodings, compare_faces, face_distance
from .cpp_cal_angle_coodinate import return_tuple
# from .sample import size