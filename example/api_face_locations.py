# For example, get the path.
import sys
sys.path.append("..")

# And we have to get some frames. All configured by config.ini.
from face01lib.video_capture import VidCap
import FACE01 as fg
next_frame_gen_obj = VidCap().frame_generator(fg.args_dict)

# Since this is example, number of frames is 50.
exec_times = 50

from face01lib.api import Dlib_api
Dlib_api_obj = Dlib_api()

# configure
number_of_times_to_upsample = 0
model = 'cnn'

for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    if model == 'cnn':
        print( [Dlib_api_obj._trim_css_to_bounds(Dlib_api_obj._rect_to_css(face.rect), next_frame.shape) for face in Dlib_api_obj._raw_face_locations(next_frame, number_of_times_to_upsample, model)])
    else:
        print( [Dlib_api_obj._trim_css_to_bounds(Dlib_api_obj._rect_to_css(face), next_frame.shape) for face in Dlib_api_obj._raw_face_locations(next_frame, number_of_times_to_upsample, model)])
