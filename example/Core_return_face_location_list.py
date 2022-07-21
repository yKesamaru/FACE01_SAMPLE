# For example, get the path.
import sys
sys.path.append("..")

# And we have to get some frames. All configured by config.ini.
from face01lib.video_capture import VidCap
import FACE01 as fg
next_frame_gen_obj = VidCap().frame_generator(fg.args_dict)

# Since this is example, number of frames is 50.
exec_times = 50

from face01lib.Core import Core

set_width = fg.args_dict['set_width']
set_height = fg.args_dict['set_height']
for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    print(Core().return_face_location_list(next_frame, set_width, set_height,0, 0.4))