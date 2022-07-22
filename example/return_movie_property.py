# For example, get the path.
import sys
sys.path.append("..")

# And we have to get some frames. All configured by config.ini.
from face01lib.video_capture import VidCap
import FACE01 as fg
next_frame_gen_obj = VidCap().frame_generator(fg.args_dict)

# Since this is example, number of frames is 50.
exec_times = 50

# For test, import open-cv
import cv2

set_width = fg.args_dict['set_width']
set_height = fg.args_dict['set_height']
VidCap_obj = VidCap()
vcap = VidCap().return_vcap(fg.args_dict['movie'])
for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    set_width,fps,height,width,set_height = \
        VidCap_obj.return_movie_property(set_width, vcap)
    print(
        'set_width: ', set_width, "\n",
        'set_height: ', set_height, "\n",
        'fps: ', fps, "\n"
    )