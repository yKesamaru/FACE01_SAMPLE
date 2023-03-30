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
for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    resized_frame = VidCap_obj.resize_frame(set_width, set_height, next_frame)
    cv2.imshow('resized_frame', resized_frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()