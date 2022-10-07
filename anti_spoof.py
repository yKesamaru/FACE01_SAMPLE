__Author__ = 'Original code written by Kazuhito Takahashi "FaceDetection-Anti-Spoof-Demo", modified by YOSHITSUGU KESAMARU'
__URL1__ = 'https://github.com/Kazuhito00/FaceDetection-Anti-Spoof-Demo/blob/main/demo_anti_spoof.py'
__model__ = 'Original model create by Prokofev Kirill, modified by PINT'
__URL2__ = 'https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3'

import cv2
import numpy as np
import onnxruntime


from face01lib.video_capture import VidCap
VidCap_obj = VidCap()
import FACE01 as fg
next_frame_gen_obj = VidCap_obj.frame_generator(fg.args_dict)
from face01lib.Core import Core
Core_obj = Core()
from face01lib import face_recognition_models
anti_spoof_model = face_recognition_models.anti_spoof_model_location()
onnx_session = onnxruntime.InferenceSession(anti_spoof_model)

class Anti_spoof:
    def return_anti_spoof(self, frame_datas_array):

# Since this is example, number of frames is 50.
exec_times = 50

for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    for dict in Core_obj.frame_pre_processing(fg.logger, fg.args_dict, next_frame):
        img = dict["img"]
        face_location_list = dict["face_location_list"]
        face_image = Core_obj.return_face_image(img, face_location_list)
        # VidCap_obj.frame_imshow_for_debug(face_image)  # DEBUG

        # 前処理:リサイズ, BGR->RGB, 標準化, 成形, float32キャスト
        input_image = cv2.resize(face_image, dsize=(128, 128))
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # VidCap_obj.frame_imshow_for_debug(input_image)  # DEBUG

        input_image = input_image.transpose(2, 0, 1).astype('float32')
        input_image = input_image.reshape(-1, 3, 128, 128)

        # 推論
        input_name = onnx_session.get_inputs()[0].name
        result = onnx_session.run(None, {input_name: input_image})

        # 後処理
        result = np.array(result)
        result = np.squeeze(result)

        as_index = np.argmax(result)
        if as_index == 0:
            print("red")
        if as_index == 1:
            print("blue")
