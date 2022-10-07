__model__ = 'Original model create by Prokofev Kirill, modified by PINT'
__URL__ = 'https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3'

import cv2
import numpy as np
import onnxruntime

from face01lib.Core import Core
Core_obj = Core()
from face01lib import face_recognition_models
anti_spoof_model = face_recognition_models.anti_spoof_model_location()
onnx_session = onnxruntime.InferenceSession(anti_spoof_model)

class Anti_spoof:
    def return_anti_spoof(self, frame, face_location_list):
        self.frame = frame
        self.face_location_list = face_location_list
        face_image = Core_obj.return_face_image(self.frame, self.face_location_list)
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
        if as_index == 0:  # (255, 0, 0)
            # print(f"red: {result}")
            return 'spoof'
        if as_index == 1:
            # print(f"blue: {result}")  # (0, 0, 255)
            return 'not_spoof'

