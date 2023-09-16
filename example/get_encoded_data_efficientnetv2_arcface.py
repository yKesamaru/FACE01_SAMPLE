"""Example of simple face encode script with EfficientNetV2 Arcface model.

Summary:
    In this example, you can learn how to execute FACE01 as simple to use "efficientnetv2_arcface.onnx" model.
    This script loads a face image file and outputs the feature vector of the face image.
    The feature vector is 512-dimensional vector, which is from `efficientnetv2_arcface.onnx` model.

Example:
    .. code-block:: bash
    
        python3 example/get_encoded_data_efficientnetv2_arcface.py
        
Source code:
    `get_encoded_data_efficientnetv2_arcface.py <../example/get_encoded_data_efficientnetv2_arcface.py>`_
"""
# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)

from typing import Dict, List

import cv2
import dlib
import numpy.typing as npt

from face01lib.api import Dlib_api
from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.utils import Utils

if __name__ == '__main__':
        
    api_obj = Dlib_api()
    utils_obj = Utils()

    # Initialize
    CONFIG: Dict =  Initialize('EFFICIENTNETV2_ARCFACE_MODEL', 'info')._configure()
    # Set up logger
    logger = Logger(CONFIG['log_level']).logger(__file__, CONFIG['RootDir'])
    """Initialize and Setup logger.
    When coding a program that uses FACE01, code `initialize` and `logger` first.
    This will read the configuration file `config.ini` and log errors etc.
    """

    face_path = 'example/img/麻生太郎_default.png'

    # Load '麻生.png' in the example/img folder
    img = dlib.load_rgb_image(face_path)

    # Convert face_image from BGR to RGB
    face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations: List = api_obj.face_locations(img, mode="cnn")
    face_encodings: List[npt.NDArray] = api_obj.face_encodings(
            deep_learning_model=1,
            resized_frame=img,
            face_location_list=face_locations
        )

    print(face_encodings)
