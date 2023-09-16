"""License for the Code.

Copyright Owner: Yoshitsugu Kesamaru
Please refer to the separate license file for the license of the code.
"""

"""Example program to create npKnown.npz file.

Summary:
    In this example you can learn how to make npKnown.npz file.

Example:
    .. code-block:: bash
    
        python3 example/make_npKnown_file.py
        
Source code:
    `make_npKnown_file.py <../example/make_npKnown_file.py>`_
"""
# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)

from face01lib.load_preset_image import LoadPresetImage

if __name__ == '__main__':
    load_preset_image_obj = LoadPresetImage()

    root_dir: str = "/media/terms/2TB_Movie/face_data_backup/less_20_face_images/tmp2"
    # root_dir: str = "/media/terms/2TB_Movie/face_data_backup/data"
    # root_dir: str = "/home/terms/ドキュメント/find_similar_faces/test"

    sub_dir_list: list = os.listdir(root_dir)
    for sub_dir in sub_dir_list:
        sub_dir = os.path.join(root_dir, sub_dir)
        load_preset_image_obj.load_preset_image(
            deep_learning_model=1,
            RootDir=sub_dir,  # npKnown.npzを作成するディレクトリ
            preset_face_imagesDir=sub_dir  # 顔画像が格納されているディレクトリ
        )
        