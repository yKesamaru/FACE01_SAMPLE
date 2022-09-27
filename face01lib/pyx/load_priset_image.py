#cython: language_level = 3
"""Load priset images"""
import os
from os.path import exists, isdir
from shutil import move

from typing import List, Tuple

import numpy as np
import numpy.typing as npt  
from .api import Dlib_api
from .logger import Logger


Dlib_api_obj = Dlib_api()


# Setup logger
import os.path
name: str = __name__
dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)

logger = Logger('info').logger(name, parent_dir)


def load_priset_image(
        self,
        RootDir: str,
        priset_face_imagesDir: str,
        upsampling: int = 0,
        jitters:int = 100,
        mode: str = 'hog',
        model: str = 'small'
    ) -> Tuple[List, List]:
    """Load face image from priset_face_images folder

    Args:
        RootDir (str): Root directory
        priset_face_imagesDir (str): Path to priset_face_images folder
        upsampling (int, optional): Value of upsampling. Defaults to 0.
        jitters (int, optional): Value of jitters. Defaults to 100.
        mode (str, optional): You can select from hog or cnn. Defaults to 'hog'.
        model (str, optional): You cannot modify this value.

    Returns:
        Tuple[List, List]: known_face_encodings_list, known_face_names_list
        - known_face_encodings_list
            - List of encoded many face images as ndarray
        - known_face_names_list
            - List of name which encoded as ndarray

    Example:
        >>> known_face_encodings, known_face_names = \\
        >>>     load_priset_image(
        >>>             self,
        >>>             self.conf_dict["RootDir"],
        >>>             self.conf_dict["priset_face_imagesDir"]
        >>>         )
    """    
    # Initialize
    known_face_names_list: List[str] = []
    known_face_encodings_list: List[npt.NDArray[np.float64]] = []
    new_files = []
    cnt: int = 1  # cnt：何番目のファイルかを表す変数
    cd: bool = False

    if cd == False:
        os.chdir(RootDir)
        cd = True

    logger.info("Loading npKnown.npz")

    ###################### npKnown.npzの有る無しで処理を分岐させる ######################
    # ============= npKnown.npzファイルが存在する場合の処理 ===============
    if exists("npKnown.npz"):

        # npKnown.npzの読み込みを行い、今までの全てのデータを格納する
        npKnown = np.load('npKnown.npz', allow_pickle = True)
        A, B = npKnown.files
        known_face_names_ndarray = npKnown[A]
        known_face_encodings_ndarray = npKnown[B]

        # ############ 各配列の整形（ndarray型からリスト型へ変換する） ############
        known_face_names_list = known_face_names_ndarray.tolist()

        list = []
        for i in known_face_encodings_ndarray:
            list.append(i)
            # for x in i:
            #     list.append(x)
        known_face_encodings_list = list
        # #########################################################################

        # priset_face_imagesフォルダ内の全てのファイル名を読み込む
        os.chdir(priset_face_imagesDir)
        # まずpriset_face_imagesDirのファイル名を全て得る
        for priset_face_image in os.listdir(priset_face_imagesDir):
            # <DEBUG>
            # if 'テスト' in priset_face_image:
            #     if not priset_face_image in known_face_names:
            #         print(priset_face_image)
            #     exit()
            # 関係ないファイルやフォルダは処理からとばす
            if priset_face_image == 'desktop.ini':
                continue
            if isdir(priset_face_image):
                continue
            if 'debug' in priset_face_image:
                continue
            # all_priset_face_images.append(priset_face_image)

            # =============== file名がnpKnownのキーに存在していない場合の処理 ===============
            if not priset_face_image in known_face_names_list:
                # priset_face_imageはknown_face_names配列にないから、new_fileに名前を変える
                new_file: str = priset_face_image

                new_file_face_image: npt.NDArray[np.uint8] = \
                    Dlib_api_obj.load_image_file(
                            new_file
                        )

                new_file_face_locations: List[Tuple[int,int,int,int]] = \
                    Dlib_api_obj.face_locations(
                            new_file_face_image,
                            upsampling,
                            mode
                        )

                # 顔検出できなかった場合hogからcnnへチェンジして再度顔検出する
                if len(new_file_face_locations) == 0:
                    if mode == 'hog':
                        logger.info("Face could not be detected. Temporarily switch to 'cnn' mode")
                        new_file_face_locations = Dlib_api_obj.face_locations(
                            new_file_face_image, upsampling, 'cnn')
                        # cnnでも顔検出できない場合はnoFaceフォルダへファイルを移動する
                        logger.info(f"{cnt} No face detected in registered face image {new_file}(CNN mode).  Move it to the 'noFace' folder")

                        try:
                            move(new_file, '../noFace/')
                        except:
                            os.remove('../noFace/' + new_file)
                            move(new_file, '../noFace/')

                        mode = 'hog'
                        logger.info('Back to HOG mode')

                # new_file顔画像のエンコーディング処理：array([encoding 配列])
                logger.info(f"{cnt} Encoding {new_file}")
                new_file_face_encodings = Dlib_api_obj.face_encodings(
                    new_file_face_image, new_file_face_locations, jitters, 'small')

                if len(new_file_face_encodings) > 1:  # 複数の顔が検出された時
                    logger.info(f"{cnt} Multiple faces detected in registered face image  {new_file}. Move it to noFace folder.")
                    move(new_file, '../noFace/')
                elif len(new_file_face_encodings) == 0:  # 顔が検出されなかった時
                    logger.info(f"{cnt} No face detected in registered face image {new_files}. Move it to noFace folder.")
                    move(new_file, '../noFace/' + new_file)

                # エンコーディングした顔画像だけ新しい配列に入れる
                known_face_names_list.append(new_file)
                known_face_encodings_list.append(new_file_face_encodings[0])

                cnt += 1

    # ============= npKnown.npzファイルが存在しない場合の処理 =============
    elif not exists("npKnown.npz"):
        os.chdir(priset_face_imagesDir)
        # まずpriset_face_imagesDirのファイル名を全て得る
        for priset_face_image_filename in os.listdir(priset_face_imagesDir):
            # 関係ないファイルやフォルダは処理からとばす
            if priset_face_image_filename == 'desktop.ini':  # desktop.iniは処理をとばす
                continue
            if isdir(priset_face_image_filename):  # フォルダの場合は処理をとばす
                continue
            if 'debug' in priset_face_image_filename:  # ファイル名にdebugを含む場合は処理をとばす
                continue

            # それぞれの顔写真について顔認証データを作成する
            priset_face_img = \
                Dlib_api_obj.load_image_file(
                        priset_face_image_filename
                    )

            priset_face_img_locations = \
                Dlib_api_obj.face_locations(
                        priset_face_img,
                        upsampling,
                        mode
                    )

            # 得られた顔データについて顔写真なのに顔が判別できない場合や複数の顔がある場合はcnnモードで再確認し、それでもな場合はnoFaceフォルダに移動する
            noFace_file = '../noFace/' + priset_face_image_filename

            if len(priset_face_img_locations) == 0 or len(priset_face_img_locations) > 1:
                
                if mode == 'hog':

                    logger.info('No face detected or multiple face detected. Temporarily switch to cnn mode')
                    
                    # CNNモードにて顔検出を行う
                    priset_face_img_locations = \
                        Dlib_api_obj.face_locations(
                                priset_face_img, upsampling,
                                'cnn'
                            )

                    # cnnでも顔検出できない場合はnoFaceフォルダへファイルを移動する
                    if len(priset_face_img_locations) == 0 or len(priset_face_img_locations) > 1:
                        
                        logger.info(f"{cnt} (CNN mode) Registered face image {priset_face_image_filename}, No face detected or multiple face detected. Move it to noFace folder.")
                        
                        if exists(noFace_file):
                            os.remove(noFace_file)
                        
                        move(priset_face_image_filename, '../noFace/')
                        
                        mode = 'hog'
                        
                        logger.info('Back to HOG mode')

            # 得られた顔データ（この場合は顔ロケーション）を元にエンコーディングする：array([encoding 配列])
            logger.info(f"{cnt} Encoding {priset_face_image_filename}")

            priset_face_image_encodings = \
                Dlib_api_obj.face_encodings(
                        priset_face_img,
                        priset_face_img_locations,
                        jitters,
                        'small'
                    )

            # エンコーディングした顔写真について複数顔や顔がない場合はnoFaceフォルダへ移動する
            if len(priset_face_image_encodings) > 1:  # 複数の顔が検出された時
                
                logger.info(f"{cnt} Multiple faces detected in registered face image {priset_face_image_filename}. Move it to noFace folder.")
                if exists(noFace_file):
                    os.remove(noFace_file)
                try:
                    move(priset_face_image_filename, '../noFace/')
                except:
                    pass
            elif len(priset_face_image_encodings) == 0:  # 顔が検出されなかった時
                logger.info(f"{cnt} No face detected in registered face image {priset_face_image_filename}. Move it to noFace folder.")
                if exists(noFace_file):
                    os.remove(noFace_file)
                try:
                    move(priset_face_image_filename, '../noFace/')
                except:
                    """TODO"""
                    pass

            # 配列に、名前やエンコーディングデータを要素として追加する
            # FACE01GRAPHICS本体の方では要素にndarrayを含むListを返り値として期待している(Dlib_api_obj APIにそう書いてある)
            known_face_names_list.append(priset_face_image_filename)
            known_face_encodings_list.append(priset_face_image_encodings[0])

            cnt += 1

    ###################### savezで保存 ######################
    os.chdir(RootDir)
    # print('debug_npKnown.npzを作成します')
    np.savez(
        'npKnown',
        # known_face_names_list_127 = known_face_names_list,
        known_face_names_list,
        # known_face_encodings_list_127 = known_face_encodings_list
        known_face_encodings_list
    )

    # ################### リスト型を返す ###################
    # <DEBUG>

    # list=[]
    # for i in known_face_encodings:  ## shape:(677, 1, 128)
    #     for x in i:
    #         list.append(x)
    # known_face_encodings = list
    return known_face_encodings_list, known_face_names_list

    # #################### 備考 ####################
    # 返り値のknown_face_encodingsと、npKnown.npzから読み込んだknown_face_encodingsとでは
    # もしかしたらデータ型とか？なにかが異なっているのかもしれない。

