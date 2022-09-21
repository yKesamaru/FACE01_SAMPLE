#cython: language_level=3

from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .Calc import Cal
from .load_priset_image import load_priset_image
# import .vidCap as video_capture  # so
from .LoadImage import LoadImage
from .video_capture import VidCap  # py


class Initialize:
    def __init__(self) -> None:
        pass

    # Initialize variables, load images
    def initialize(
            self,
            conf_dict: Dict
        ) -> Dict:

        """Initialize values

        Returns:
            Dict: Dictionary of initialized preferences
        """        

        self.conf_dict: Dict = conf_dict
        headless: bool = self.conf_dict["headless"]
        
        known_face_encodings: List[npt.NDArray[np.float64]]
        known_face_names: List[str]
        known_face_encodings, known_face_names = \
            load_priset_image(
                    self,
                    self.conf_dict["kaoninshoDir"],
                    self.conf_dict["priset_face_imagesDir"]
                )

        # set_width,fps,height,width,set_height
        set_width : int
        fps : int
        height : int
        width : int
        set_height : int
        set_width,fps,height,width,set_height = \
            VidCap().return_movie_property(
                    self.conf_dict["set_width"],
                    VidCap().return_vcap(self.conf_dict["movie"])
                )
        
        tolerance: float = Cal().to_tolerance(self.conf_dict["similar_percentage"])

        LoadImage_obj: LoadImage = LoadImage(
                headless,
                self.conf_dict
            )
        
        rect01_png : npt.NDArray[np.uint8]
        rect01_NG_png : List[npt.NDArray[np.uint8]]
        rect01_REAL_png : npt.NDArray[np.uint8]
        rect01_SPOOF_png : npt.NDArray[np.uint8]
        rect01_CANNOT_DISTINCTION_png : npt.NDArray[np.uint8]
        resized_telop_image : npt.NDArray[np.uint8]
        cal_resized_telop_nums : Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]
        resized_logo_image : npt.NDArray[np.uint8]
        cal_resized_logo_nums : Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]
        load_unregistered_face_image : bool
        telop_image : npt.NDArray[np.uint8]
        logo_image : npt.NDArray[np.uint8]
        unregistered_face_image : npt.NDArray[np.uint8]

        rect01_png, rect01_NG_png, rect01_REAL_png, rect01_SPOOF_png, rect01_CANNOT_DISTINCTION_png, resized_telop_image, cal_resized_telop_nums, resized_logo_image, \
            cal_resized_logo_nums, load_unregistered_face_image, telop_image, logo_image, unregistered_face_image = \
            LoadImage_obj.LI(set_height, set_width)

        # 日付時刻算出
        date = datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒

        # 辞書作成
        init_dict: Dict 
        if headless == False:
            init_dict = {
                'known_face_encodings': known_face_encodings,
                'known_face_names': known_face_names,
                'date': date,
                'rect01_png': rect01_png,
                'rect01_NG_png': rect01_NG_png,
                'rect01_REAL_png': rect01_REAL_png,
                'rect01_SPOOF_png': rect01_SPOOF_png,
                'rect01_CANNOT_DISTINCTION_png': rect01_CANNOT_DISTINCTION_png,
                'telop_image': telop_image,
                'resized_telop_image': resized_telop_image,
                'cal_resized_telop_nums': cal_resized_telop_nums,
                'logo_image': logo_image,
                'cal_resized_logo_nums': cal_resized_logo_nums,
                'unregistered_face_image': unregistered_face_image,
                'height': height,
                'width': width,
                'set_height': set_height,
                'tolerance': tolerance,
                'default_face_image_dict': {}
            }
        elif headless == True:
            init_dict = {
                'known_face_encodings': known_face_encodings,
                'known_face_names': known_face_names,
                'date': date,
                'height': height,
                'width': width,
                'set_height': set_height,
                'tolerance': tolerance,
                'default_face_image_dict': {}
            }

        # 辞書結合
        args_dict: Dict = {**init_dict, **self.conf_dict}

        # ヘッドレス実装
        if headless == True:
            args_dict['rectangle'] = False
            args_dict['target_rectangle'] = False
            args_dict['show_video'] = False
            args_dict['default_face_image_draw'] = False
            args_dict['show_overlay'] = False
            args_dict['show_percentage'] = False
            args_dict['show_name'] = False
            args_dict['draw_telop_and_logo'] = False
            args_dict['person_frame_face_encoding'] = False
            args_dict['headless'] = True

        return args_dict
