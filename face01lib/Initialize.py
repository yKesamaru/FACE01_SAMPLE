#cython: language_level=3

import os
from configparser import ConfigParser
from datetime import datetime
from traceback import format_exc
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from face01lib.logger import Logger

from .Calc import Cal
from .load_priset_image import load_priset_image
# import .vidCap as video_capture  # so
from .LoadImage import LoadImage
from .video_capture import VidCap  # py


name: str = __name__
dir: str = os.path.dirname(__file__)
head, tail = os.path.split(dir)

logger = Logger().logger(name, head, 'info')



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
            Dict: args_dict
            Dictionary of initialized preferences
        
        Example:
            >>> args_dict: Dict =  Initialize().initialize(Initialize().configure())
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


    def configure(self) -> Dict:

        """Load conf.ini and return conf_dict dictionary

        Args:
            self
        Returns:
            Dict: conf_dict
        """        
        priset_face_imagesDir: str = f'{head}/priset_face_images/'

        try:
            conf: ConfigParser = ConfigParser()
            conf.read('config.ini', 'utf-8')
            # dict作成
            conf_dict: Dict = {
                'model' : conf.get('DEFAULT','model'),
                'headless' : conf.getboolean('MAIN','headless'),
                'anti_spoof' : conf.getboolean('MAIN','anti_spoof'),
                'output_debug_log' : conf.getboolean('MAIN','output_debug_log'),
                'set_width' : int(conf.get('SPEED_OR_PRECISE','set_width')),
                'similar_percentage' : float(conf.get('SPEED_OR_PRECISE','similar_percentage')),
                'jitters' : int(conf.get('SPEED_OR_PRECISE','jitters')),
                'priset_face_images_jitters' : int(conf.get('SPEED_OR_PRECISE','priset_face_images_jitters')),
                'priset_face_imagesDir' :priset_face_imagesDir,
                'upsampling' : int(conf.get('SPEED_OR_PRECISE','upsampling')),
                'mode' : conf.get('SPEED_OR_PRECISE','mode'),
                'frame_skip' : int(conf.get('SPEED_OR_PRECISE','frame_skip')),
                'number_of_people' : int(conf.get('SPEED_OR_PRECISE','number_of_people')),
                'use_pipe' : conf.getboolean('dlib','use_pipe'),
                'model_selection' : int(conf.get('dlib','model_selection')),
                'min_detection_confidence' : float(conf.get('dlib','min_detection_confidence')),
                'person_frame_face_encoding' : conf.getboolean('dlib','person_frame_face_encoding'),
                'same_time_recognize' : int(conf.get('dlib','same_time_recognize')),
                'set_area' : conf.get('INPUT','set_area'),
                'movie' : conf.get('INPUT','movie'),
                'user': conf.get('Authentication','user'),
                'passwd': conf.get('Authentication','passwd'),
                'rectangle' : conf.getboolean('DRAW_INFOMATION','rectangle'),
                'target_rectangle' : conf.getboolean('DRAW_INFOMATION','target_rectangle'),
                'draw_telop_and_logo' : conf.getboolean('DRAW_INFOMATION', 'draw_telop_and_logo'),
                'default_face_image_draw' : conf.getboolean('DRAW_INFOMATION', 'default_face_image_draw'),
                'show_overlay' : conf.getboolean('DRAW_INFOMATION', 'show_overlay'),
                'show_percentage' : conf.getboolean('DRAW_INFOMATION', 'show_percentage'),
                'show_name' : conf.getboolean('DRAW_INFOMATION', 'show_name'),
                'crop_face_image' : conf.getboolean('SAVE_FACE_IMAGE', 'crop_face_image'),
                'frequency_crop_image' : int(conf.get('SAVE_FACE_IMAGE','frequency_crop_image')),
                'crop_with_multithreading' : conf.getboolean('SAVE_FACE_IMAGE','crop_with_multithreading'),
                'Python_version': conf.get('system_check','Python_version'),
                'cpu_freq': conf.get('system_check','cpu_freq'),
                'cpu_count': conf.get('system_check','cpu_count'),
                'memory': conf.get('system_check','memory'),
                'gpu_check' : conf.getboolean('system_check','gpu_check'),
                'calculate_time' : conf.getboolean('DEBUG','calculate_time'),
                'show_video' : conf.getboolean('Scheduled_to_be_abolished','show_video'),
                'kaoninshoDir' :head,
            }
            return conf_dict
        except:
            logger.warning("config.ini 読み込み中にエラーが発生しました")
            logger.exception("conf_dictが正常に作成できませんでした")
            logger.warning("以下のエラーをシステム管理者様へお伝えください")
            logger.warning("-" * 20)
            logger.warning(format_exc(limit=None, chain=True))
            logger.warning("-" * 20)
            logger.warning("終了します")
            exit(0)
