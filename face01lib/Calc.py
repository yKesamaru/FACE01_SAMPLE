#cython: language_level=3
"""A module that performs various calculations
Calculation results are output to log"""

from datetime import datetime
from os.path import dirname
from time import perf_counter
from typing import Tuple
from functools import wraps
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFile, ImageFont

from .logger import Logger

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Cal:

    HANDLING_FRAME_TIME: float
    HANDLING_FRAME_TIME_FRONT: float
    HANDLING_FRAME_TIME_REAR: float
    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, log_level: str = 'info') -> None:
        # Setup logger: common way
        self.log_level: str = log_level
        import os.path
        name: str = __name__
        dir: str = os.path.dirname(__file__)
        parent_dir, _ = os.path.split(dir)

        self.logger = Logger(self.log_level).logger(name, parent_dir)
    

    @staticmethod
    def Measure_processing_time(
            HANDLING_FRAME_TIME_FRONT,
            HANDLING_FRAME_TIME_REAR
        ) -> float:
        """Measurement of processing time (calculation) and output to log

        Args:
            HANDLING_FRAME_TIME_FRONT (float): First half point
            HANDLING_FRAME_TIME_REAR (float): Second half point
        """        
        HANDLING_FRAME_TIME = \
            (HANDLING_FRAME_TIME_REAR - HANDLING_FRAME_TIME_FRONT)  ## 小数点以下がミリ秒
        
        # logger.info(f'Processing time: {round(HANDLING_FRAME_TIME, 3)}[Sec]')

        return HANDLING_FRAME_TIME

    @staticmethod
    def Measure_processing_time_forward() -> float:
        """Measurement of processing time (first half)

        Returns:
            float: First half point
        """        
        HANDLING_FRAME_TIME_FRONT = perf_counter()
        
        return HANDLING_FRAME_TIME_FRONT


    @staticmethod
    def Measure_processing_time_backward() -> float:
        """Measurement of processing time (second half)

        Returns:
            float: Second half point
        """        
        HANDLING_FRAME_TIME_REAR: float = perf_counter()

        return HANDLING_FRAME_TIME_REAR

    
    def Measure_func(self, func) :
        """Used as a decorator to time a function"""
        self.func = func

        @wraps(self.func)
        def wrapper(*args, **kargs) :
            start: float = perf_counter()
            result = func(*args,**kargs)
            elapsed_time: float =  round((perf_counter() - start) * 1000, 2)

            print(f"{func.__name__}.............{elapsed_time}[mSec]")

            return result
        return wrapper


    def cal_specify_date(self, logger) -> None:
        """Run evaluation version only"""
        self.logger = logger
        limit_date = datetime(2022,12, 1, 0, 0, 0)   # 指定日付
        today = datetime.now()

        if today >= limit_date:
            self.logger.warning("試用期限を過ぎました")
            self.logger.warning("引き続きご利用になる場合は下記までご連絡下さい")
            self.logger.warning("東海顔認証 担当: 袈裟丸 y.kesamaru@tokai-kaoninsho.com")
            exit(0)
        elif today < limit_date:
            remaining_days = limit_date - today
            if remaining_days.days < 30:
                self.logger.info("お使いいただける残日数は",  str(remaining_days.days) + "日です")
                self.logger.info("引き続きご利用になる場合は下記までご連絡下さい")
                self.logger.info("東海顔認証 担当: 袈裟丸 y.kesamaru@tokai-kaoninsho.com")


    def cal_resized_telop_image(
            self,
            resized_telop_image: npt.NDArray[np.float64]
        ) -> Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        """Calculate telop image data

        Args:
            resized_telop_image (npt.NDArray[np.float64]): Resized telop image data

        Returns:
            Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]: Tuple
        
        Example:
            >>> cal_resized_telop_nums = \\
                        Cal().cal_resized_telop_image(resized_telop_image)
        """        
        self.resized_telop_image = resized_telop_image

        x1, y1, x2, y2 = 0, 0, resized_telop_image.shape[1], resized_telop_image.shape[0]
        a = (1 - resized_telop_image[:,:,3:] / 255)
        b = resized_telop_image[:,:,:3] * (resized_telop_image[:,:,3:] / 255)
        cal_resized_telop_nums = (x1, y1, x2, y2, a, b)
        
        return cal_resized_telop_nums


    def cal_resized_logo_image(
            self,
            resized_logo_image: npt.NDArray[np.float64],
            set_height: int,
            set_width: int
        ) -> Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        """Calculate logo image data

        Args:
            resized_logo_image (npt.NDArray[np.float64]): Resized logo image data
            set_height (int): Height
            set_width (int): Width

        Returns:
            Tuple[int,int,int,int,npt.NDArray[np.float64],npt.NDArray[np.float64]]: Return tuple

        Example:
            >>> cal_resized_logo_nums = \\
                        Cal().cal_resized_logo_image(
                            resized_logo_image,
                            set_height,
                            set_width
                        )
        """        
        self.resized_logo_image: npt.NDArray[np.float64] = resized_logo_image
        self.set_height: int = set_height
        self.set_width: int = set_width

        x1, y1, x2, y2 = set_width - resized_logo_image.shape[1], set_height - resized_logo_image.shape[0], set_width, set_height
        a = (1 - resized_logo_image[:,:,3:] / 255)
        b = resized_logo_image[:,:,:3] * (resized_logo_image[:,:,3:] / 255)
        
        cal_resized_logo_nums: Tuple[int, int, int, int, npt.NDArray[np.float64], npt.NDArray[np.float64]] = \
            (x1, y1, x2, y2, a, b)
        
        return cal_resized_logo_nums


    def to_tolerance(
            self,
            similar_percentage: float
        ) -> float:
        """Receive similar_percentage and return tolerance

        Args:
            similar_percentage (float): 'Probability of similarity' described in config.ini
        
        Returns:
            float: tolerance

        Example:
            >>> tolerance: float = Cal().to_tolerance(self.CONFIG["similar_percentage"])
        """        
        
        ## 算出式
        ## percentage = -4.76190475*(p*p)+(-0.380952375)*p+100
        ## percentage_example = -4.76190475*(0.45*0.45)+(-0.380952375)*0.45+100
        ## -4.76190475*(p*p)+(-0.380952375)*p+(100-similar_percentage) = 0
        
        self.similar_percentage: float = similar_percentage
        tolerance: float = 0.0

        tolerance_plus: float = (-1*(-0.380952375) + np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-self.similar_percentage))) / (2*(-4.76190475))
        tolerance_minus: float = (-1*(-0.380952375)-np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-self.similar_percentage))) / (2*(-4.76190475))
        
        if 0 < tolerance_plus < 1:
            tolerance= tolerance_plus
        elif 0 < tolerance_minus < 1:
            tolerance= tolerance_minus
        
        return tolerance


    def to_percentage(
            self,
            tolerance: float
        ) -> float:
        """Receive 'tolerance' and return 'percentage'

        Args:
            tolerance (float): tolerance

        Returns:
            float: percentage
        """        
        self.tolerance: float = tolerance

        percentage: float = -4.76190475*(self.tolerance ** 2)+(-0.380952375) * self.tolerance +100
        
        return percentage


    def decide_text_position(
            self,
            error_messg_rectangle_bottom,
            error_messg_rectangle_left,
            error_messg_rectangle_right,
            error_messg_rectangle_fontsize,
            error_messg_rectangle_messg
        ):
        """Not use"""
        self.error_messg_rectangle_bottom = error_messg_rectangle_bottom
        self.error_messg_rectangle_left = error_messg_rectangle_left
        self.error_messg_rectangle_right = error_messg_rectangle_right
        self.error_messg_rectangle_fontsize = error_messg_rectangle_fontsize
        self.error_messg_rectangle_messg = error_messg_rectangle_messg
        error_messg_rectangle_center = int((self.error_messg_rectangle_left + self.error_messg_rectangle_right)/2)
        error_messg_rectangle_chaCenter = int(len(self.error_messg_rectangle_messg)/2)
        error_messg_rectangle_pos = error_messg_rectangle_center - (error_messg_rectangle_chaCenter * self.error_messg_rectangle_fontsize) - int(self.error_messg_rectangle_fontsize / 2)
        error_messg_rectangle_position = (error_messg_rectangle_pos + self.error_messg_rectangle_fontsize, self.error_messg_rectangle_bottom - (self.error_messg_rectangle_fontsize * 2))
        
        return error_messg_rectangle_position


    def make_error_messg_rectangle_font(
            self,
            fontpath: str,
            error_messg_rectangle_fontsize: str,
            encoding = 'utf-8'
        ):
        """Not use"""
        self.fontpath = fontpath
        self.error_messg_rectangle_fontsize = error_messg_rectangle_fontsize
        
        error_messg_rectangle_font = ImageFont.truetype(self.fontpath, self.error_messg_rectangle_fontsize, encoding = 'utf-8')
        
        return error_messg_rectangle_font


    def return_percentage(
            self,
            p: float
        ) -> float:  # python版
        """Receive 'distance' and return percentage

        Args:
            p (float): distance

        Returns:
            float: percentage

        Example:
            >>> percentage = Cal().return_percentage(p)
        """        
        self.p: float = p
        percentage: float = -4.76190475 *(self.p**2)-(0.380952375*self.p)+100
        
        return percentage


    def pil_img_instance(
            self,
            frame: npt.NDArray[np.uint8]
        ) :
        """Generate pil_img object

        Args:
            frame (npt.NDArray[np.uint8]): Image data

        Returns:
            object: PIL object
        """        
        self.frame: npt.NDArray[np.uint8] = frame

        pil_img_obj = Image.fromarray(self.frame)

        return pil_img_obj


    def  make_draw_rgb_object(
            self,
            pil_img_obj_rgb
        ):
        """Generate object

        Args:
            pil_img_obj_rgb (object): object

        Returns:
            object: object
        """        
        self.pil_img_obj_rgb = pil_img_obj_rgb

        draw_rgb = ImageDraw.Draw(self.pil_img_obj_rgb)
        
        return draw_rgb
