"""Calc.pyを作成したためこのファイルは廃止予定"""


__doc__ = 'similar_percentageを受取りtoleranceを返す'

from numpy import sqrt
import logging


"""Logging"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')

file_handler = logging.FileHandler('face01.log', mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



# 変数初期化 ---------
similar_percentage:float = 0.0
tolerance: float = 0.0
# --------------------

def to_tolerance(similar_percentage) -> float:

    ## 算出式
    ## percentage = -4.76190475*(p*p)+(-0.380952375)*p+100
    ## percentage_example = -4.76190475*(0.45*0.45)+(-0.380952375)*0.45+100
    ## -4.76190475*(p*p)+(-0.380952375)*p+(100-similar_percentage) = 0

    tolerance_plus: float = (-1*(-0.380952375) + sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
    tolerance_minus: float = (-1*(-0.380952375)-sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
    if 0 < tolerance_plus < 1:
        tolerance=tolerance_plus
    elif 0 < tolerance_minus < 1:
        tolerance=tolerance_minus

    return tolerance

def to_percentage(tolerance):
    # str型で渡されてもいいようにfloatに型変換
    tolerance = float(tolerance)

    percentage = -4.76190475*(tolerance ** 2)+(-0.380952375) * tolerance +100

    return percentage