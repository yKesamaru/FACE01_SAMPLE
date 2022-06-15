# ===========================================
# similar_percentageを受取りtoleranceを返す =
# ===========================================

import numpy as np
import logging
logger = logging.getLogger('face01lib/similar_percentage_to_tolerance')

# 変数初期化 ---------
similar_percentage:float
tolerance: float
# --------------------

def to_tolerance(similar_percentage) -> float:

    ## 算出式
    ## percentage = -4.76190475*(p*p)+(-0.380952375)*p+100
    ## percentage_example = -4.76190475*(0.45*0.45)+(-0.380952375)*0.45+100
    ## -4.76190475*(p*p)+(-0.380952375)*p+(100-similar_percentage) = 0

    tolerance_plus: float = (-1*(-0.380952375)+np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
    tolerance_minus: float = (-1*(-0.380952375)-np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
    if 0 < tolerance_plus < 1:
        tolerance=tolerance_plus
    elif 0 < tolerance_minus < 1:
        tolerance=tolerance_minus

    return tolerance

def to_percentage(tolerance):
    # str型で渡されてもいいようにfloatに型変換
    tolerance = float(tolerance)

    percentage = -4.76190475*(tolerance * tolerance)+(-0.380952375) * tolerance +100

    return percentage