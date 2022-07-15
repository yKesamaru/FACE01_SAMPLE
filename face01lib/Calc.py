class Cal:
    import logging
    from numpy import sqrt

    def __init__(self) -> None:
        self.similar_percentage:float = 0.0
        self.tolerance: float = 0.0
        """Logging"""
        self.logger = self.logging.getLogger(__name__)
        self.logger.setLevel(self.logging.INFO)
        formatter = self.logging.Formatter('[%(asctime)s] [%(name)s] [%(filename)s] [%(levelname)s] %(message)s')
        file_handler = self.logging.FileHandler('face01.log', mode='a')
        file_handler.setLevel(self.logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = self.logging.StreamHandler()
        stream_handler.setLevel(self.logging.INFO)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def cal_specify_date(self) -> None:
        __doc__ = "指定日付計算: 評価版のみ実行"
        from datetime import datetime
        limit_date = datetime(2022, 12, 1, 0, 0, 0)   # 指定日付
        self.logger.info("試用期限:", limit_date)
        today = datetime.now()

        def limit_date_alart() -> None:
            if today >= limit_date:
                self.logger.warning("試用期限を過ぎました")
                self.logger.warning("引き続きご利用になる場合は下記までご連絡下さい")
                self.logger.warning("東海顔認証　担当：袈裟丸", "y.kesamaru@tokai-kaoninsho.com")
                exit(0)
            elif today < limit_date:
                remaining_days = limit_date - today
                if remaining_days.days < 30:
                    self.logger.info("お使いいただける残日数は",  str(remaining_days.days) + "日です")
                self.logger.info("引き続きご利用になる場合は下記までご連絡下さい")
                self.logger.info("東海顔認証　担当：袈裟丸", "y.kesamaru@tokai-kaoninsho.com")
        limit_date_alart()

    def cal_resized_telop_image(self, resized_telop_image):
        self.resized_telop_image = resized_telop_image
        x1, y1, x2, y2 = 0, 0, resized_telop_image.shape[1], resized_telop_image.shape[0]
        a = (1 - resized_telop_image[:,:,3:] / 255)
        b = resized_telop_image[:,:,:3] * (resized_telop_image[:,:,3:] / 255)
        cal_resized_telop_nums = (x1, y1, x2, y2, a, b)
        return cal_resized_telop_nums

    def cal_resized_logo_image(self, resized_logo_image,  set_height,set_width):
        self.resized_logo_image = resized_logo_image
        self.set_height = set_height
        self.set_width = set_width
        x1, y1, x2, y2 = set_width - resized_logo_image.shape[1], set_height - resized_logo_image.shape[0], set_width, set_height
        a = (1 - resized_logo_image[:,:,3:] / 255)
        b = resized_logo_image[:,:,:3] * (resized_logo_image[:,:,3:] / 255)
        cal_resized_logo_nums = (x1, y1, x2, y2, a, b)
        return cal_resized_logo_nums

    def to_tolerance(self, similar_percentage) -> float:
        __doc__ = 'similar_percentageを受取りtoleranceを返す'

        ## 算出式
        ## percentage = -4.76190475*(p*p)+(-0.380952375)*p+100
        ## percentage_example = -4.76190475*(0.45*0.45)+(-0.380952375)*0.45+100
        ## -4.76190475*(p*p)+(-0.380952375)*p+(100-similar_percentage) = 0
        tolerance = self.tolerance
        tolerance_plus: float = (-1*(-0.380952375) + self.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
        tolerance_minus: float = (-1*(-0.380952375)-self.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-similar_percentage))) / (2*(-4.76190475))
        if 0 < tolerance_plus < 1:
            tolerance= tolerance_plus
        elif 0 < tolerance_minus < 1:
            tolerance= tolerance_minus
        return tolerance

    def to_percentage(self, tolerance):
        self.tolerance = tolerance
        tolerance = float(tolerance)  # str型で渡されてもいいようにfloatに型変換
        percentage = -4.76190475*(tolerance ** 2)+(-0.380952375) * tolerance +100
        return percentage