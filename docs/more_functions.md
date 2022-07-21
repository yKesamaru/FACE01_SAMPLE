FACE01 have many functions inner `face01lib/'.
This section, we will talk about a little bit small functions.

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
        self.similar_percentage = similar_percentage
        tolerance: float = 0.0
        tolerance_plus: float = (-1*(-0.380952375) + np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-self.similar_percentage))) / (2*(-4.76190475))
        tolerance_minus: float = (-1*(-0.380952375)-np.sqrt((-0.380952375)*(-0.380952375)-4*(-4.76190475)*(100-self.similar_percentage))) / (2*(-4.76190475))
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

    def decide_text_position(self, error_messg_rectangle_bottom,error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_fontsize,error_messg_rectangle_messg):
        """未使用"""
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

    def make_error_messg_rectangle_font(self, fontpath, error_messg_rectangle_fontsize, encoding = 'utf-8'):
        self.fontpath = fontpath
        self.error_messg_rectangle_fontsize = error_messg_rectangle_fontsize
        error_messg_rectangle_font = ImageFont.truetype(self.fontpath, self.error_messg_rectangle_fontsize, encoding = 'utf-8')
        return error_messg_rectangle_font

    def return_percentage(self, p):  # python版
        self.p = p
        percentage = -4.76190475 *(self.p**2)-(0.380952375*self.p)+100
        return percentage

    # pil_imgオブジェクトを生成
    def pil_img_instance(self, frame):
        self.frame = frame
        pil_img_obj= Image.fromarray(self.frame)
        return pil_img_obj

    # draw_rgbオブジェクトを生成
    def  make_draw_rgb_object(self, pil_img_obj_rgb):