class LoadImage:
    import cv2
    import numpy as np
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    from face01lib.Calc import Cal
    
    def __init__(self, headless: bool, conf_dict) -> None:
        self.headless = headless
        self.conf_dict = conf_dict
        if self.headless == False:
            # それぞれの画像が1度だけしか読み込まれない仕組み
            self.load_telop_image: bool = False
            self.load_logo_image: bool = False
            self.load_unregistered_face_image: bool = False
        else:
            self.load_telop_image: bool = True
            self.load_logo_image: bool = True
            self.load_unregistered_face_image: bool = True

    def LI(self, set_height, set_width):
        rect01_png: self.cv2.Mat = self.cv2.imread("images/rect01.png", self.cv2.IMREAD_UNCHANGED)

        # Load Telop image
        telop_image: self.cv2.Mat
        if not self.load_telop_image:
            telop_image = self.cv2.imread("images/telop.png", self.cv2.IMREAD_UNCHANGED)
            load_telop_image = True
            _, orgWidth = telop_image.shape[:2]
            ratio: float = self.conf_dict["set_width"] / orgWidth / 3  ## テロップ幅は横幅を分母として設定
            resized_telop_image = self.cv2.resize(telop_image, None, fx = ratio, fy = ratio)
            cal_resized_telop_nums = self.Cal().cal_resized_telop_image(resized_telop_image)
        else:
            resized_telop_image = ''
            cal_resized_telop_nums = ''
            telop_image = ''

        # Load Logo image
        logo_image: self.cv2.Mat
        if not self.load_logo_image:
            logo_image: self.cv2.Mat = self.cv2.imread("images/Logo.png", self.cv2.IMREAD_UNCHANGED)
            load_logo_image = True
            _, logoWidth = logo_image.shape[:2]
            logoRatio = self.conf_dict["set_width"] / logoWidth / 15
            resized_logo_image = self.cv2.resize(logo_image, None, fx = logoRatio, fy = logoRatio)
            cal_resized_logo_nums = self.Cal().cal_resized_logo_image(resized_logo_image,  set_height,set_width)
        else:
            resized_logo_image = ''
            cal_resized_logo_nums = ''
            logo_image = ''

        # Load unregistered_face_image
        unregistered_face_image: self.cv2.Mat
        if not self.load_unregistered_face_image:
            unregistered_face_image = self.np.array(self.Image.open('./images/顔画像未登録.png'))
            unregistered_face_image = self.cv2.cvtColor(unregistered_face_image, self.cv2.COLOR_BGR2RGBA)
            load_unregistered_face_image = True
        else:
            load_unregistered_face_image = False
            unregistered_face_image = ''

        return rect01_png, resized_telop_image, cal_resized_telop_nums, resized_logo_image, \
            cal_resized_logo_nums, load_unregistered_face_image, telop_image, logo_image, unregistered_face_image
