import cv2
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageFont
from platform import system

ImageFile.LOAD_TRUNCATED_IMAGES = True
from os.path import exists


class LibDraw:
    def __init__(self) -> None:
        pass

    def draw_pink_rectangle(self, resized_frame, top,bottom,left,right) -> np.ndarray:
        self.resized_frame = resized_frame
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        cv2.rectangle(self.resized_frame, (self.left, self.top), (self.right, self.bottom), (255, 87, 243), 2) # pink
        return self.resized_frame
        
    def draw_white_rectangle(self, rectangle, resized_frame, top, left, right, bottom) -> np.ndarray:
        self.rectangle = rectangle
        self.resized_frame = resized_frame
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
        cv2.rectangle(self.resized_frame, (self.left-18, self.top-18), (self.right+18, self.bottom+18), (175, 175, 175), 2) # 灰色内枠
        cv2.rectangle(self.resized_frame, (self.left-20, self.top-20), (self.right+20, self.bottom+20), (255,255,255), 2) # 白色外枠
        return self.resized_frame

    # パーセンテージ表示
    def display_percentage(self, percentage_and_symbol,resized_frame, p, left, right, bottom, tolerance) -> np.ndarray:
        self.percentage_and_symbol = percentage_and_symbol
        self.resized_frame = resized_frame
        self.p = p
        self.left = left
        self.right = right
        self.bottom = bottom
        self.tolerance = tolerance
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # パーセンテージ表示用の灰色に塗りつぶされた四角形の描画
        cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 75), (self.right+25, self.bottom+50), (30,30,30), cv2.FILLED) # 灰色
        # テキスト表示位置
        fontsize = 14
        putText_center = int((self.left-25 + self.right+25)/2)
        putText_chaCenter = int(5/2)
        putText_pos = putText_center - (putText_chaCenter*fontsize) - int(fontsize/2)
        putText_position = (putText_pos, self.bottom + 75 - int(fontsize / 2))
        # toleranceの値によってフォント色を変える
        if self.p < self.tolerance:
            # パーセンテージを白文字表示
            self.resized_frame = cv2.putText(self.resized_frame, self.percentage_and_symbol, putText_position, font, 1, (255,255,255), 1, cv2.LINE_AA)
        else:
            # パーセンテージをピンク表示
            self.resized_frame = cv2.putText(self.resized_frame, self.percentage_and_symbol, putText_position, font, 1, (255, 87, 243), 1, cv2.LINE_AA)
        return self.resized_frame

    # デフォルト顔画像の描画処理
    def draw_default_face_image(self, logger, resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width):
        self.resized_frame = resized_frame
        self.default_face_small_image = default_face_small_image
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.number_of_people = number_of_people
        self.face_image_width = face_image_width
        try:
            self.x1 = self.x1 + (self.number_of_people * self.face_image_width)
            self.x2 = self.x2 + (self.number_of_people * self.face_image_width)
            self.resized_frame[self.y1:self.y2, self.x1:self.x2] = self.resized_frame[self.y1:self.y2, self.x1:self.x2] * (1 - self.default_face_small_image[:,:,3:] / 255) + self.default_face_small_image[:,:,:3] * (default_face_small_image[:,:,3:] / 255)
            # resized_frame[y1:y2, x1:x2] = resized_frame[y1:y2, x1:x2] * a + b  # ValueError: assignment destination is read-only
            """DEBUG"""
            # frame_imshow_for_debug(resized_frame)
        except:
            logger.info('デフォルト顔画像の描画が出来ません')
            logger.info('描画面積が足りないか他に問題があります')
        return self.resized_frame

    # デフォルト顔画像の表示面積調整
    def adjust_display_area(self, args_dict, default_face_image) -> tuple:
        self.args_dict = args_dict
        self.default_face_image = default_face_image
        """TODO
        繰り返し計算させないようリファクタリング"""
        face_image_width = int(self.args_dict["set_width"] / 15)
        default_face_small_image = cv2.resize(self.default_face_image, dsize=(face_image_width, face_image_width))  # 幅・高さともに同じとする
        # 高さ = face_image_width
        x1, y1, x2, y2 = 0, self.args_dict["set_height"] - face_image_width - 10, face_image_width, self.args_dict["set_height"] - 10
        return x1, y1, x2, y2, default_face_small_image, face_image_width

    def draw_default_face(self, logger, args_dict, name, resized_frame, number_of_people):
        self.logger = logger
        self.args_dict = args_dict
        self.name = name
        self.resized_frame = resized_frame
        self.number_of_people = number_of_people
        default_face_image_dict = self.args_dict["default_face_image_dict"]

        default_name_png = self.name + '_default.png'
        default_face_image_name_png = './priset_face_images/' + default_name_png
        if not self.name in default_face_image_dict:  # default_face_image_dictにnameが存在しなかった場合
            # 各人物のデフォルト顔画像ファイルの読み込み
            if exists(default_face_image_name_png):
                # WINDOWSのopencv-python4.2.0.32ではcv2.imread()でpng画像を読み込めないバグが
                # 存在する可能性があると思う。そこでPNG画像の読み込みにはpillowを用いることにする
                default_face_image = np.array(Image.open(default_face_image_name_png))
                """DEBUG
                frame_imshow_for_debug(default_face_image)
                """
                # BGAをRGBへ変換
                default_face_image = cv2.cvtColor(default_face_image, cv2.COLOR_BGR2RGBA)
                """DEBUG
                frame_imshow_for_debug(default_face_image)
                """
                # if default_face_image.ndim == 3:  # RGBならアルファチャンネル追加 resized_frameがアルファチャンネルを持っているから。
                # default_face_imageをメモリに保持
                default_face_image_dict[self.name] = default_face_image  # キーnameと値default_face_imageの組み合わせを挿入する
            else:
                self.logger.info(f'{self.name}さんのデフォルト顔画像ファイルがpriset_face_imagesフォルダに存在しません')
                self.logger.info(f'{self.name}さんのデフォルト顔画像ファイルをpriset_face_imagesフォルダに用意してください')
        else:  # default_face_image_dictにnameが存在した場合
            default_face_image = default_face_image_dict[self.name]  # キーnameに対応する値をdefault_face_imageへ格納する
            """DEBUG
            frame_imshow_for_debug(default_face_image)  # OK
            """
            x1, y1, x2, y2 , default_face_small_image, face_image_width = LibDraw().adjust_display_area(args_dict, default_face_image)
            resized_frame = LibDraw().draw_default_face_image(logger, resized_frame, default_face_small_image, x1, y1, x2, y2, number_of_people, face_image_width)
        return resized_frame

    def draw_rectangle_for_name(self, name,resized_frame, left, right,bottom):
        self.name = name
        self.resized_frame = resized_frame
        self.left = left
        self.right = right
        self.bottom = bottom
        if self.name == 'Unknown':   # nameがUnknownだった場合
            self.resized_frame = cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 25), (self.right+25, self.bottom+50), (255, 87, 243), cv2.FILLED) # pink
        else:                   # nameが既知だった場合
            # cv2.rectangle(resized_frame, (left-25, bottom + 25), (right+25, bottom+50), (211, 173, 54), thickness = 1) # 濃い水色の線
            self.resized_frame = cv2.rectangle(self.resized_frame, (self.left-25, self.bottom + 25), (self.right+25, self.bottom+50), (211, 173, 54), cv2.FILLED) # 濃い水色
        return self.resized_frame

    # 帯状四角形（ピンク）の描画
    def draw_error_messg_rectangle(self, resized_frame, set_height, set_width):
        """廃止予定
        """        
        self.resized_frame = resized_frame
        self.set_height = set_height
        self.set_width = set_width
        error_messg_rectangle_top: int  = int((self.set_height + 20) / 2)
        error_messg_rectangle_bottom : int = int((self.set_height + 120) / 2)
        error_messg_rectangle_left: int  = 0
        error_messg_rectangle_right : int = self.set_width
        cv2.rectangle(self.resized_frame, (error_messg_rectangle_left, error_messg_rectangle_top), (error_messg_rectangle_right, error_messg_rectangle_bottom), (255, 87, 243), cv2.FILLED)  # pink
        return error_messg_rectangle_left, error_messg_rectangle_right, error_messg_rectangle_bottom

    # drawオブジェクトを生成
    def  make_draw_object(self, frame):
        self.frame = frame
        draw = ImageDraw.Draw(Image.fromarray(self.frame))
        return draw

    def draw_error_messg_rectangle_messg(self, draw, error_messg_rectangle_position, error_messg_rectangle_messg, error_messg_rectangle_font):
        """廃止予定
        """
        self.draw = draw
        self.error_messg_rectangle_position = error_messg_rectangle_position
        self.error_messg_rectangle_messg = error_messg_rectangle_messg
        self.error_messg_rectangle_font = error_messg_rectangle_font
        draw.text(self.error_messg_rectangle_position, self.error_messg_rectangle_messg, fill=(255, 255, 255, 255), font = self.error_messg_rectangle_font)

    def return_fontpath(self, logger):
        # フォントの設定(フォントファイルのパスと文字の大きさ)
        operating_system: str  = system()
        fontpath: str = ''
        if (operating_system == 'Linux'):
            fontpath = "/usr/share/fonts/truetype/mplus/mplus-1mn-bold.ttf"
        elif (operating_system == 'Windows'):
                        # fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICR.TTC"
            fontpath = "C:/WINDOWS/FONTS/BIZ-UDGOTHICB.TTC"  ## bold体
        else:
            logger.info('オペレーティングシステムの確認が出来ません。システム管理者にご連絡ください')
        return fontpath

    def calculate_text_position(self, left,right,name,fontsize,bottom):
        self.left = left
        self.right = right
        self.name = name
        self.fontsize = fontsize
        self.bottom = bottom
        center = int((self.left + self.right)/2)
        chaCenter = int(len(self.name)/2)
        pos = center - (chaCenter* self.fontsize) - int(self.fontsize/2)
        position = (pos, self.bottom + (self.fontsize * 2))
        Unknown_position = (pos + self.fontsize, self.bottom + (self.fontsize * 2))
        return position, Unknown_position

    def draw_name(self, name,pil_img_obj, Unknown_position, font, p, tolerance, position):
        self.name = name
        self.pil_img_obj = pil_img_obj
        self.Unknown_position = Unknown_position
        self.font = font
        self.p = p
        self.tolerance = tolerance
        self.position = position
        local_draw_obj = ImageDraw.Draw(self.pil_img_obj)
        if self.name == 'Unknown':  ## nameがUnknownだった場合
            # draw.text(Unknown_position, '照合不一致', fill=(255, 255, 255, 255), font = font)
            local_draw_obj.text(self.Unknown_position, '　未登録', fill=(255, 255, 255, 255), font = self.font)
        else:  ## nameが既知の場合
            # if percentage > 99.0:
            if self.p < self.tolerance:
                # nameの描画
                local_draw_obj.text(self.position, self.name, fill=(255, 255, 255, 255), font = self.font)
            else:
                local_draw_obj.text(self.position, self.name, fill=(255, 87, 243, 255), font = self.font)
        return self.pil_img_obj

    # pil_img_objをnumpy配列に変換
    def convert_pil_img_to_ndarray(self, pil_img_obj):
        self.pil_img_obj = pil_img_obj
        frame = np.array(pil_img_obj)
        return frame

    def draw_text_for_name(self, logger, left,right,bottom,name, p,tolerance,pil_img_obj):
        self.logger = logger
        self.left = left
        self.right = right
        self.bottom = bottom
        self.name = name
        self.p = p
        self.tolerance = tolerance
        self.pil_img_obj = pil_img_obj
        fontpath = LibDraw().return_fontpath(logger)
        """TODO FONTSIZEハードコーティング訂正"""
        fontsize = 14
        font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
        # テキスト表示位置決定
        position, Unknown_position = LibDraw().calculate_text_position(self.left,self.right,self.name,fontsize,self.bottom)
        # nameの描画
        self.pil_img_obj = LibDraw().draw_name(self.name,self.pil_img_obj, Unknown_position, font, self.p, self.tolerance, position)
        # pil_img_objをnumpy配列に変換
        resized_frame = LibDraw().convert_pil_img_to_ndarray(self.pil_img_obj)
        return resized_frame

    # target_rectangleの描画
    def draw_target_rectangle(self, anti_spoof, rect01_png, rect01_NG_png, resized_frame,top,bottom,left,right,name):
        self.anti_spoof = anti_spoof,
        self.rect01_png = rect01_png
        self.rect01_NG_png = rect01_NG_png
        self.resized_frame = resized_frame
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        face_location_list = (self.top, self.left, self.bottom, self.right) 
        self.name = name
        width_ratio: float
        height_ratio: float
        face_width: int
        face_height: int
        if not self.name == 'Unknown':  ## self.nameが既知の場合
            face_width: int = self.right - self.left
            face_height: int = self.bottom - self.top
            orgHeight, orgWidth = self.rect01_png.shape[:2]
            width_ratio = 1.0 * (face_width / orgWidth)
            height_ratio = 1.0 * (face_height / orgHeight)
            self.rect01_png = cv2.resize(self.rect01_png, None, fx = width_ratio, fy = height_ratio)
            x1, y1, x2, y2 = self.left, self.top, self.left + self.rect01_png.shape[1], self.top + self.rect01_png.shape[0]
            try:
                self.resized_frame[y1:y2, x1:x2] = self.resized_frame[y1:y2, x1:x2] * (1 - self.rect01_png[:,:,3:] / 255) + \
                            self.rect01_png[:,:,:3] * (self.rect01_png[:,:,3:] / 255)
            except:
                pass
        else:  ## self.nameがUnknownだった場合
            fx: float = 0.0
            face_width = self.right - self.left
            face_height = self.bottom - self.top
            # rect01_NG_png←ピンクのtarget_rectangle
            # rect01_NG_png: cv2.Mat = cv2.imread("images/rect01_NG.png", cv2.IMREAD_UNCHANGED)
            orgHeight, orgWidth = self.rect01_NG_png.shape[:2]
            width_ratio = float(1.0 * (face_width / orgWidth))
            height_ratio = 1.0 * (face_height / orgHeight)
            self.rect01_NG_png = cv2.resize(self.rect01_NG_png, None, fx = width_ratio, fy = height_ratio)
            x1, y1, x2, y2 = self.left, self.top, self.left + self.rect01_NG_png.shape[1], self.top + self.rect01_NG_png.shape[0]
            try:
                self.resized_frame[y1:y2, x1:x2] = self.resized_frame[y1:y2, x1:x2] * (1 - self.rect01_NG_png[:,:,3:] / 255) + \
                            self.rect01_NG_png[:,:,:3] * (self.rect01_NG_png[:,:,3:] / int(255))
            except:
                pass
        return self.resized_frame