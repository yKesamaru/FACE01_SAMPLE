class Initialize:
    from face01lib.Calc import Cal
    from datetime import datetime
    from face01lib.load_priset_image import load_priset_image
    import face01lib.video_capture as video_capture  # py
    from face01lib.LoadImage import LoadImage
    def __init__(self) -> None:
        pass
    # Initialize variables, load images
    def initialize(self, conf_dict):
        self.conf_dict = conf_dict
        self.Cal().cal_specify_date
        headless = self.conf_dict["headless"]
        # kaoninshoDir: str = 
        known_face_encodings, known_face_names = self.load_priset_image(self.conf_dict["kaoninshoDir"],self.conf_dict["priset_face_imagesDir"])

        # set_width,fps,height,width,set_height
        set_width,fps,height,width,set_height = \
            self.video_capture.return_movie_property(self.conf_dict["set_width"], self.video_capture.return_vcap(self.conf_dict["movie"]))
        
        # toleranceの算出
        tolerance = self.Cal().to_tolerance(self.conf_dict["similar_percentage"])

        LoadImage_obj = self.LoadImage(headless, self.conf_dict)
        rect01_png, resized_telop_image, cal_resized_telop_nums, resized_logo_image, \
            cal_resized_logo_nums, load_unregistered_face_image, telop_image, logo_image, unregistered_face_image = \
            LoadImage_obj.LI(set_height, set_width)

        # 日付時刻算出
        date = self.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒

        # 辞書作成
        if headless == False:
            init_dict = {
                'known_face_encodings': known_face_encodings,
                'known_face_names': known_face_names,
                'date': date,
                'rect01_png': rect01_png,
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
        args_dict = {**init_dict, **self.conf_dict}

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