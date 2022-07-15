class Core:
    import mediapipe as mp
    import numpy as np

    def __init__(self) -> None:
        pass

    def mp_face_detection_func(self, resized_frame, model_selection=0, min_detection_confidence=0.4):
        face = self.mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        """refer to
        https://solutions.mediapipe.dev/face_detection#python-solution-api
        """    
        # 推論処理
        inference = face.process(resized_frame)
        """
        Processes an RGB image and returns a list of the detected face location data.
        Args:
            image: An RGB image represented as a numpy ndarray.
        Raises:
            RuntimeError: If the underlying graph throws any error.
        ValueError: 
            If the input image is not three channel RGB.
        Returns:
            A NamedTuple object with a "detections" field that contains a list of the
            detected face location data.'
        """
        return inference

    def return_face_location_list(self, resized_frame, set_width, set_height, model_selection, min_detection_confidence) -> tuple:
        """
        return: face_location_list
        """
        self.resized_frame = resized_frame
        self.set_width = set_width
        self.set_height = set_height
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.resized_frame.flags.writeable = False
        face_location_list: list = []
        person_frame = self.np.empty((2,0), dtype = self.np.float64)
        result = self.mp_face_detection_func(self.resized_frame, self.model_selection, self.min_detection_confidence)
        if not result.detections:
            return face_location_list
        else:
            for detection in result.detections:
                xleft:int = int(detection.location_data.relative_bounding_box.xmin * self.set_width)
                xtop :int= int(detection.location_data.relative_bounding_box.ymin * self.set_height)
                xright:int = int(detection.location_data.relative_bounding_box.width * self.set_width + xleft)
                xbottom:int = int(detection.location_data.relative_bounding_box.height * self.set_height + xtop)
                # see bellow
                # https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python
                
                if xleft <= 0 or xtop <= 0:  # xleft or xtop がマイナスになる場合があるとバグる
                    continue
                face_location_list.append((xtop,xright,xbottom,xleft))  # faceapi order

        self.resized_frame.flags.writeable = True
        return face_location_list