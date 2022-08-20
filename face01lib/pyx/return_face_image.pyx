# cython: language_level=3
"""
# cython: profile = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False
# cython: cdivision = True
# cython: always_allow_keywords = False
# cython: unraisable_tracebacks = False
# cython: binding = False
"""

"""cythonでは使用不可
from __future__ import annotations
"""
import numpy as np

class Return_face_image():
    def return_face_image(
        self,
        resized_frame,
        face_location: tuple
    ):
        """Return face image array which contain ndarray

        Args:
            resized_frame (numpy.ndarray): frame data
            face_location (tuple): face location which ordered top, right, bottom, left

        Returns:
            list: face image of ndarray or empty array
        """        
        self.resized_frame = resized_frame
        empty_ndarray = \
            np.empty(shape=(2,2,3), dtype=np.uint8)
        self.face_location: tuple = face_location

        if len(self.face_location) > 0:
            top: int = face_location[0]
            right: int = face_location[1]
            bottom: int = face_location[2]
            left: int = face_location[3]
            face_image = self.resized_frame[top:bottom, left:right]
            # face_image = self.resized_frame[:,top:bottom, left:right]
            """DEBUG
            from face01lib.video_capture import VidCap
            VidCap().frame_imshow_for_debug(face_image)
            # print(face_image)
            VidCap().frame_imshow_for_debug(self.resized_frame)
            # print(self.resized_frame)
            """
            return face_image
        else:
            return empty_ndarray