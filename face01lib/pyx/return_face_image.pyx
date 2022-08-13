# cython: language_level=3
# cython: profile = True
"""
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
import cython
import nptyping
import numpy as np

class Return_face_image():
    @cython.returns(nptyping.NDArray)
    def return_face_image(
        self,
        resized_frame:nptyping.NDArray,
        face_location: tuple
    ):
        """Return face image array which contain ndarray

        Args:
            resized_frame (numpy.ndarray): frame data
            face_location (tuple): face location which ordered top, right, bottom, left

        Returns:
            list: face image of ndarray or empty array
        """        
        self.resized_frame: nptyping.NDArray = resized_frame
        empty_ndarray: nptyping.NDArray = \
            np.empty(shape=(2,2,3), dtype=np.uint8)
        self.face_location: tuple = face_location

        if len(self.face_location) > 0:
            top: cython.int = face_location[0]
            right: cython.int = face_location[1]
            bottom: cython.int = face_location[2]
            left: cython.int = face_location[3]
            face_image: nptyping.NDArray = self.resized_frame[top:bottom, left:right]
            """DEBUG
            from face01lib.video_capture import VidCap
            VidCap().frame_imshow_for_debug(face_image)
            VidCap().frame_imshow_for_debug(self.resized_frame)
            """
            return face_image
        else:
            return empty_ndarray