"""Example of to get face encoded data.

Summary:
    In this example, you can learn how to get face encoded datas.

Usage:
    >>> python3 example/get_encoded_data.py

Note:
   Make sure this example must be applied for 'one person'.
   Not more two persons in same input.
"""

# Operate directory: Common to all examples
import os.path
import sys
dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)


from typing import Dict

from face01lib.Core import Core
from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.video_capture import VidCap


def main(exec_times: int = 50) -> None:
    """Simple example.

    This simple example script prints out results of face encoded datas.

    Args:
        exec_times (int, optional): Number of frames for process. Defaults to 50.
    """    
    # Initialize
    CONFIG: Dict =  Initialize('DEFAULT', 'info').initialize()

    # Make generator
    frame_generator_obj = VidCap().frame_generator(CONFIG)

    # Make logger
    log = Logger().logger(__file__, dir)

    # Make generator
    core = Core()

    # Repeat 'exec_times' times
    for i in range(0, exec_times):

        # Call __next__() from the generator object
        resized_frame = frame_generator_obj.__next__()

        # VidCap().frame_imshow_for_debug(resized_frame)

        frame_datas_array = core.frame_pre_processing(log, CONFIG, resized_frame)
        face_encodings, frame_datas_array = \
            core.face_encoding_process(log, CONFIG, frame_datas_array)

        for encoded_data in face_encodings:
            print(f"face encoded data: {encoded_data}\n")


if __name__ == '__main__':
    # Call main function.
    main(exec_times = 10)