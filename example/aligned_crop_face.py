"""Example of to detect, rotate and crop face images.

Summary:
    In this example, you can learn how to get aligned face images.
    
Args:
    path (str): Directory path where images containing faces exist
    size (int, optional): Specify the number of px for the extracted face image with an integer. Default is 200.

Usage:
    >>> python3 example/aligned_crop_face.py path size

"""

# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)

from typing import Dict

from face01lib.Initialize import Initialize
from face01lib.utils import Utils


# Initialize
CONFIG: Dict =  Initialize('DEFAULT', 'info').initialize()
utils = Utils(CONFIG['log_level'])


def main(path: str, padding: float = 0.4, size: int = 200) -> None:
    """Simple example.

    This simple example script takes a path which contained png, jpg, jpeg files in the directory, 
    extracts the face, aligns, crops and saves them.
    
    Args:
        path (str): Directory path where images containing faces exist
        padding (float): Padding around the face. Large = 0.8, Medium = 0.4, Small = 0.25. Default = 0.4
        size (int, optional): Specify the number of px for the extracted face image with an integer. Default is 200.
    """
    utils.align_and_resize_maintain_aspect_ratio(path, padding, size)


if __name__ == '__main__':
    args: list = sys.argv
    if 2 == len(args):
        main(args[1])
    elif 3 == len(args):
        main(args[1], float(args[2]))
    elif 4 == len(args):
        main(args[1], float(args[2]), int(args[3]))
    else:
        print("Argument must be PATH")