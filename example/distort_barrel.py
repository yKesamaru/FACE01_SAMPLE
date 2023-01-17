"""Example of to distort images.

Summary:
    In this example, you can learn how to get distorted images.
    
Args:
    path (str): Directory path where images containing faces exist
    size (int, optional): Specify the number of px for the extracted face image with an integer. Default is 200.

Usage:
    >>> python3 example/distort_barrel.py path size

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


def main(path: str, size: int = 200) -> None:
    """Simple example.

    This simple example script takes a path which contained png, jpg, jpeg files in the directory, 
    distort barrel and saves them.
    
    Args:
        path (str): absolute path
        size (int, optional): Width and height. Defaults to 200.
        initial_value (float): Initial value. Default is -0.2.
        closing_value (float): Closing value. Default is 0.2.
        step_value (float): Step value. Default is 0.01.

    Return:
        None

    Note:
        ImageMagick must be installed on your system.
        - See[ImageMagick](https://imagemagick.org/script/download.php)
    """
    utils.distort_barrel(path, size)


if __name__ == '__main__':
    args: list = sys.argv
    main(args[1], size=200)