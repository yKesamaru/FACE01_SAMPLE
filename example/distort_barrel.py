"""Example of to distort images.

Summary:
    In this example, you can learn how to get distorted images.
    
Args:
    path (str): Directory path where images containing faces exist
    size (int, optional): Specify the number of px for the extracted face image with an integer. Default is 200.

Usage:
    .. code-block:: bash
    
        python3 example/distort_barrel.py path size
"""

# Operate directory: Common to all examples
import os.path
import sys

dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)

from typing import Dict

from face01lib.Initialize import Initialize
from face01lib.logger import Logger
from face01lib.utils import Utils


# Initialize
CONFIG: Dict =  Initialize('DEFAULT', 'info').initialize()
# Set up logger
logger = Logger(CONFIG['log_level']).logger(__file__, CONFIG['RootDir'])
"""Initialize and Setup logger.
When coding a program that uses FACE01, code `initialize` and `logger` first.
This will read the configuration file `config.ini` and log errors etc.
"""

utils = Utils(CONFIG['log_level'])


def main(path: str, size: int = 200) -> None:
    """Simple example.

    This simple example script takes a path which contained png, jpg, jpeg files in the directory, 
    distort barrel and saves them.
    Tokai-kaoninsho:レンズの歪曲収差と対応方法(6)_
    
    See _Tokai-kaoninsho:レンズの歪曲収差と対応方法(6) https://tokai-kaoninsho.com/%e3%82%b3%e3%83%a9%e3%83%a0/%e3%83%ac%e3%83%b3%e3%82%ba%e3%81%ae%e6%ad%aa%e6%9b%b2%e5%8f%8e%e5%b7%ae%e3%81%a8%e5%af%be%e5%bf%9c%e6%96%b9%e6%b3%956/
    
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
        ImageMagick_
        
        - See _ImageMagick https://imagemagick.org/script/download.php
    
    Result:
        .. image:: ../docs/img/distort_barrel.png
            :scale: 100%
            :alt: distort_barrel

    Image:
        Pakutaso_
        
        _Pakutaso https://www.pakutaso.com/20220158028post-38602.html
    """
    utils.distort_barrel(path, size)


if __name__ == '__main__':
    args: list = sys.argv
    main(args[1], size=200)