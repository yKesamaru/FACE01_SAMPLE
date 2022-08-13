from setuptools import setup
from Cython.Build import cythonize
import glob
from os.path import basename, splitext
from copy import copy


py_file_list = glob.glob('/home/terms/bin/FACE01/face01lib/pyx/*pyx')
for pyfile in py_file_list:
    setup(
        ext_modules = cythonize(
            pyfile,
            # annotate=True  # htmlファイルを出力
        )
    )

"""compile
cd ~/bin/FACE01/face01lib/pyx
python compile.py build_ext --inplace
pactl set-sink-volume @DEFAULT_SINK@ 30%
play -v 0.2 -q /home/terms/done.wav


"""