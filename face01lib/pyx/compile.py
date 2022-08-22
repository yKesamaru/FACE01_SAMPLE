from setuptools import setup
from Cython.Build import cythonize
import glob


# py_file_list = glob.glob('/home/terms/bin/FACE01/face01lib/pyx/*pyx')
# for pyfile in py_file_list:
#     setup(
#         ext_modules = cythonize(
#             pyfile,
#         )
#     )

setup(
    ext_modules = cythonize(
        "Core.pyx",
    )
)

# setup(
#     ext_modules = cythonize(
#         "return_face_image.pyx",
#     )
# )

"""compile
cd ~/bin/FACE01/face01lib/pyx
python compile.py build_ext --inplace
pactl set-sink-volume @DEFAULT_SINK@ 30%
play -v 0.2 -q /home/terms/done.wav


"""