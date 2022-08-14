from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = \
    [Pybind11Extension(
        "return_face_image",
        # sorted(glob("src/*.cpp")),
        ["./return_face_image.cpp"]
    )]

setup(
    # ..., 
    cmdclass={"build_ext": build_ext},
    ext_modules = ext_modules
)

"""compile
cd ~/bin/FACE01/face01lib/cpp
python compile.py build_ext --inplace
pactl set-sink-volume @DEFAULT_SINK@ 30% && play -v 0.2 -q /home/terms/done.wav


"""