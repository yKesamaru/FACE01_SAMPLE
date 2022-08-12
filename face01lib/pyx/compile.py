from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "Core.pyx",
        "return_face_image.pyx",
        annotate=True  # htmlファイルを出力
    )
)

"""compile
cd ~/bin/FACE01/face01lib/pyx
python compile.py build_ext --inplace

"""