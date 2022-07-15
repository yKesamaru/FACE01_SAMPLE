#cython: language_level=3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    # soファイル名
    Extension("FACE01",  ["FACE01.pyx"])
]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)