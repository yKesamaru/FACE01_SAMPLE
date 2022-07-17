#cython: language_level=3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("Core",  ["Core.pyx"])  # soファイル名
]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
"""compile
python compile.py build_ext --inplace
"""