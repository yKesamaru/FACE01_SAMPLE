#cython: language_level=3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    # see bellow
    # [C コンパイラのオプションの設定](https://blog.ymyzk.com/2014/11/setuptools-cython/)
    Extension(
        "load_priset_image",  
        sources=["load_priset_image.pyx"],
        # extra_compile_args=['--option-a', '-O3'],
        include_dirs=[numpy.get_include()]
    ),
]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
"""compile
cd ~/bin/FACE01/face01lib/pyx
python compile.py build_ext --inplace

"""
"""メモ
[ライブラリヘッダファイルの追加](https://qiita.com/gwappa/items/db1f6f27218da0c5a932)
> Cythonからnumpyを使ったりする場合、numpyライブラリのCヘッダファイル群が必要になる。gccやらを用いて直にコンパイルする場合はわかりやすいが、setup.py経由でコンパイルする場合は-Iオプションが使えない。
> この場合、distutilsのExtensionオブジェクトを利用する
"""