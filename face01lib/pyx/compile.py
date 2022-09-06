# from setuptools import setup
# from Cython.Build import cythonize
# import glob


# py_file_list = glob.glob('/home/terms/bin/FACE01/face01lib/pyx/*pyx')
# for pyfile in py_file_list:
#     setup(
#         ext_modules = cythonize(
#             pyfile,
#         )
#     )

# setup(
#     ext_modules = cythonize(
#         "Core.pyx",
#     )
# )

# setup(
#     ext_modules = cythonize(
#         "return_face_image.py",
#     )
# )


# compile.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

# [Compiler options](https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#compiler-options)
Options.docstrings = False
Options.annotate = True
Options.gcc_branch_hints = True
Options.buffer_max_dims = 4

# [Specify C++ language in setup.py](https://cython.readthedocs.io/en/stable/src/userguide/wrapping_CPlusPlus.html?highlight=language#specify-c-language-in-setup-py)
extensions = [
    # Extension("*", ["*.py"],  # If use glob.
    Extension(
        "return_face_image",
        ["return_face_image.py"],
        extra_compile_args=['-O3'],
        # language="c++"  # If you want.
    )
]
setup(
    ext_modules = cythonize(
        # [Cythonize arguments](https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#cythonize-arguments)
        extensions,
        nthreads=4,
        compiler_directives= \
        {
            # [Compiler directives](https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#compiler-directives)
            'binding': False,  # Default is False.
            'boundscheck': False,  # Default is True.
            'wraparound': False,  # Default is True.
            'overflowcheck': True,  # Default is False.
            'cdivision': True,  # Default is False.
            'profile': True,  # Default is False.
            'infer_types': True,  # Default is None.
            'language_level': 3,  # Default is not defined. 2/3/3str
            'unraisable_tracebacks': False,  # Default is not defined. True/False
        }
    ),
)

"""compile
cd ~/bin/FACE01/face01lib/pyx
python compile.py build_ext --inplace
pactl set-sink-volume @DEFAULT_SINK@ 30%
play -v 0.2 -q /home/terms/done.wav


"""