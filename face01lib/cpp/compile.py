# from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# ext_modules = \
#     [Pybind11Extension(
#         "test_numpy",
#         ["./bk_test_numpy.cpp"]
#     )]

# ext_modules = \
#     [Pybind11Extension(
#         "test_numpy",
#         ["./test_numpy.cpp"]
#     )]

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
mv return_face_image.cpython-38-x86_64-linux-gnu.so ../


mv test_numpy.cpython-38-x86_64-linux-gnu.so ../

pactl set-sink-volume @DEFAULT_SINK@ 30% && play -v 0.2 -q /home/terms/done.wav
"""
"""
 terms  terms-Desks  ../bin/FACE01/face01lib/cpp  python compile.py build_ext --inplace
x86_64-linux-gnu.so ../
running build_ext

x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/home/terms/bin/FACE01/include -I/usr/include/python3.8 -c flagcheck.cpp -o flagcheck.o -std=c++17

building 'return_face_image' extension

x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/home/terms/bin/FACE01/lib/python3.8/site-packages/pybind11/include -I/home/terms/bin/FACE01/include -I/usr/include/python3.8 -c ./return_face_image.cpp -o build/temp.linux-x86_64-cpython-38/./return_face_image.o -std=c++17 -fvisibility=hidden -g0

x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 build/temp.linux-x86_64-cpython-38/./return_face_image.o -L/usr/lib -o build/lib.linux-x86_64-cpython-38/return_face_image.cpython-38-x86_64-linux-gnu.so

copying build/lib.linux-x86_64-cpython-38/return_face_image.cpython-38-x86_64-linux-gnu.so -> 
"""
"""
(FACE01) 
 terms  terms-Desks  ../bin/FACE01/face01lib/cpp  python compile.py --help build_ext
Common commands: (see '--help-commands' for more)

  setup.py build      will build the package underneath 'build/'
  setup.py install    will install the package

Global options:
  --verbose (-v)  run verbosely (default)
  --quiet (-q)    run quietly (turns verbosity off)
  --dry-run (-n)  don't actually do anything
  --help (-h)     show detailed help message
  --no-user-cfg   ignore pydistutils.cfg in your home directory

Options for 'build_ext' command:
  --build-lib (-b)           directory for compiled extension modules
  --build-temp (-t)          directory for temporary files (build by-products)
  --plat-name (-p)           platform name to cross-compile for, if supported
                             (default: linux-x86_64)
  --inplace (-i)             ignore build-lib and put compiled extensions into
                             the source directory alongside your pure Python
                             modules
  --include-dirs (-I)        list of directories to search for header files
                             (separated by ':')
  --define (-D)              C preprocessor macros to define
  --undef (-U)               C preprocessor macros to undefine
  --libraries (-l)           external C libraries to link with
  --library-dirs (-L)        directories to search for external C libraries
                             (separated by ':')
  --rpath (-R)               directories to search for shared C libraries at
                             runtime
  --link-objects (-O)        extra explicit link objects to include in the
                             link
  --debug (-g)               compile/link with debugging information
  --force (-f)               forcibly build everything (ignore file
                             timestamps)
  --compiler (-c)            specify the compiler type
  --parallel (-j)            number of parallel build jobs
  --swig-cpp                 make SWIG create C++ files (default is C)
  --swig-opts                list of SWIG command line options
  --swig                     path to the SWIG executable
  --user                     add user include, library and rpath
  --cython-cplus             generate C++ source files
  --cython-create-listing    write errors to a listing file
  --cython-line-directives   emit source line directives
  --cython-include-dirs      path to the Cython include files (separated by
                             ':')
  --cython-c-in-temp         put generated C files in temp directory
  --cython-gen-pxi           generate .pxi file for public declarations
  --cython-directives        compiler directive overrides
  --cython-gdb               generate debug information for cygdb
  --cython-compile-time-env  cython compile time environment
  --pyrex-cplus              generate C++ source files
  --pyrex-create-listing     write errors to a listing file
  --pyrex-line-directives    emit source line directives
  --pyrex-include-dirs       path to the Cython include files (separated by
                             ':')
  --pyrex-c-in-temp          put generated C files in temp directory
  --pyrex-gen-pxi            generate .pxi file for public declarations
  --pyrex-directives         compiler directive overrides
  --pyrex-gdb                generate debug information for cygdb
  --help-compiler            list available compilers

usage: compile.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: compile.py --help [cmd1 cmd2 ...]
   or: compile.py --help-commands
   or: compile.py cmd --help
"""