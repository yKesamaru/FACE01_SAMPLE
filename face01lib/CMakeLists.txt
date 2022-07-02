message("\n")
message("--------------------------")
cmake_minimum_required(VERSION 3.16)
project(お手前cmake VERSION 0.1.0)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# コンパイラー Option
set(CMAKE_CXX_FLAGS "-O1 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Og -g")

# venvを使う場合
# PyBind11 を取得
set(pybind11_DIR "/home/terms/bin/FACE01/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
message("pybind11はみつかりましたか？: " ${pybind11_FOUND})

# Pythonのパスを探索
set(Python_VIRTUALENV FIRST)
# set(Python_EXECUTABLE "/home/terms/bin/FACE01/bin/python3")
set(Python_EXECUTABLE $(python3 -c "import sys; print(sys.executable)"))
find_package(Python REQUIRED)
message("Pythonはみつかりましたか？: " ${Python_FOUND})
message("今動いているPythonは？: " ${Python_EXECUTABLE})


# -I : インクルードパス
include_directories(/usr/include)
include_directories(/home/terms/bin/FACE01/lib/python3.8/site-packages/pybind11/include)
include_directories(/home/terms/bin/FACE01/include)
# include_directories(/usr/include/python3.8)

# -L : ライブラリ探索パス
link_directories(/home/terms/bin/FACE01/lib/python3.8/site-packages)

# 作成するモジュール名
pybind11_add_module(add_cpp MODULE  add_cpp.cpp)

message("--------------------------")
# 備考
# cmake [<options>] -S <path-to-source> -B <path-to-build>


