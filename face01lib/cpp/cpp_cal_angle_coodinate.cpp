# include <iostream>
# include <string.h>
# include <pybind11/pybind11.h>
# include <Python.h>

using namespace std;
namespace py = pybind11;

py::tuple return_tuple(int height, int width){
    std::tuple<int, int, int, int> top_left = std::tuple<int, int, int, int>(0, int(height / 2), 0, int(width / 2));
    std::tuple<int, int, int, int> top_right = std::tuple<int, int, int, int>(0, int(height / 2), int(width / 2), width);
    std::tuple<int, int, int, int> bottom_left = std::tuple<int, int, int, int>(int(height / 2), height, 0, int(width / 2));
    std::tuple<int, int, int, int> bottom_right = std::tuple<int, int, int, int>(int(height / 2), height, int(width / 2), width);
    std::tuple<int, int, int, int> center = std::tuple<int, int, int, int>(int(height / 4), int(height / 4) * 3, int(width / 4), int(width / 4) * 3);
    py::tuple tup = py::make_tuple(top_left, top_right, bottom_left, bottom_right, center);
    return tup;
}
int main(int argc, char* argv[]){
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    return_tuple(height, width);
    return 0;
}

PYBIND11_MODULE(cpp_cal_angle_coodinate, m)
{
    m.def("return_tuple", &return_tuple, "return tuple");
}