#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
/*
https://stackoverflow.com/questions/61821844/how-to-express-thisa03-03-in-python-as-cpybind11
*/

namespace py = pybind11;
class Test_numpy
{
public:
    Test_numpy() {}
    py::array_t<uint8_t> test_numpy(
        const py::array dim3,
        const std::tuple<int, int, int, int> &start_stop)
    {

        int zero = std::get<0>(start_stop);
        int one = std::get<1>(start_stop);
        int two = std::get<2>(start_stop);
        int three = std::get<3>(start_stop);
        int step = 1;
        py::array return_dim3 = dim3[py::make_tuple(py::slice(zero, one, step), py::slice(two, three, step), py::slice(zero, one, step))];
        return return_dim3;
    }
};
PYBIND11_MODULE(test_numpy, m)
{
    py::class_<Test_numpy>(m, "Test_numpy")
        .def(py::init<>())
        .def(
            "test_numpy",
            &Test_numpy::test_numpy)
        .def("__init__", [](const Test_numpy &)
             { return "<Test_numpy>"; });
};
