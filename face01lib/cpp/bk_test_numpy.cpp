#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// # include <string.h>
// # include <tuple>
// # include <Eigen/Core>
// # include <pybind11/stl_bind.h>

/*
https://stackoverflow.com/questions/61821844/how-to-express-thisa03-03-in-python-as-cpybind11
*/

namespace py = pybind11;

class Test_numpy
{
public:
    // コンストラクタ
    Test_numpy() {}

    py::array_t<py::int_> test_numpy(
        py::array_t<py::int_> dim2,
        // const py::array_t<uint8_t> &dim3,
        const std::tuple<int, int, int, int> &start_stop)
    {
        const auto &buff_info = dim2.request();
        // const auto &buff_info = dim3.request();
        const auto &shape = buff_info.shape;
        // const auto &strides = buff_info.strides;

        // py::int_ a, b, c, d,e;
        // py::int_ *zero = &a;
        // py::int_ *one = &b;
        // py::int_ *two = &c;
        // py::int_ *three = &d;
        // py::int_ *step = &e;

        py::int_ zero = std::get<0>(start_stop);
        py::int_ one = std::get<1>(start_stop);
        py::int_ two = std::get<2>(start_stop);
        py::int_ three = std::get<3>(start_stop);
        py::int_ step = 1;
        // py::int_ zero = std::get<0>(start_stop);
        // py::int_ one = std::get<1>(start_stop);
        // py::int_ two = std::get<2>(start_stop);
        // py::int_ three = std::get<3>(start_stop);
        // py::int_ step = 1;
        // *zero = 0;
        // *one = 3;
        // *two = 0;
        // *three = 3;
        // *step = 1;

        // std::cout << "zero: " << zero << std::endl;
        // std::cout << "one: " << one << std::endl;
        // std::cout << "two: " << two << std::endl;
        // std::cout << "three: " << three << std::endl;
        // std::cout << "zero: " << *zero << std::endl;
        // std::cout << "one: " << *one << std::endl;
        // std::cout << "two: " << *two << std::endl;
        // std::cout << "three: " << *three << std::endl;

        int axis0 = shape[0];
        int axis1 = shape[1];
        // int axis2 = shape[2];
        std::cout << "axis0: " << axis0 << std::endl;
        std::cout << "axis1: " << axis1 << std::endl;
        // std::cout << "axis2: " << axis2 << std::endl;

        // auto return_dim2;
        // py::array_t<py::int_, py::array::c_style> return_dim2({3,3});
        // py::array_t<int, py::array::c_style> return_dim2({int(one), int(three)});
        // py::array_t<uint8_t> return_dim3({int(one), int(three), axis2});
        // py::array_t<uint8_t> return_dim3({3, 3, axis2});
        // py::array_t<uint8_t, py::array::c_style> return_dim3({3, 3, axis2});

        // return_dim2.resize({3, 3});
        // return_dim3.reshape({3,3,3});
        // return_dim3.reshape({axis0, axis1, axis2});
        // std::cout << "writable: " << return_dim3.writeable() << std::endl;
        // std::cout << "writable: " << return_dim2.writeable() << std::endl;

        // return_dim3.resize({one - zero, three - two, axis2});

        // if (one == 0)
        // {
        //     std::cout << "ERROR OCURRED!" << std::endl; // DEBUG
        //     return return_dim3;
        // }
        // for (uint_t i = 0; i < return_dim3.size(); i++)
        // {
        //     std::cout << "i: " << i << std::endl;
        //     std::cout << return_dim3.at(i) << ",";
        //     std::cout << std::endl;
        // }

        // *return_dim3 = *dim3[py::slice(zero,two, 1), py::slice(three, one, 1), py::slice(0,3,1)];
        // *return_dim3 = *dim3[py::make_tuple(py::slice(two, zero, 1), py::slice(one, three, 1), py::slice(0,3,1))];
        // *return_dim3 = *dim3[py::make_tuple(py::slice(zero, two, 1), py::slice(three, one, 1), py::slice(0,axis2,1))];

// *return_dim3 = *dim3[py::make_tuple(py::slice(zero, one, step), py::slice(two, three, step), py::slice(zero, one, step))];
// *return_dim3 = *dim3[py::make_tuple(py::slice(3, 0, 1), py::slice(3, 0, 1), py::slice(zero, one, 1))];
// *return_dim3 = *dim3[py::make_tuple(py::slice(*zero, *one, *step), py::slice(*two, *three, *step), py::slice(*zero, *one, *step))];
// *return_dim3 = *dim3[py::make_tuple(py::slice(zero, one, 1), py::slice(two, three, 1), py::slice(zero, one, 1))];
// *return_dim3 = *dim3[py::make_tuple(py::slice(zero, one, 1), py::slice(two, three, 1), py::slice(zero, one, 1))];
// *return_dim2 = *dim2[py::make_tuple(py::slice(zero, one, step), py::slice(two, three, step))];
// std::cout << "*dim2: " << std::endl;
// std::cout << "dim2: " << dim2 << std::endl;
// return_dim2 = dim2;
auto return_dim2 = dim2[py::make_tuple(py::slice(0, 3, 1), py::slice(0, 3, 1))];
// return_dim2 = dim2;

// return dim2;
// return dim3;
// return return_dim3;
return return_dim2;
    }
};

PYBIND11_MODULE(test_numpy, m)
{
    py::class_<Test_numpy>(m, "Test_numpy")
        .def(py::init<>())
        .def(
            "test_numpy",
            &Test_numpy::test_numpy<py::int_>)
        .def("__init__", [](const Test_numpy &)
             { return "<Test_numpy>"; });
};

// template <typename py::int_>
// py::array_t<py::int_> main(
//     py::array_t<py::int_> dim2,
//     const std::tuple<int, int, int, int> &start_stop)
// {
//     result = Test_numpy::Test_numpy(
//         py::array_t<py::int_> dim2,
//     const std::tuple<int, int, int, int> &start_stop
//     )
// }