#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
/*
https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#numpy
https://stackoverflow.com/questions/61821844/how-to-express-thisa03-03-in-python-as-cpybind11
*/
namespace py = pybind11;
    
class Return_face_image
{
public:
    // コンストラクタ
    Return_face_image(){}

    py::array_t<uint8_t> return_face_image(
        const py::array resized_frame,
        const std::tuple<int, int, int, int> &face_location)
    {
        int top = std::get<0>(face_location);
        int right = std::get<1>(face_location);
        int bottom = std::get<2>(face_location);
        int left = std::get<3>(face_location);
        int step = 1;

        py::array face_image = 
        resized_frame[py::make_tuple( py::slice(top,bottom, step), py::slice(left,right,step),py::slice(0,3, step))];

        return face_image;
    }
};

PYBIND11_MODULE(return_face_image, m)
{
    py::class_<Return_face_image>(m, "Return_face_image")
        .def(py::init<>())
        .def(
            "return_face_image",
            &Return_face_image::return_face_image)
        .def("__init__", [](const Return_face_image &)
             { return "<Return_face_image>"; });
};
