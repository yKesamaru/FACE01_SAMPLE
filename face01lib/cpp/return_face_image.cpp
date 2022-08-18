#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// # include <string.h>
// # include <tuple>
// # include <Eigen/Core>
// # include <pybind11/stl_bind.h>

namespace py = pybind11;
    
class Return_face_image
{
public:
    // コンストラクタ
    Return_face_image(){}

    py::array_t<uint8_t> return_face_image(
        const py::array_t<uint8_t> &resized_frame,
        const std::tuple<int, int, int, int> &face_location)
    {
        // const auto &buff_info = resized_frame.request();
        // const auto &shape = buff_info.shape;

        int top = std::get<0>(face_location);
        int right = std::get<1>(face_location);
        int bottom = std::get<2>(face_location);
        int left = std::get<3>(face_location);

        // py::array_t<uint8_t> face_image{shape};  // 縮小せねば。
        int height = bottom - top;
        int width = right - left;
        py::array_t<uint8_t> face_image;
        face_image.resize({height, width, 3});

        // std::cout << *shape << std::endl;

        // py::print("shape = ", to_string(&shape, resized_frame.ndim()));

        if (right == 0)
        {
            std::cout << "ERROR OCURRED!" << std::endl;  //DEBUG
            return face_image;
        }

        std::cout << "top: " << top << std::endl;
        std::cout << "right: " << right << std::endl;
        std::cout << "bottom: " << bottom << std::endl;
        std::cout << "left: " << left << std::endl;

        // *face_image = *resized_frame[py::slice(top,bottom, 1), py::slice(left, right, 1), py::slice(0,3,1)];
        // *face_image = *resized_frame[py::make_tuple(py::slice(bottom, top, 1), py::slice(right, left, 1), py::slice(0,3,1))];
        // *face_image = *resized_frame[py::make_tuple(py::slice(top, bottom, 1), py::slice(left, right, 1), py::slice(0,255,1))];
        *face_image = 
        *resized_frame[py::make_tuple( py::slice(top,bottom, 1), py::slice(left,right,1),py::slice(0,3, 1))];

        return face_image;
    }
};

PYBIND11_MODULE(return_face_image, m)
{
    py::class_<Return_face_image>(m, "Return_face_image")
        .def(py::init<>())
        .def(
            "return_face_image",
            &Return_face_image::return_face_image
            )
        .def("__init__", [](const Return_face_image &)
            { return "<Return_face_image>"; });
};
