#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
        const auto &buff_info = resized_frame.request();
        const auto &shape = buff_info.shape;
        py::array_t<uint8_t> face_image{shape};

        int top = std::get<0>(face_location);
        int right = std::get<1>(face_location);
        int bottom = std::get<2>(face_location);
        int left = std::get<3>(face_location);

        if (right == 0){
            std::cout << "ERROR OCURRED!" << std::endl;  //DEBUG
            return face_image;
        }
        for (auto i = 0; i < shape[0]; i++)
        {
            if (i < top){continue;}
            if (i > bottom){continue;}
            for (auto j = 0; j < shape[1]; j++)
            {
                if (j < left){continue;}
                if (j > right){continue;}
                *face_image.mutable_data(i, j) = *resized_frame.data(i, j);
            }
        }
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
