#include <iostream>
#include <fmt/core.h>
#include <utility>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
# include <tuple>

namespace py = pybind11;

class Return_face_image
{
private:
    std::tuple<int, int, int, int> face_location = std::make_tuple(0, 0, 0, 0);
    int top, right, bottom, left;

public:
    py::array_t<uint8_t> return_face_image(
        py::array_t<uint8_t> &resized_frame,
        face_location
    )
    {
        const auto &buff_info = resized_frame.request();
        const auto &shape = buff_info.shape;

        int {top, right, bottom, left} = face_location;
        // int right = *face_location[1];
        // int bottom = *face_location[2];
        // int left = *face_location[4];

        py::array_t<uint8_t> face_image{shape};

        std::cout << "C++" << std::endl;
        for (auto i = 0; i < shape[0]; i++)
        {
            for (auto j = 0; j < shape[1]; j++)
            {
                auto v = *resized_frame.data(i, j);
                std::cout << fmt::format("x[{}, {}] = {}", i, j, v) << std::endl;
            }
        }
    }
};

PYBIND11_MODULE(return_face_image, m)
{
    py::class_<Return_face_image>("Return_face_image", m)
        .def(
            "return_face_image",
            &Return_face_image::return_face_image,
            "",
            py::arg("resized_frame"),
            py::arg("face_location"));
}
