# include <Eigen/Core>
# include <Eigen/SparseCore>
#include <Python.h>
#include <Eigen/Dense>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <string.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

// [Eigen and numpy](https: //qiita.com/lucidfrontier45/items/5048ef74fbf32eeb9f08)
// 別名の型を作る
template <typename T> 
using RMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;


// template  <typename T>
void modify_frame(
    // string set_area,
    Eigen::Ref<RMatrix<uint8_t>> frame,
    tuple<int,int,int,int> top_left,
    tuple<int,int,int,int> top_right,
    tuple<int,int,int,int> bottom_left,
    tuple<int,int,int,int> bottom_right,
    tuple<int,int,int,int> center)
{
    // if (set_area == string("TOP_LEFT"))
        // frame_range = frame [top_left[0]:top_left[1], top_left[2]:top_left[3]];
        // cout << "set_area: " << set_area << endl;
        cout << "frame: " << frame << endl;
        cout << "top_left[0]: " << std::get<0>(top_left) << endl;
};

int main(int argc, char *argv[])
{
    uint8_t frame;
    tuple<int, int, int, int> top_left;
    tuple<int, int, int, int> top_right;
    tuple<int, int, int, int> bottom_left;
    tuple<int, int, int, int> bottom_right;
    tuple<int, int, int, int> center;
    modify_frame(frame, top_left,top_right,bottom_left,bottom_right,center);
    return 0;
}

// PYBIND11_MODULE(cpp_angle_of_view_specification, m)
// {
//     m.doc() = "my test module";
//     m.def("modify_frame", &modify_frame, "This is test.");
// }