#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <boost/python/numpy.hpp>
#include "boost/tuple/tuple.hpp"
# include <iostream>
# include <string>  //std::to_string()
namespace py = boost::python;
namespace np = boost::python::numpy;

int a, b, c, d;
// auto size(const char *str, Eigen::MatrixXd &frame, py::tuple &pyTuple)
auto size(py::tuple shape, py::tuple strides,  char *str, np::ndarray &frame, py::tuple &pyTuple)
{
    const int frame_rows = py::extract<int>(shape[0]);
    const int frame_cols = py::extract<int>(shape[1]);
    const int frame_ndim = py::extract<int>(shape[2]);
    std::cout << "python_shape: ("<< frame_rows << "," << frame_cols << "," <<frame_ndim << ")" << std::endl;
    const int stride_0 = py::extract<int>(strides[0]);
    const int stride_1 = py::extract<int>(strides[1]);
    const int stride_2 = py::extract<int>(strides[2]);
    std::cout << "python_stride: (" << stride_0 << "," << stride_1 << "," << stride_2 << ")" << std::endl;

    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    const Stride frame_stride(stride_0,stride_1);
    
    // std::cout << str << std::endl;
    // auto N = frame.shape(0);
    // std::cout << N << std::endl;

    const auto
        frame_rows_ = frame.shape(0),
        frame_cols_ = frame.shape(1),
        frame_ndim_ = frame.shape(2);
    std::cout << "origin rows,cols: (" << frame_rows_ << "," << frame_cols_ << "," << frame_ndim_ << ")" << std::endl;
    // const Stride
    //     frame_stride_(frame.strides(0) / sizeof(double), frame.strides(1) / sizeof(double));
    // std::cout << frame_stride_.inner << std::endl;
    int frame_stride_0 = frame.strides(0) / sizeof(double);
    int frame_stride_1 = frame.strides(1) / sizeof(double);
    int frame_stride_2 = frame.strides(2) / sizeof(double);
    std::cout << "original frame stride: (" << frame_stride_0 << "," << frame_stride_1 << "," << frame_stride_2 << ")" << std::endl;

    const Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Stride>
        frame_mat(reinterpret_cast<double *>(frame.get_data()), frame_rows, frame_cols, frame_stride);
    np::ndarray ret = np::empty(
        py::make_tuple(frame_mat.rows(), frame_mat.cols()),
        np::dtype::get_builtin<double>());
    Eigen::Map<Eigen::MatrixXd>
        ret_mat(reinterpret_cast<double *>(ret.get_data()), frame_mat.rows(),frame_mat.cols());
    // std::cout << ret_mat << std::endl;

    // frame = frame[TOP_LEFT[0]:TOP_LEFT[1],TOP_LEFT[2]:TOP_LEFT[3]]
    int val0 = py::extract<int>(pyTuple[0]);
    int val1 = py::extract<int>(pyTuple[1]);
    int val2 = py::extract<int>(pyTuple[2]);
    int val3 = py::extract<int>(pyTuple[3]);
    std::cout << "TOP_LEFT: (" << val0 << "," << val1 << "," << val2 << "," << val3 << ")" << std::endl;
}

BOOST_PYTHON_MODULE(sample)
{
    Py_Initialize();
    np::initialize();
    py::def("size", size);
}

int main(){
    return 0;
}