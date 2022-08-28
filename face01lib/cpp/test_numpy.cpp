#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

// template <typename T>
// using RMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;
// using RMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> m;
// template <typename T>
// RMatrix<T> test_numpy(RMatrix<T> m, const std::tuple<int, int, int, int> &start_stop)
Eigen::Matrix<double> test_numpy(Eigen::Matrix<double> m, const std::tuple<int, int, int, int> &start_stop)
{
    std::cout << m << std::endl;
    return m ;
}

PYBIND11_MODULE(test_numpy, m)
{
    m.doc() = "my test module";
    m.def("test_numpy", &test_numpy<double>, "");
}