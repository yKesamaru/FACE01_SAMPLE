# include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class Matrix
{
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols)
    {
        m_data = new float[rows * cols];
    }
    float *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

private:
    size_t m_rows, m_cols;
    float *m_data;
};

py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
    .def_buffer([](Matrix &m) -> py::buffer_info
                { return py::buffer_info(
                      m.data(),                               /* Pointer to buffer */
                      sizeof(float),                          /* Size of one scalar */
                      py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                      2,                                      /* Number of dimensions */
                      {m.rows(), m.cols()},                   /* Buffer dimensions */
                      {sizeof(float) * m.cols(),              /* Strides (in bytes) for each index */
                       sizeof(float)}); });