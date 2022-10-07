#include <pybind11/pybind11.h>
# include <Python.h>

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(cal_specify_date, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}