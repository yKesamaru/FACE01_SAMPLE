#include <pybind11/pybind11.h>
# include <Python.h>
# include <iostream>

int add(int i, int j)
{
    std::cout << i * j << std::endl;
    int result = i * j * i * j;
    std::cout << result << std::endl;
    return i + j + result;
}

PYBIND11_MODULE(add_cpp, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
}