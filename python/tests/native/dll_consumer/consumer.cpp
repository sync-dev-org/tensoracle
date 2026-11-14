#include <Python.h>
#include <tensoracle/tensoracle.h>

static PyObject* version(PyObject*, PyObject*) {
    tensoracle_version value{sizeof(tensoracle_version)};
    if (tensoracle_get_version(&value, nullptr) != TENSORACLE_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "tensoracle_get_version failed");
    }
    return Py_BuildValue("(III)", value.major, value.minor, value.patch);
}

static PyMethodDef methods[] = {
    {"version", version, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_dll_consumer", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__dll_consumer() { return PyModule_Create(&module); }
