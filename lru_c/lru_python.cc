#include <pybind11/pybind11.h>
#include "lru.hpp"
#include <pybind11/stl.h>
void fl(py::list pl) {
    size_t len = pl.size();
    for (auto i = 0; i < len; ++i) {
        pl[i] = pl[i].cast<int>() + 1;
    }
    pl.append(py::make_tuple("fuckyou", 123));
}
PYBIND11_MODULE(lru_utils, m) {
    py::class_<LRU>(m, "LRU")
    .def(py::init<int>())
    .def("try_get", &LRU::TryGet)
    .def("get", &LRU::Get)
    .def("rollback_steps", &LRU::Rollback)
    .def("keys", &LRU::Keys)
    .def("__contains__", &LRU::Contains)
    .def("view", &LRU::View)
    .def("values", &LRU::Values)
    .def("state_dict", &LRU::state_dict)
    .def("restore", &LRU::restore);

    m.def("fl", &fl, "modify list");
}
