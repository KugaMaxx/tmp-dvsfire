#include "edn.hpp"
#include <array>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;


std::vector<bool> edn::EventDenoisor::initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	py::buffer_info bufp = arrp.request(), bufx = arrx.request(), bufy = arry.request(), bufts = arrts.request();
	assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);

	evlen = bufts.size;

	ptrts = static_cast<uint64_t *> (bufts.ptr);
	ptrx  = static_cast<uint16_t *> (bufx.ptr);
	ptry  = static_cast<uint16_t *> (bufy.ptr);
	ptrp  = static_cast<bool *> (bufp.ptr);

	std::vector<bool> vec(evlen, false);

	return vec;
}

PYBIND11_MODULE(eventdenoisor, m)
{
	py::class_<edn::ReclusiveEventDenoisor>(m, "reclusive_event_denoisor")
		.def(py::init<uint16_t, uint16_t, std::tuple<float, float, float>>())
		.def("run", &edn::ReclusiveEventDenoisor::run);
}
