#ifndef DVS_H
#define DVS_H

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

namespace dvs {
    struct Event {
        uint64_t ts;
        int16_t   x;
        int16_t   y;
        int8_t    p;
        
        Event(uint64_t ts_, uint16_t x_, uint16_t y_, bool p_) : ts(ts_), x(x_), y(y_), p(2 * p_ - 1) {}
    };

    typedef py::array_t<uint64_t> arrTs;
    typedef py::array_t<uint16_t> arrX;
    typedef py::array_t<uint16_t> arrY;
    typedef py::array_t<bool>     arrP;
}

#endif
