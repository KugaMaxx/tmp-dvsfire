#ifndef EDN_H
#define EDN_H

#include "dvs.hpp"

#include <vector>
#include <stdlib.h>
#include <iostream>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

namespace edn {
    class EventDenoisor {
    protected:
        int32_t sizeX;
        int32_t sizeY;
        uint32_t evlen;  // Length of noise events
        
        bool *ptrp;
        uint16_t *ptrx;
        uint16_t *ptry;
        uint64_t *ptrts;
        
    public:
        EventDenoisor(uint16_t sizeX, uint16_t sizeY) : sizeX(sizeX), sizeY(sizeY) {};
        virtual ~EventDenoisor() {};
        std::vector<bool> initialize(dvs::arrTs ts, dvs::arrX x, dvs::arrY y, dvs::arrP p) {
            py::buffer_info bufp = p.request(), bufx = x.request(), bufy = y.request(), bufts = ts.request();
	        assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);
            
            evlen = bufts.size;
            ptrts = static_cast<uint64_t *> (bufts.ptr);
            ptrx  = static_cast<uint16_t *> (bufx.ptr);
            ptry  = static_cast<uint16_t *> (bufy.ptr);
            ptrp  = static_cast<bool *> (bufp.ptr);

	        std::vector<bool> vec(evlen, false);

	        return vec;
        }
    };

    class ReclusiveEventDenoisor : public EventDenoisor {
    private:
        int32_t procX; // multiple integral of (sizeX / _MULTI_)
        int32_t procY; // multiple integral of (sizeY / _MULTI_)

        float sigmaT;
        float sigmaS;
        float nThres;
        size_t _LENGTH_;

        /* temporal state filter */
        static const size_t _POLES_ = 4;
        float sampleT;
        float A[_POLES_ * _POLES_] = {
            -0.348174256797851, -0.101759749190219, -0.0132846052724056, -0.000841996822694814, 
             1.               ,  0.               ,  0.                ,  0.                  ,
             0.               ,  1.               ,  0.                ,  0.                  ,
             0.               ,  0.               ,  1.                ,  0.                  ,
        };
        float expmA[_POLES_ * _POLES_] = {
             0.663969918056621, -0.090449028116203, -0.0113960589465566, -0.000698661633195582,
             0.829767541116738,  0.952873614999921, -0.0060120912459797, -0.000372924694966144,
             0.442905109514069,  0.983975698453770,  0.9979435278591400, -0.000128271692953740,
             0.152342252959110,  0.495946760217017,  0.9994780079059620,  0.999967334556010000,
        };
        float B[_POLES_ * 1      ] = {1., 0., 0., 0.};
        float C[1       * _POLES_] = {-0.0347802770578387, 0.0361368959533555, -0.0166748570101696, 0.00531742216408524};
        float expmAB[1  * _POLES_];

        /* spatial deriche filter */
        static const size_t _MULTI_ = 8;
        float n0, n1, n2, n3;
        float d1, d2, d3, d4;
        float m1, m2, m3, m4;

        /* private function */
        bool initStateCof();
        bool initDericheCof();

    public:
        ReclusiveEventDenoisor(uint16_t sizeX, uint16_t sizeY, std::tuple<float, float, float> params);
        void updateStateSpace(float *Xt, float *Yt, float *Ut);
        void fastDericheBlur(float *Yt);
        py::array_t<bool> run(dvs::arrTs ts, dvs::arrX x, dvs::arrY y, dvs::arrP p);
    };
}

#endif
