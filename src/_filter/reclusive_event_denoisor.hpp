#include "kore.hpp"
#include "denoisor.hpp"

#include <math.h>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>

#include <cblas.h>
#include <dv-sdk/module.hpp>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

namespace edn {

class ReclusiveEventDenoisor : public EventDenoisor {
public:
    float_t sigmaT    = 1.2;
    float_t sigmaS    = 1.0;
    float_t threshold = 0.7;

    static const size_t _POLES_  = 4;
    static const size_t _THREAD_ = 8;

    float_t *Xt;
    float_t *Yt;
    float_t *Ut;

    int64_t ts_start = INT64_MAX;
    float_t samplarT = 1000.;
    float_t A[_POLES_ * _POLES_] = {
        -0.348174256797851, -0.101759749190219, -0.0132846052724056, -0.000841996822694814, 
        1.               ,  0.               ,  0.                ,  0.                  ,
        0.               ,  1.               ,  0.                ,  0.                  ,
        0.               ,  0.               ,  1.                ,  0.                  ,
    };
    float_t B[_POLES_ * 1      ] = {
        1.               ,  0.               ,  0.                ,  0.                  ,
    };
    float_t C[1       * _POLES_] = {
        -0.034780277057838,  0.036136895953355, -0.0166748570101696,  0.005317422164085240,
    };
    float_t expmA[_POLES_ * _POLES_] = {
        0.663969918056621, -0.090449028116203, -0.0113960589465566, -0.000698661633195582,
        0.829767541116738,  0.952873614999921, -0.0060120912459797, -0.000372924694966144,
        0.442905109514069,  0.983975698453770,  0.9979435278591400, -0.000128271692953740,
        0.152342252959110,  0.495946760217017,  0.9994780079059620,  0.999967334556010000,
    };
    float_t expmAB[1      * _POLES_];

    float_t a0 = 1.6800, a1 = -0.6803;
    float_t b0 = 3.7350, b1 = -0.2598;
    float_t w0 = 0.6319, w1 =  1.9970;
    float_t k0 = -1.783, k1 = -1.7230;

    float_t n0, n1, n2, n3;
    float_t d1, d2, d3, d4;
    float_t m1, m2, m3, m4;

    void setCoefficient();
    
    void updateStateSpace(float *Xt, float *Yt, float *Ut);
    
    void fastDericheBlur(float *Yt);
};

}

// namespace kpy {

// class ReclusiveEventDenoisor : public edn::ReclusiveEventDenoisor {
// public:
//     ReclusiveEventDenoisor(int16_t sizeX, int16_t sizeY, std::tuple<float, float, float> params);
    
//     py::array_t<bool> run(kore::EventPybind input);
// };

// }

namespace kdv {

class ReclusiveEventDenoisor : public edn::ReclusiveEventDenoisor, public dv::ModuleBase {
public:
	static const char *initDescription();

	static void initInputs(dv::InputDefinitionList &in);

	static void initOutputs(dv::OutputDefinitionList &out);

	static void initConfigOptions(dv::RuntimeConfig &config);

    ReclusiveEventDenoisor();

	void run() override;

	void configUpdate() override;
};

}
