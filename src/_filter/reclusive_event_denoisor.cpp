#include "reclusive_event_denoisor.hpp"


namespace py = pybind11;

namespace edn {

void ReclusiveEventDenoisor::setCoefficient() {
    /* initialize state space filter parameters */
    samplarT = samplarT * (sigmaT / 1.);
    for (size_t i = 0; i < _POLES_; i++) {
        expmAB[i] = *(expmA + i * _POLES_);
    }

    /* initialize deriche blur filter parameters */
    float_t scale = 1.0000 / (sqrt(2 * M_PI) * sigmaS);
    a0 *= scale, a1 *= scale, b0 *= scale, b1 *= scale;

    n0 = a1 + a0;
    n1 = exp(k1/sigmaS)          * (b1*sin(w1/sigmaS)-(a1+2*a0)*cos(w1/sigmaS)) + exp(k0/sigmaS) * (b0*sin(w0/sigmaS) - (a0+2*a1) * cos(w0/sigmaS));
    n2 = 2 * exp((k0+k1)/sigmaS) * ((a0+a1)*cos(w1/sigmaS)*cos(w0/sigmaS) - b0*cos(w1/sigmaS)*sin(w0/sigmaS) - b1*cos(w0/sigmaS)*sin(w1/sigmaS)) + a1 * exp(2*k0/sigmaS) + a0 * exp(2*k1/sigmaS);
    n3 = exp((k1+2*k0)/sigmaS)   * (b1*sin(w1/sigmaS)-a1*cos(w1/sigmaS)) + exp((k0+2*k1)/sigmaS) * (b0*sin(w0/sigmaS) - a0*cos(w0/sigmaS));

    d1 = -2 * exp(k1/sigmaS)        * cos(w1/sigmaS)                    - 2 * exp(k0/sigmaS) * cos(w0/sigmaS);
    d2 =  4 * exp((k0+k1)/sigmaS)   * cos(w1/sigmaS) * cos(w0/sigmaS)   + exp(2*k1/sigmaS) + exp(2*k0/sigmaS);
    d3 = -2 * exp((k0+2*k1)/sigmaS) * cos(w1/sigmaS)                    - 2 * exp((k1+2*k0)/sigmaS) * cos(w1/sigmaS);
    d4 =  1 * exp(2*(k0+k1)/sigmaS);

    m1 = n1 - d1 * n0;
    m2 = n2 - d2 * n0;
    m3 = n3 - d3 * n0;
    m4 = -d4 * n0;

    return;
}

void ReclusiveEventDenoisor::updateStateSpace(float *Xt, float *Yt, float* Ut) {
    float *expmABU = (float *) calloc(_POLES_ * _LENGTH_, sizeof(float));
    
    // Yt = C * Xt
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, _LENGTH_, _POLES_, 1., C, _POLES_, Xt, _LENGTH_, 0., Yt, _LENGTH_);
    
    // expmABU = expm(A) * B * Ut
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                _POLES_, _LENGTH_, 1, 1., expmAB, 1, Ut, _LENGTH_, 0., expmABU, _LENGTH_);

    // expmABU = expm(A) * Xt + expmABU
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                _POLES_, _LENGTH_, _POLES_, 1., expmA, _POLES_, Xt, _LENGTH_, 1., expmABU, _LENGTH_);
    
    // after-processing
    cblas_scopy(_POLES_ * _LENGTH_, expmABU, 1, Xt, 1);
    memset(Ut, 0, _LENGTH_ * sizeof(*Ut));
    free(expmABU);

    return;
}

void ReclusiveEventDenoisor::fastDericheBlur(float *Yt) {
    int32_t *index = (int32_t *) calloc(_THREAD_, sizeof(int32_t));
    float_t *tmpRe = (float_t *) calloc(_THREAD_, sizeof(float_t));
    float_t *tmpYt = (float_t *) calloc(_LENGTH_, sizeof(float_t));

    __m256i mIndex;
    __m256 mPrevIn1, mPrevIn2, mPrevIn3, mPrevIn4;
    __m256 mPrevOut1, mPrevOut2, mPrevOut3, mPrevOut4;

    __m256 mCurIn, mSumIn, mCurOut, mSumOut;
    __m256 mSumN0, mSumN1, mSumN2, mSumN3;
    __m256 mSumD1, mSumD2, mSumD3, mSumD4;

    // from left to right
    for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
        for (size_t th = 0; th < _THREAD_; th++) {
            index[th] = (idx * _THREAD_ + th) * sizeY;
        }
        mIndex = _mm256_loadu_si256((__m256i *) index);

        mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

        mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        
        for (size_t idy = 0; idy < sizeY; idy++) {
            // In = image
            mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
            // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
            mSumN0 = _mm256_mul_ps(mCurIn,   _mm256_set1_ps(n0));
            mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
            mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
            mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
            mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
            // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
            mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
            mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
            mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
            mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
            mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
            // Out = PreIn - PreOut
            mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

            _mm256_storeu_ps((float *) tmpRe, mCurOut);
            for (size_t k = 0; k < _THREAD_; k++) {
                *(tmpYt + index[k] + idy) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(1));
        }
    }

    // from right to left
    for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
        for (size_t th = 0; th < _THREAD_; th++) {
            index[th] = (idx * _THREAD_ + th + 1) * sizeY - 1;
        }
        mIndex = _mm256_loadu_si256((__m256i *) index);

        mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

        mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        
        for (size_t idy = 0; idy < sizeY; idy++) {
            // In = image
            mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
            // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
            mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
            mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
            mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
            mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
            mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
            // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
            mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
            mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
            mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
            mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
            mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
            // Out = PreIn - PreOut
            mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

            _mm256_storeu_ps((float *) tmpRe, mCurOut);
            for (size_t k = 0; k < _THREAD_; k++) {
                *(tmpYt + index[k] - idy) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-1));
        }
    }

    // from top to bottom
    for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
        for (size_t th = 0; th < _THREAD_; th++) {
            index[th] = idy * _THREAD_ + th;
        }
        mIndex = _mm256_loadu_si256((__m256i *) index);

        mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

        mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        
        for (size_t idx = 0; idx < sizeX; idx++) {
            // In = image
            mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
            // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
            mSumN0 = _mm256_mul_ps(mCurIn,   _mm256_set1_ps(n0));
            mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
            mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
            mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
            mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
            // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
            mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
            mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
            mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
            mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
            mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
            // Out = PreIn - PreOut
            mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

            _mm256_storeu_ps((float *) tmpRe, mCurOut);
            for (size_t k = 0; k < _THREAD_; k++) {
                *(Yt + index[k] + idx * sizeY) = tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(sizeY));
        }
    }

    // from bottom to top
    for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
        for (size_t th = 0; th < _THREAD_; th++) {
            index[th] = (sizeX - 1) * sizeY + idy * _THREAD_ + th;
        }
        mIndex = _mm256_loadu_si256((__m256i *) index);

        mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

        mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
        
        for (size_t idx = 0; idx < sizeX; idx++) {
            // In = image
            mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
            // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
            mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
            mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
            mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
            mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
            mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
            // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
            mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
            mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
            mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
            mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
            mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
            // Out = PreIn - PreOut
            mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

            _mm256_storeu_ps((float *) tmpRe, mCurOut);
            for (size_t k = 0; k < _THREAD_; k++) {
                *(Yt + index[k] - idx * sizeY) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-sizeY));
        }
    }

    free(index);
    free(tmpRe);
    free(tmpYt);

    return;
}

}

// namespace kpy {

// ReclusiveEventDenoisor::ReclusiveEventDenoisor(int16_t sizeX, int16_t sizeY, std::tuple<float, float, float> params) {
//     std::tie(threshold, sigmaT, sigmaS) = params;
//     this->sizeX = ceil((float) sizeX / _THREAD_) * _THREAD_;
//     this->sizeY = ceil((float) sizeY / _THREAD_) * _THREAD_;
//     _LENGTH_ = this->sizeX * this->sizeY;
// }
    
// py::array_t<bool> ReclusiveEventDenoisor::run(kore::EventPybind input) {
//     auto inEvent = kore::toEventOutput(input);
//     setCoefficient();

//     Xt = (float_t *) calloc(_POLES_ * _LENGTH_, sizeof(float_t));
//     Yt = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));
//     Ut = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));

//     std::vector<bool> vec;
//     vec.reserve(inEvent.size());
//     for (auto &evt : inEvent){
//         uint32_t ind = evt.x() * sizeY + evt.y();
//         if (evt.timestamp() - ts_start < 0) {
//             ts_start = evt.timestamp();
//         } else if (evt.timestamp() - ts_start >= samplarT) {
//             updateStateSpace(Xt, Yt, Ut);
//             fastDericheBlur(Yt);
//             ts_start = evt.timestamp();
//         }
        
//         Ut[ind] += 1;
//         if (Yt[ind] + Ut[ind] > threshold) {
//             vec.push_back(true);
//         } else {
//             vec.push_back(false);
//         }
//     }

//     free(Xt);
//     free(Yt);
//     free(Ut);

//     return py::cast(vec);
// }

// PYBIND11_MODULE(eventdenoisor, m)
// {
//     PYBIND11_NUMPY_DTYPE(kore::Event, timestamp_, y_, x_, polarity_);
// 	py::class_<ReclusiveEventDenoisor>(m, "reclusive_event_denoisor")
// 		.def(py::init<uint16_t, uint16_t, std::tuple<float, float, float>>())
// 		.def("run", &ReclusiveEventDenoisor::run);
// }

// }

namespace kdv {

const char *ReclusiveEventDenoisor::initDescription() {
	return "Noise filter using the Reclusive filter.";
}

void ReclusiveEventDenoisor::initInputs(dv::InputDefinitionList &in) {
	in.addEventInput("events");
}

void ReclusiveEventDenoisor::initOutputs(dv::OutputDefinitionList &out) {
	out.addEventOutput("events");
}

void ReclusiveEventDenoisor::initConfigOptions(dv::RuntimeConfig &config) {
	config.add("sigmaT",    dv::ConfigOption::intOption("Time sigma /ms.",       1.2, 0.1, 10.0));
	config.add("sigmaS",    dv::ConfigOption::intOption("Spatial sigma /pixel.", 1.0, 0.1, 9.0));
	config.add("threshold", dv::ConfigOption::intOption("Threshold value.",      0.7, 0.0, 3.0));

	config.setPriorityOptions({"sigmaT", "sigmaS", "threshold"});
}

ReclusiveEventDenoisor::ReclusiveEventDenoisor () {
    sizeX = ceil((float) inputs.getEventInput("events").sizeX() / _THREAD_) * _THREAD_;
    sizeY = ceil((float) inputs.getEventInput("events").sizeY() / _THREAD_) * _THREAD_;
    _LENGTH_ = sizeX * sizeY;
}

void ReclusiveEventDenoisor::run() {
    auto inEvent  = inputs.getEventInput("events").events();
	auto outEvent = outputs.getEventOutput("events").events();

    Xt = (float_t *) calloc(_POLES_ * _LENGTH_, sizeof(float_t));
    Yt = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));
    Ut = (float_t *) calloc(1       * _LENGTH_, sizeof(float_t));

	if (!inEvent) {
		return;
	}

    for (auto &evt : inEvent){
        uint32_t ind = evt.x() * sizeY + evt.y();
        if (evt.timestamp() - ts_start < 0) {
            ts_start = evt.timestamp();
        } else if (evt.timestamp() - ts_start >= samplarT) {
            updateStateSpace(Xt, Yt, Ut);
            fastDericheBlur(Yt);
            ts_start = evt.timestamp();
        }
        
        Ut[ind] += 1;
        if (Yt[ind] + Ut[ind] > threshold) {
            outEvent << evt;
        }
    }
	outEvent << dv::commit;

    free(Xt);
    free(Yt);
    free(Ut);
}

void ReclusiveEventDenoisor::configUpdate() {
	sigmaT    = config.getInt("sigmaT");
	sigmaS    = config.getInt("sigmaS");
    threshold = config.getInt("threshold");
    setCoefficient();
}

}
