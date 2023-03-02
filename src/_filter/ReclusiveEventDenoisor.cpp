#include <math.h>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>

#include <cblas.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "edn.hpp"

namespace py = pybind11;


edn::ReclusiveEventDenoisor::ReclusiveEventDenoisor(uint16_t sizeX, uint16_t sizeY, std::tuple<float, float, float> params) : EventDenoisor(sizeX, sizeY) {
    std::tie(nThres, sigmaT, sigmaS) = params;
    procX = ceil((float) sizeX / _MULTI_) * _MULTI_;
    procY = ceil((float) sizeY / _MULTI_) * _MULTI_;
    _LENGTH_ = procX * procY;
    sampleT = 1000. * (sigmaT / 1.);
}

bool edn::ReclusiveEventDenoisor::initStateCof() {
    for (size_t i = 0; i < _POLES_; i++) {
        expmAB[i] = *(expmA + i * _POLES_);
    }
    return true;
}

bool edn::ReclusiveEventDenoisor::initDericheCof() {
    float scale = 1.0000 / (sqrt(2 * M_PI) * sigmaS);
    
    /* parameters */
    float a0 = 1.6800 * scale, a1 = -0.6803 * scale;
    float b0 = 3.7350 * scale, b1 = -0.2598 * scale;
    float w0 = 0.6319, w1 =  1.9970;
    float k0 = -1.783, k1 = -1.7230;

    n0 = a1 + a0;
    n1 = exp(k1/sigmaS) * (b1*sin(w1/sigmaS)-(a1+2*a0)*cos(w1/sigmaS)) + exp(k0/sigmaS) * (b0*sin(w0/sigmaS) - (a0+2*a1) * cos(w0/sigmaS));
    n2 = 2 * exp((k0+k1)/sigmaS) * ((a0+a1)*cos(w1/sigmaS)*cos(w0/sigmaS) - b0*cos(w1/sigmaS)*sin(w0/sigmaS) - b1*cos(w0/sigmaS)*sin(w1/sigmaS)) + a1 * exp(2*k0/sigmaS) + a0 * exp(2*k1/sigmaS);
    n3 = exp((k1+2*k0)/sigmaS) * (b1*sin(w1/sigmaS)-a1*cos(w1/sigmaS)) + exp((k0+2*k1)/sigmaS) * (b0*sin(w0/sigmaS) - a0*cos(w0/sigmaS));

    d1 = -2 * exp(k1/sigmaS) * cos(w1/sigmaS) - 2 * exp(k0/sigmaS) * cos(w0/sigmaS);
    d2 =  4 * exp((k0+k1)/sigmaS) * cos(w1/sigmaS) * cos(w0/sigmaS) + exp(2*k1/sigmaS) + exp(2*k0/sigmaS);
    d3 = -2 * exp((k0+2*k1)/sigmaS) * cos(w1/sigmaS) - 2 * exp((k1+2*k0)/sigmaS) * cos(w1/sigmaS);
    d4 =  1 * exp(2*(k0+k1)/sigmaS);

    m1 = n1 - d1 * n0;
    m2 = n2 - d2 * n0;
    m3 = n3 - d3 * n0;
    m4 = -d4 * n0;

    return true;
}

void edn::ReclusiveEventDenoisor::updateStateSpace(float *Xt, float *Yt, float* Ut) {
    // initialize
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

void edn::ReclusiveEventDenoisor::fastDericheBlur(float *Yt) {
    int stepX = procX / _MULTI_;
    int stepY = procY / _MULTI_;

    int *index = (int *) calloc(_MULTI_, sizeof(int));
    float *tmpRe = (float *) calloc(_MULTI_, sizeof(float));
    float *tmpYt = (float *) calloc(_LENGTH_, sizeof(float));

    __m256i mIndex;
    __m256 mPrevIn1, mPrevIn2, mPrevIn3, mPrevIn4;
    __m256 mPrevOut1, mPrevOut2, mPrevOut3, mPrevOut4;

    __m256 mCurIn, mSumIn, mCurOut, mSumOut;
    __m256 mSumN0, mSumN1, mSumN2, mSumN3;
    __m256 mSumD1, mSumD2, mSumD3, mSumD4;

    // from left to right
    for (size_t idx = 0; idx < stepX; idx++) {
        for (size_t th = 0; th < _MULTI_; th++) {
            index[th] = (idx * _MULTI_ + th) * procY;
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
        
        for (size_t idy = 0; idy < procY; idy++) {
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
            for (size_t k = 0; k < _MULTI_; k++) {
                *(tmpYt + index[k] + idy) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(1));
        }
    }

    // from right to left
    for (size_t idx = 0; idx < stepX; idx++) {
        for (size_t th = 0; th < _MULTI_; th++) {
            index[th] = (idx * _MULTI_ + th + 1) * procY - 1;
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
        
        for (size_t idy = 0; idy < procY; idy++) {
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
            for (size_t k = 0; k < _MULTI_; k++) {
                *(tmpYt + index[k] - idy) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-1));
        }
    }

    // from top to bottom
    for (size_t idy = 0; idy < stepY; idy++) {
        for (size_t th = 0; th < _MULTI_; th++) {
            index[th] = idy * _MULTI_ + th;
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
        
        for (size_t idx = 0; idx < procX; idx++) {
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
            for (size_t k = 0; k < _MULTI_; k++) {
                *(Yt + index[k] + idx * procY) = tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(procY));
        }
    }

    // from bottom to top
    for (size_t idy = 0; idy < stepY; idy++) {
        for (size_t th = 0; th < _MULTI_; th++) {
            index[th] = (procX - 1) * procY + idy * _MULTI_ + th;
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
        
        for (size_t idx = 0; idx < procX; idx++) {
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
            for (size_t k = 0; k < _MULTI_; k++) {
                *(Yt + index[k] - idx * procY) += tmpRe[k];
            }

            // step
            mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
            mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
            mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-procY));
        }
    }

    free(index);
    free(tmpRe);
    free(tmpYt);

    return;
}

py::array_t<bool> edn::ReclusiveEventDenoisor::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    bool initState = initStateCof();
    bool initDeriche = initDericheCof();
    
    uint64_t t0 = *ptrts;
    float *Xt = (float *) calloc(_POLES_ * _LENGTH_, sizeof(float));
    float *Yt = (float *) calloc(1       * _LENGTH_, sizeof(float));
    float *Ut = (float *) calloc(1       * _LENGTH_, sizeof(float));

    for (size_t i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
        uint32_t ind = event.x * procY + event.y;

        if (event.ts - t0 >= sampleT) {
            updateStateSpace(Xt, Yt, Ut);
            fastDericheBlur(Yt);
            t0 = event.ts;
        }
        
        Ut[ind] += 1;
        if (Yt[ind] + Ut[ind] > nThres) {
            vec[i] = true;
        }
    }

    free(Xt);
    free(Yt);
    free(Ut);

    return py::cast(vec);
}
